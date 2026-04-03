"""
Rubric implementation for the QED Math Environment.


Grades mathematical proofs on a 0-7 scale using an LLM judge and
normalizes the score to a [0, 1] reward signal.

The grader is prompted to produce a ``<score>N</score>`` tag; that tag is
parsed and the integer score is normalized to a reward. Optional
``custom_threshold`` collapses partial-credit scores (1-5) to 1.
"""

from __future__ import annotations


import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Union

import openai

from openenv.core.rubrics.base import Rubric


def parse_schema(schema: Any) -> str:
    """Normalize a schema payload into the Markdown string used as {marking_scheme}."""
    if isinstance(schema, str):
        return schema
    if not isinstance(schema, list):
        raise TypeError(
            f"parse_schema expects a string or list of dicts, got {type(schema).__name__}"
        )

    sections: list[str] = []
    for idx, entry in enumerate(schema):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Schema entry at index {idx} must be a dict, got {type(entry).__name__}"
            )
        title = entry.get("title")
        points = entry.get("points")
        description = entry.get("desc") or entry.get("description")
        if title is None or points is None or description is None:
            raise ValueError(
                f"Schema entry at index {idx} is missing 'title', 'points', or 'desc'/'description'"
            )
        sections.append(
            f"# {title} ({points} points)\nDescription: {description}".strip()
        )

    return "\n\n".join(sections)


MAX_SCORE = 7

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BACKOFF = [15, 30, 60]

_TEMPLATE_VARS = ("{problem}", "{human_solution}", "{marking_scheme}", "{solution}")


@dataclass
class GradingResult:
    """Structured output from a single MathProofRubric grading call."""

    score: int
    feedback: str
    reward: float
    metrics: dict[str, float | int] = field(default_factory=dict)


def apply_score_threshold(score: float) -> float:
    """Apply reward thresholding based on verification score.

    Proofs with partial credit (score 1-5) are collapsed to 1.
    """
    if score < 1.0:
        return score
    if score < 6.0:
        return 1.0
    return score


def length_penalty(max_length: int, sequence_length: int, buffer_tokens: int) -> float:
    """Compute an overlong penalty for sequences approaching *max_length*."""
    if buffer_tokens <= 0:
        return 0.0
    if sequence_length > (max_length - buffer_tokens) and sequence_length <= max_length:
        return ((max_length - buffer_tokens) - sequence_length) / buffer_tokens
    return 0.0


class MathProofRubric(Rubric):
    """LLM-based rubric for grading mathematical proofs on a 0-7 scale."""

    def __init__(
        self,
        grader_model: str = "gemini-2.0-flash",
        prompt_template: str = "",
        custom_threshold: bool = False,
        api_base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff: list[int] | None = None,
        timeout_seconds: int = 900,
    ):
        super().__init__()
        self.grader_model = grader_model
        self.prompt_template = prompt_template
        self.custom_threshold = custom_threshold
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff or list(_DEFAULT_RETRY_BACKOFF)
        self.timeout_seconds = timeout_seconds

        client_kwargs: dict[str, Any] = {}
        if api_base_url is not None:
            client_kwargs["base_url"] = api_base_url
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        self._client = openai.AsyncOpenAI(**client_kwargs)

    async def forward(self, action: Any, observation: Any) -> float: # type: ignore
        """Evaluate a proof submission and return the normalized reward."""
        proof = getattr(action, "proof", str(action))
        problem = getattr(observation, "problem", "")
        reference_solution = getattr(observation, "reference_solution", "")
        grading_guidelines = getattr(observation, "grading_guidelines", "")
        result = await self.grade(
            proof, problem, reference_solution, grading_guidelines
        )
        return result.reward

    async def grade(
        self,
        proof: str,
        problem: str,
        reference_solution: str,
        grading_guidelines: Union[str, list, None] = "",
    ) -> GradingResult:
        """Grade a proof and return a :class:`GradingResult`."""
        metrics: dict[str, float | int] = {
            "verifier/rollouts/success": 0,
            "verifier/rollouts/failure": 0,
            "verifier/failures/timeout": 0,
            "verifier/failures/rate_limit": 0,
            "verifier/failures/no_input": 0,
            "verifier/failures/no_score_tag": 0,
            "verifier/failures/all_attempts_failed": 0,
            "verifier/failures/num_retries": 0,
            "verifier/runtime/latency_per_request": 0.0,
            "verifier/runtime/input_tokens": 0,
            "verifier/runtime/output_tokens": 0,
        }

        if not proof.strip():
            metrics["verifier/rollouts/failure"] = 1
            metrics["verifier/failures/no_input"] = 1
            return GradingResult(
                score=0,
                feedback="Empty proof submission.",
                reward=0.0,
                metrics=metrics,
            )

        guidelines_str: str
        if grading_guidelines is None:
            guidelines_str = ""
        elif isinstance(grading_guidelines, str):
            guidelines_str = grading_guidelines
        else:
            guidelines_str = parse_schema(grading_guidelines)

        prompt = self._build_prompt(proof, problem, reference_solution, guidelines_str)

        attempt_causes: list[str] = []
        t0 = time.perf_counter()
        for attempt in range(1, self.max_retries + 1):
            try:
                response_text = await asyncio.wait_for(
                    self._call_llm(prompt),
                    timeout=self.timeout_seconds,
                )
                score, feedback = self._parse_response(response_text)

                elapsed = time.perf_counter() - t0
                metrics["verifier/runtime/latency_per_request"] = round(elapsed, 4)
                metrics["verifier/failures/num_retries"] = attempt - 1
                metrics["verifier/runtime/input_tokens"] = max(1, len(prompt) // 4)
                metrics["verifier/runtime/output_tokens"] = max(
                    1, len(response_text) // 4
                )

                if not re.search(r"<score>\d+</score>", response_text):
                    metrics["verifier/failures/no_score_tag"] = 1
                    metrics["verifier/rollouts/failure"] = 1
                else:
                    metrics["verifier/rollouts/success"] = 1

                reward = self.normalize_reward(score)
                return GradingResult(
                    score=score,
                    feedback=feedback,
                    reward=reward,
                    metrics=metrics,
                )

            except openai.RateLimitError:
                attempt_causes.append("rate_limit")
                metrics["verifier/failures/rate_limit"] += 1
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))

            except asyncio.TimeoutError:
                attempt_causes.append("timeout")
                metrics["verifier/failures/timeout"] += 1
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))

            except Exception as exc:
                attempt_causes.append(f"error:{exc}")
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))

        elapsed = time.perf_counter() - t0
        metrics["verifier/runtime/latency_per_request"] = round(elapsed, 4)
        metrics["verifier/failures/num_retries"] = self.max_retries
        metrics["verifier/failures/all_attempts_failed"] = 1
        metrics["verifier/rollouts/failure"] = 1

        return GradingResult(
            score=0,
            feedback=(
                f"Grading failed after {self.max_retries} attempt(s): "
                + "; ".join(attempt_causes)
            ),
            reward=0.0,
            metrics=metrics,
        )

    def normalize_reward(self, score: int) -> float:
        """Normalize a 0-7 score to a [0, 1] reward."""
        effective = (
            apply_score_threshold(float(score))
            if self.custom_threshold
            else float(score)
        )
        return effective / MAX_SCORE

    def _build_prompt(
        self,
        proof: str,
        problem: str,
        reference_solution: str,
        grading_guidelines: str,
    ) -> str:
        """Format the evaluator prompt template with grading variables."""
        if self.prompt_template and any(
            v in self.prompt_template for v in _TEMPLATE_VARS
        ):
            return self.prompt_template.format(
                problem=problem,
                human_solution=reference_solution,
                marking_scheme=grading_guidelines,
                solution=proof,
            )

        parts: list[str] = [
            self.prompt_template
            or (
                "You are a strict math proof grader. "
                "Score the submission from 0 to 7 based on mathematical "
                "correctness, completeness, and logical rigor."
            )
        ]
        if problem:
            parts.append(f"\n\nProblem:\n{problem}")
        if reference_solution:
            parts.append(f"\n\nReference Solution:\n{reference_solution}")
        if grading_guidelines:
            parts.append(f"\n\nGrading Guidelines:\n{grading_guidelines}")
        parts.append(f"\n\nSubmitted Proof:\n{proof}")
        parts.append(
            "\n\nProvide your score using exactly this format: <score>N</score> "
            "where N is an integer from 0 to 7."
        )
        return "".join(parts)

    async def _call_llm(self, prompt: str) -> str:
        """Send prompt to the model and return response text."""
        try:
            response = await self._client.responses.create(
                model=self.grader_model,
                input=prompt,
            )
            text = getattr(response, "output_text", None)
            if isinstance(text, str) and text:
                return text
            return self._extract_response_text(response)
        except Exception:
            response = await self._client.chat.completions.create(
                model=self.grader_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""

    def _extract_response_text(self, response: Any) -> str:
        """Best-effort extraction of plain text from a Responses API payload."""
        output = getattr(response, "output", None)
        if not isinstance(output, list):
            return ""

        chunks: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                part_type = getattr(part, "type", None)
                if part_type == "output_text":
                    text = getattr(part, "text", None)
                    if isinstance(text, str) and text:
                        chunks.append(text)

        return "\n".join(chunks)

    def _parse_response(self, text: str) -> tuple[int, str]:
        """Parse ``<score>N</score>`` from a grader response."""
        match = re.search(r"<score>(\d+)</score>", text)
        if match:
            score = max(0, min(MAX_SCORE, int(match.group(1))))
            return score, text
        return 0, text

    def _backoff(self, attempt: int) -> int:
        """Return the sleep duration (seconds) for a given attempt number."""
        idx = min(attempt - 1, len(self.retry_backoff) - 1)
        return self.retry_backoff[idx]
