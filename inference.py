"""Inference script for QED Math with strict stdout compliance.

This script emits only [START], [STEP], and [END] lines to stdout.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Optional, cast

from openai import OpenAI

from client import QEDMathEnv


def _load_local_dotenv() -> None:
    """Load .env values if present without overriding exported env vars."""
    candidates = [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]
    seen: set[Path] = set()
    for env_path in candidates:
        resolved = env_path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)

        for raw_line in resolved.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()

            key, sep, value = line.partition("=")
            if not sep:
                continue

            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]

            if key:
                os.environ.setdefault(key, value)


_load_local_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b:novita")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

QED_MATH_URL = os.getenv("QED_MATH_URL", "http://localhost:8000")
TASK_NAME = os.getenv("TASK_NAME", "solve-qed-math")
BENCHMARK = os.getenv("BENCHMARK", "qed-math")

MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))

SYSTEM_PROMPT = (
    "You are an expert mathematician. Always call get_problem first, then reason "
    "carefully and call submit_proof with a complete solution. Use "
    "get_grading_guidelines if helpful."
)


def _single_line(value: Any) -> str:
    """Normalize text values so each log record stays on one line."""
    return re.sub(r"\s+", " ", str(value)).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(
        f"[START] task={_single_line(task)} env={_single_line(env)} model={_single_line(model)}",
        flush=True,
    )


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = str(error) if error else "null"
    print(
        f"[STEP] step={step} action={_single_line(action)} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def _tools_to_openai_format(tools: list[Any]) -> list[dict[str, Any]]:
    openai_tools: list[dict[str, Any]] = []
    for tool in tools:
        properties: dict[str, Any] = {}
        required: list[str] = []
        input_schema = (
            getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
        )
        if input_schema and "properties" in input_schema:
            for name, schema in input_schema["properties"].items():
                properties[name] = {
                    "type": schema.get("type", "string"),
                    "description": schema.get("description", ""),
                }
            required = input_schema.get("required", [])

        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )
    return openai_tools


def _extract_tool_call(response: Any) -> tuple[str, dict[str, Any], str]:
    message = response.choices[0].message
    if message.tool_calls:
        tool_call_obj = cast(Any, message.tool_calls[0])
        function_payload = getattr(tool_call_obj, "function", None)
        tool_call_id = str(getattr(tool_call_obj, "id", "fallback"))

        if function_payload is not None:
            tool_name = str(getattr(function_payload, "name", "submit_proof"))
            raw_arguments = str(getattr(function_payload, "arguments", "{}"))
            try:
                tool_args = json.loads(raw_arguments)
            except json.JSONDecodeError:
                tool_args = {"proof": raw_arguments}
        else:
            tool_name = "submit_proof"
            raw_input = getattr(tool_call_obj, "input", "")
            tool_args = {"proof": str(raw_input)}
    else:
        tool_name = "submit_proof"
        tool_args = {"proof": message.content or ""}
        tool_call_id = "fallback"

    return tool_name, tool_args, tool_call_id


def _as_mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {"result": str(value)}


async def run_episode(
    env: QEDMathEnv,
    client: OpenAI,
    tools: list[dict[str, Any]],
) -> tuple[bool, int, list[float]]:
    tool_names = {tool["function"]["name"] for tool in tools}

    await env.reset()

    chat_history: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Solve the current QED math problem."},
    ]

    rewards: list[float] = []
    steps_taken = 0
    success = False

    for step in range(1, MAX_STEPS + 1):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=cast(Any, chat_history),
            tools=cast(Any, tools),
            tool_choice="required",
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )

        tool_name, tool_args, tool_call_id = _extract_tool_call(response)

        if tool_name not in tool_names:
            tool_name = "submit_proof"
            tool_args = {"proof": tool_args.get("proof", str(tool_args))}

        call_kwargs = dict(tool_args)

        step_result = await env.call_tool(tool_name, **call_kwargs)

        result_dict = _as_mapping(step_result)

        reward = float(result_dict.get("reward") or 0.0)
        done = bool(result_dict.get("done", False))
        error_raw = result_dict.get("last_action_error")
        error = str(error_raw) if error_raw is not None else None

        action_str = json.dumps({"tool": tool_name, "args": tool_args}, ensure_ascii=True)
        log_step(step=step, action=action_str, reward=reward, done=done, error=error)

        rewards.append(reward)
        steps_taken = step

        if done:
            success = bool(result_dict.get("is_correct", reward > 0.0))
            break

        chat_history.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args),
                        },
                    }
                ],
            }
        )
        chat_history.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(result_dict),
            }
        )

    return success, steps_taken, rewards


async def async_main() -> None:
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN must be set.\nOptional fallback: API_KEY.")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    success = False
    steps_taken = 0
    rewards: list[float] = []

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    caught_error: Exception | None = None

    try:
        async with QEDMathEnv(base_url=QED_MATH_URL) as raw_env:
            env = cast(QEDMathEnv, raw_env)
            mcp_tools = await env.list_tools()
            tools = _tools_to_openai_format(mcp_tools)
            success, steps_taken, rewards = await run_episode(
                env=env,
                client=client,
                tools=tools,
            )
    except Exception as exc:
        caught_error = exc
        success = False
    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    if caught_error is not None:
        raise SystemExit(1) from caught_error


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
