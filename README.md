---
title: qed-math-openenv
emoji: "🧮"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# QED Math Environment

A mathematical proof generation and evaluation environment for OpenEnv.

## Features

- **MCP Tools**: Agent interacts via MCP (Model Context Protocol)
  - `get_problem`: Get current problem statement and metadata
  - `submit_proof`: Submit proof for LLM-judge rubric grading (0-7 scale)
  - `get_grading_guidelines`: Get grading rubric for current problem

- **LLM-Judge Rubric**: Proofs graded on 0-7 scale with normalized rewards
- **Answer-mode verification**: Uses `math_verify` for fast \\boxed{} checking
- **Reward shaping**: Discount factor, length penalty, optional score thresholding
- **Flexible datasets**: Local JSONL/JSON, Hugging Face Hub, or built-in bootstrap

## Quick Start

```bash
# Install
uv sync --all-extras

# Run server
uv run server

# Or via Docker
docker build -t qed-math-env:latest -f server/Dockerfile .
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY qed-math-env:latest
```

## Usage

```python
from qed_math_env import QEDMathEnv


with QEDMathEnv(base_url="http://localhost:8000") as env:
    env.reset()
    problem = env.call_tool("get_problem")
    result = env.call_tool("submit_proof", proof="Let a=2m..."
```

## Testing

```bash
PYTHONPATH=src:envs uv run pytest tests/envs/test_qed_math_environment.py -v
```
