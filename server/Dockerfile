# Multi-stage build using openenv-base
# This Dockerfile is flexible and works for both:
# - In-repo environments (with local OpenEnv sources)
# - Standalone environments (with openenv from PyPI/Git)
# The build script (openenv build) handles context detection and sets appropriate build args.
#
# Build from the env directory:
#   docker build -t qed-math-env:latest -f server/Dockerfile .
#
# Or from the repo root:
#   docker build -t qed-math-env:latest -f envs/qed_math_env/server/Dockerfile envs/qed_math_env
#
# Run:
#   docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY qed-math-env:latest

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

ARG BUILD_MODE=in-repo

COPY . /app/env

WORKDIR /app/env

RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

FROM ${BASE_IMAGE}

WORKDIR /app

COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env/.venv /app/env/.venv
COPY --from=builder /root/.local/share/uv /root/.local/share/uv

COPY --from=builder /app/env /app/env

ENV PATH="/app/.venv/bin:$PATH"

ENV PYTHONPATH="/app/env:$PYTHONPATH"

ENV PYTHONUNBUFFERED=1
ENV QED_DATASET_SPEC_JSON="{\"hub_id\":\"lm-provers/FineProofs-RL-test\",\"split\":\"test[:10]\"}"

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--ws-ping-interval", "120", "--ws-ping-timeout", "600"]
