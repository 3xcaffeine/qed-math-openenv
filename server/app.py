"""
FastAPI application for the QED Math Environment.

Exposes QEDMathEnvironment over HTTP and WebSocket endpoints,
compatible with MCPToolClient.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Or via uv:
    uv run --project . server
"""

from openenv.core.env_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from server.qed_math_environment import QEDMathEnvironment

app = create_app(
    QEDMathEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="qed_math_env",
)


@app.get("/healthz")
async def health() -> dict[str, str]:
    """Lightweight service health endpoint for basic orchestration checks."""
    return {"status": "ok"}


def main():
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
