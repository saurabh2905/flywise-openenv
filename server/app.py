# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Flywise Environment.

This module creates an HTTP server that exposes the FlywiseEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    From the project root, use the **same Python as your venv** (avoid ~/.local/bin/uvicorn).

    # Recommended (after: python3 -m venv .venv && source .venv/bin/activate && pip install -e .):
    python -m uvicorn FlyWise.server.app:app --host 0.0.0.0 --port 8000

    # Or use the console script from pyproject (same venv):
    server

    # Development with auto-reload:
    python -m uvicorn FlyWise.server.app:app --reload --host 0.0.0.0 --port 8000

    # If you prefer the flat module path, stay in project root and use venv Python:
    python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import FlywiseAction, FlywiseObservation
    from .FlyWise_environment import FlywiseEnvironment
except (ImportError, ModuleNotFoundError):
    try:
        from FlyWise.models import FlywiseAction, FlywiseObservation
        from FlyWise.server.FlyWise_environment import FlywiseEnvironment
    except ImportError:
        from models import FlywiseAction, FlywiseObservation
        from server.FlyWise_environment import FlywiseEnvironment


# Create the app with web interface and README integration
app = create_app(
    FlywiseEnvironment,
    FlywiseAction,
    FlywiseObservation,
    env_name="FlyWise",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m FlyWise.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn FlyWise.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    # OpenEnv validate expects the literal substring main() in server/app.py (see openenv/cli/_validation.py).
    main(port=args.port)
