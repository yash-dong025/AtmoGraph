r"""Development entry point for the AtmoGraph NLP service.

Usage (PowerShell, from the atmograph-nlp folder):
    .\.venv\Scripts\python.exe run.py
    .\.venv\Scripts\python.exe run.py --reload
    .\.venv\Scripts\python.exe run.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse

import uvicorn

from app.config import get_settings


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the AtmoGraph NLP & News Intelligence API.")
    parser.add_argument("--host", default=settings.host, help=f"Bind host (default: {settings.host})")
    parser.add_argument("--port", type=int, default=settings.port, help=f"Bind port (default: {settings.port})")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (development)")
    parser.add_argument(
        "--log-level",
        default=settings.log_level.lower(),
        choices=["debug", "info", "warning", "error", "critical"],
        help="Uvicorn log level",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        # Our logger owns console/file formatting (see app/utils/logger.py).
        log_config=None,
    )


if __name__ == "__main__":
    main()
