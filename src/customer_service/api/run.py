"""Platform-safe local development launcher for the FastAPI application."""

from __future__ import annotations

import asyncio
import sys

import uvicorn


def selector_loop_factory() -> asyncio.AbstractEventLoop:
    """Supply Uvicorn a Psycopg-compatible loop on Windows.

    Current Uvicorn versions explicitly select ProactorEventLoop on Windows,
    so setting the global policy alone is not enough for Psycopg's async
    connections. Uvicorn imports custom loop values as a direct zero-argument
    loop factory.
    """

    return asyncio.SelectorEventLoop()


def main() -> None:
    """Set Psycopg's required Windows loop policy before Uvicorn creates a loop."""

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run(
        "customer_service.api.app:app",
        host="127.0.0.1",
        port=8000,
        loop="customer_service.api.run:selector_loop_factory",
    )


if __name__ == "__main__":
    main()
