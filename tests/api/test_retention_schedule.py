from __future__ import annotations

import asyncio

import pytest

from customer_service.api.dependencies import ApplicationRuntime
from customer_service.config.settings import Settings


@pytest.mark.asyncio
async def test_retention_loop_runs_immediately_and_cancels_cleanly() -> None:
    runtime = ApplicationRuntime(Settings(postgres_dsn="postgresql://unused"))
    invoked = asyncio.Event()

    async def prune() -> list[str]:
        invoked.set()
        return []

    runtime.prune_expired_threads = prune  # type: ignore[method-assign]
    task = asyncio.create_task(runtime._retention_loop())
    await asyncio.wait_for(invoked.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
