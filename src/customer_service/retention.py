"""Manual retention command; scheduling is intentionally deferred to Iteration 5."""

from __future__ import annotations

import asyncio

from customer_service.api.dependencies import ApplicationRuntime


async def _run() -> None:
    runtime = ApplicationRuntime()
    await runtime.start()
    try:
        deleted = await runtime.prune_expired_threads()
        print(f"Pruned {len(deleted)} expired thread(s): {', '.join(deleted) or 'none'}")
    finally:
        await runtime.stop()


if __name__ == "__main__":
    asyncio.run(_run())
