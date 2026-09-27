"""Explicit production-safe FAQ indexing and optional threshold calibration."""

from __future__ import annotations

import argparse
import asyncio
import sys

from customer_service.config.settings import get_settings
from customer_service.retrieval.calibration import calibrate
from customer_service.retrieval.corpus import load_corpus
from customer_service.retrieval.runtime import faq_repository_runtime


async def _run(should_calibrate: bool) -> None:
    async with faq_repository_runtime(get_settings()) as repository:
        result = await repository.sync(load_corpus())
        print(f"FAQ sync complete: inserted={result.inserted} updated={result.updated} unchanged={result.unchanged} deleted={result.deleted}")
        if should_calibrate:
            result = await calibrate(repository)
            print("Calibration passed. Set this explicitly in your deployment configuration:")
            print(f"FAQ_MIN_SIMILARITY={result.threshold}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize the bilingual FAQ corpus to pgvector")
    parser.add_argument("--calibrate", action="store_true", help="derive a threshold from committed fixtures")
    args = parser.parse_args()
    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            runner.run(_run(args.calibrate))
    else:
        asyncio.run(_run(args.calibrate))


if __name__ == "__main__":
    main()
