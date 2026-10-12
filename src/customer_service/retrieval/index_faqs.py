"""Explicit production-safe FAQ indexing and optional threshold calibration."""

from __future__ import annotations

import argparse
import asyncio
import sys

from customer_service.config.settings import get_settings
from customer_service.retrieval.calibration import calibrate
from customer_service.retrieval.corpus import load_corpus, load_sink, load_variants
from customer_service.retrieval.runtime import faq_repository_runtime


async def _run(should_calibrate: bool) -> None:
    settings = get_settings()
    async with faq_repository_runtime(settings) as repository:
        result = await repository.sync(load_corpus(store_name=settings.store_name))
        print(f"FAQ sync complete: inserted={result.inserted} updated={result.updated} unchanged={result.unchanged} deleted={result.deleted}")
        variants = await repository.sync_variants([*load_variants(), *load_sink()])
        print(f"FAQ variant sync complete: inserted={variants.inserted} updated={variants.updated} unchanged={variants.unchanged} deleted={variants.deleted}")
        if should_calibrate:
            result = await calibrate(repository, margin=settings.faq_min_margin)
            print("Calibration passed. Set this explicitly in your deployment configuration:")
            print(f"FAQ_MIN_SIMILARITY={result.threshold}")
            if result.margin is not None:
                print(f"FAQ_MIN_MARGIN={result.margin}")


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
