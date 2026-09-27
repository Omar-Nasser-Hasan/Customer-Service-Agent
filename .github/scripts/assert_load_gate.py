"""Fail the live gate unless success is at least 99% and p95 is at most 8 seconds."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def main(path: str) -> int:
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
    aggregate = next((row for row in rows if row.get("Name") == "Aggregated"), None)
    if aggregate is None:
        raise SystemExit("Locust aggregate statistics were not produced")
    requests = int(aggregate["Request Count"])
    failures = int(aggregate["Failure Count"])
    p95_ms = float(aggregate["95%"])
    success = 0 if requests == 0 else (requests - failures) / requests
    print(f"requests={requests} success={success:.3%} p95_ms={p95_ms:.0f}")
    return 0 if success >= 0.99 and p95_ms <= 8_000 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
