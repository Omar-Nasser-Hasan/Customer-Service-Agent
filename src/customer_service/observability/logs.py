"""Make application log lines, and their structured extras, visible in container logs."""

from __future__ import annotations

import logging

# Every attribute a bare LogRecord carries; anything else came from `extra=`.
_STANDARD_ATTRIBUTES = frozenset(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class KeyValueFormatter(logging.Formatter):
    """Append `extra=` fields as key=value so they show up in `docker compose logs`."""

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        extras = {k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRIBUTES}
        if extras:
            line += " " + " ".join(f"{key}={value}" for key, value in sorted(extras.items()))
        return line


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(KeyValueFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger = logging.getLogger("customer_service")
    logger.setLevel(logging.INFO)
    logger.handlers = [handler]