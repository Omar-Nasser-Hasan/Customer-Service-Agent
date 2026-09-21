"""Dependency wiring for the current Iteration 1 HTTP surface."""

from __future__ import annotations

from functools import lru_cache

from fastapi import HTTPException

from customer_service.config.settings import get_settings
from customer_service.graph.build import build_graph
from customer_service.infrastructure.orders import default_order_repository


@lru_cache
def get_graph():
    try:
        return build_graph(settings=get_settings(), repository=default_order_repository())
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
