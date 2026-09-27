from __future__ import annotations

from unittest.mock import patch

from customer_service.api.run import main, selector_loop_factory


def test_local_launcher_starts_uvicorn_with_the_application() -> None:
    with patch("customer_service.api.run.uvicorn.run") as run:
        main()

    run.assert_called_once_with(
        "customer_service.api.app:app",
        host="127.0.0.1",
        port=8000,
        loop="customer_service.api.run:selector_loop_factory",
    )


def test_windows_loop_factory_creates_a_selector_loop() -> None:
    loop = selector_loop_factory()
    try:
        assert "Selector" in type(loop).__name__
    finally:
        loop.close()
