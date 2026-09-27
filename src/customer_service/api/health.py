"""Health endpoint independent of runtime model configuration."""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    runtime = getattr(request.app.state, "runtime", None)
    observability = getattr(getattr(runtime, "observability", None), "status", "disabled")
    return {"status": "ok", "observability": observability}
