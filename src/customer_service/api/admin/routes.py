from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from customer_service.api.dependencies import ApplicationRuntime
from customer_service.auth.service import AuthenticationError, CsrfError, StaffAuthService
from customer_service.operations.models import StaffIdentity
from customer_service.operations.repository import CaseConflictError, CaseNotFoundError, CasePermissionError
from customer_service.services.cases import CaseService, CustomerWindowClosedError, TemplateValidationError


class GoogleLogin(BaseModel):
    credential: str = Field(min_length=1)


class VersionRequest(BaseModel):
    version: int = Field(ge=1)


class ReplyRequest(VersionRequest):
    text: str | None = Field(default=None, min_length=1, max_length=4000)
    template_id: str | None = None
    parameters: list[str] = Field(default_factory=list)


def create_router(runtime_provider: Any) -> APIRouter:
    router = APIRouter(prefix="/admin", tags=["admin"])

    def runtime() -> ApplicationRuntime:
        value = runtime_provider()
        if value is None or value.auth is None or value.cases is None:
            raise HTTPException(503, "Admin runtime is not ready")
        return value

    async def identity(request: Request, rt: Annotated[ApplicationRuntime, Depends(runtime)]) -> StaffIdentity:
        try:
            return rt.auth.read_session(request.cookies.get(rt.settings.staff_session_cookie_name))
        except AuthenticationError as error:
            raise HTTPException(401, str(error)) from error

    async def mutation_guard(request: Request, rt: Annotated[ApplicationRuntime, Depends(runtime)]) -> None:
        try:
            rt.auth.validate_origin(request.headers.get("origin"))
            rt.auth.validate_csrf(request.cookies.get(rt.settings.csrf_cookie_name), request.headers.get("x-csrf-token"))
        except CsrfError as error:
            raise HTTPException(403, str(error)) from error

    @router.post("/auth/google")
    async def google_login(payload: GoogleLogin, request: Request, response: Response, rt: Annotated[ApplicationRuntime, Depends(runtime)]) -> dict[str, str]:
        # Login is also protected from cross-site credential injection.
        try:
            rt.auth.validate_origin(request.headers.get("origin"))
            identity_value = rt.auth.verify_google_credential(payload.credential)
        except (AuthenticationError, CsrfError) as error:
            raise HTTPException(401, str(error)) from error
        csrf = rt.auth.mint_csrf()
        cookie = rt.auth.mint_session(identity_value)
        common = {"domain": rt.settings.admin_cookie_domain, "secure": rt.settings.admin_cookie_secure, "samesite": "lax", "path": "/"}
        response.set_cookie(rt.settings.staff_session_cookie_name, cookie, httponly=True, max_age=rt.settings.staff_session_ttl_seconds, **common)
        response.set_cookie(rt.settings.csrf_cookie_name, csrf, httponly=False, max_age=rt.settings.staff_session_ttl_seconds, **common)
        return {"status": "authenticated"}

    @router.get("/auth/session")
    async def session(current: Annotated[StaffIdentity, Depends(identity)]) -> StaffIdentity:
        return current

    @router.post("/auth/logout")
    async def logout(response: Response, _: Annotated[StaffIdentity, Depends(identity)], __: Annotated[None, Depends(mutation_guard)], rt: Annotated[ApplicationRuntime, Depends(runtime)]) -> dict[str, str]:
        response.delete_cookie(rt.settings.staff_session_cookie_name, domain=rt.settings.admin_cookie_domain, path="/")
        response.delete_cookie(rt.settings.csrf_cookie_name, domain=rt.settings.admin_cookie_domain, path="/")
        return {"status": "logged_out"}

    @router.post("/auth/ws-token")
    async def websocket_token(current: Annotated[StaffIdentity, Depends(identity)], _: Annotated[None, Depends(mutation_guard)], rt: Annotated[ApplicationRuntime, Depends(runtime)]) -> dict[str, str]:
        return {"token": await rt.auth.mint_websocket_token(current)}

    @router.get("/cases")
    async def list_cases(status: str | None = None, rt: Annotated[ApplicationRuntime, Depends(runtime)] = None, _: Annotated[StaffIdentity, Depends(identity)] = None) -> list[dict[str, object]]:
        return [item.model_dump(mode="json") for item in await rt.operations.list_cases(status=status)]

    @router.get("/cases/{case_id}")
    async def case_detail(case_id: str, current: Annotated[StaffIdentity, Depends(identity)], rt: Annotated[ApplicationRuntime, Depends(runtime)]) -> dict[str, object]:
        try:
            case = await rt.operations.get_case(case_id)
        except CaseNotFoundError as error:
            raise HTTPException(404, "Case was not found") from error
        await rt.operations.audit(case_id, current, "case_viewed")
        return {"case": case.model_dump(mode="json"), "messages": [m.model_dump(mode="json") for m in await rt.operations.messages(case_id)]}

    def error(error: Exception) -> HTTPException:
        if isinstance(error, CaseNotFoundError): return HTTPException(404, "Case was not found")
        if isinstance(error, (CaseConflictError, CustomerWindowClosedError)): return HTTPException(409, str(error))
        if isinstance(error, (CasePermissionError, TemplateValidationError)): return HTTPException(403, str(error))
        return HTTPException(400, str(error))

    @router.post("/cases/{case_id}/claim")
    async def claim(case_id: str, payload: VersionRequest, current: Annotated[StaffIdentity, Depends(identity)], _: Annotated[None, Depends(mutation_guard)], rt: Annotated[ApplicationRuntime, Depends(runtime)]) -> dict[str, object]:
        try: return (await rt.cases.claim(case_id, current, payload.version)).model_dump(mode="json")
        except Exception as exc: raise error(exc) from exc

    @router.post("/cases/{case_id}/release")
    async def release(case_id: str, payload: VersionRequest, current: Annotated[StaffIdentity, Depends(identity)], _: Annotated[None, Depends(mutation_guard)], rt: Annotated[ApplicationRuntime, Depends(runtime)]) -> dict[str, object]:
        try: return (await rt.cases.release(case_id, current, payload.version)).model_dump(mode="json")
        except Exception as exc: raise error(exc) from exc

    @router.post("/cases/{case_id}/reply")
    async def reply(case_id: str, payload: ReplyRequest, current: Annotated[StaffIdentity, Depends(identity)], _: Annotated[None, Depends(mutation_guard)], rt: Annotated[ApplicationRuntime, Depends(runtime)]) -> dict[str, object]:
        if bool(payload.text) == bool(payload.template_id):
            raise HTTPException(422, "Provide exactly one of text or template_id")
        try: return (await rt.cases.queue_staff_reply(case_id, current, payload.version, text=payload.text, template_id=payload.template_id, parameters=payload.parameters)).model_dump(mode="json")
        except Exception as exc: raise error(exc) from exc

    @router.post("/cases/{case_id}/resolve")
    async def resolve(case_id: str, payload: VersionRequest, current: Annotated[StaffIdentity, Depends(identity)], _: Annotated[None, Depends(mutation_guard)], rt: Annotated[ApplicationRuntime, Depends(runtime)]) -> dict[str, object]:
        try: return (await rt.cases.resolve(case_id, current, payload.version)).model_dump(mode="json")
        except Exception as exc: raise error(exc) from exc

    @router.get("/templates")
    async def templates(_: Annotated[StaffIdentity, Depends(identity)], rt: Annotated[ApplicationRuntime, Depends(runtime)]) -> list[dict[str, object]]:
        return [template.model_dump() for template in rt.settings.whatsapp_template_catalog]

    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket, token: str | None = None) -> None:
        rt = runtime()
        try:
            await rt.auth.consume_websocket_token(token)
        except AuthenticationError:
            await websocket.close(code=1008)
            return
        await rt.realtime.connect(websocket)
        try:
            while True:
                # The browser only receives events; client messages are ignored.
                await websocket.receive_text()
        except WebSocketDisconnect:
            await rt.realtime.disconnect(websocket)

    return router
