"""User-facing CV session API with local-by-default control protection."""

from __future__ import annotations

import hmac
import ipaddress

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from config import settings
from cv_manager import CvTransitionError, cv_manager


router = APIRouter(prefix="/api/cv", tags=["computer-vision"])


class CvSessionStart(BaseModel):
    run_id: str | None = Field(default=None, min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")


class CvStatus(BaseModel):
    state: str
    ready: bool
    running: bool
    run_id: str | None
    started_at: str | None
    stopped_at: str | None
    pid: int | None
    loading_stage: str | None
    error: str | None
    mqtt_broker_reachable: bool
    control_allowed: bool
    control_mode: str


def _is_loopback(request: Request) -> bool:
    host = request.client.host if request.client is not None else ""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _control_allowed(request: Request, supplied_token: str | None) -> bool:
    if _is_loopback(request):
        return True
    if not settings.cv_control_allow_lan:
        return False
    expected = settings.cv_control_token.get_secret_value()
    return bool(expected and supplied_token and hmac.compare_digest(expected, supplied_token))


def _require_control(request: Request, supplied_token: str | None) -> None:
    if _control_allowed(request, supplied_token):
        return
    if not settings.cv_control_allow_lan:
        raise HTTPException(
            status_code=403,
            detail="CV control is restricted to this computer.",
        )
    if not settings.cv_control_token.get_secret_value():
        raise HTTPException(
            status_code=503,
            detail="LAN control is enabled but no operator token is configured.",
        )
    raise HTTPException(status_code=401, detail="A valid operator token is required.")


def _status_for_request(request: Request, token: str | None) -> dict:
    status = cv_manager.status()
    status["control_allowed"] = _control_allowed(request, token)
    status["control_mode"] = "token" if settings.cv_control_allow_lan else "local_only"
    return status


@router.get("/status", response_model=CvStatus)
def get_cv_status(
    request: Request,
    operator_token: str | None = Header(default=None, alias="X-Operator-Token"),
) -> dict:
    return _status_for_request(request, operator_token)


@router.post("/session/start", response_model=CvStatus)
def start_cv_session(
    payload: CvSessionStart,
    request: Request,
    operator_token: str | None = Header(default=None, alias="X-Operator-Token"),
) -> dict:
    _require_control(request, operator_token)
    try:
        cv_manager.start_session(payload.run_id)
    except CvTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _status_for_request(request, operator_token)


@router.post("/session/stop", response_model=CvStatus)
def stop_cv_session(
    request: Request,
    operator_token: str | None = Header(default=None, alias="X-Operator-Token"),
) -> dict:
    _require_control(request, operator_token)
    try:
        cv_manager.stop_session()
    except CvTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _status_for_request(request, operator_token)
