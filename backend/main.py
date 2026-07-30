from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

import crud
from camera import CameraStreamer, camera_manager, mjpeg_frame_generator
from config import settings
from cv_api import router as cv_router
from cv_manager import cv_manager
from database import get_db, init_db
from evacuees.router import router as evacuee_router
from evacuees.storage import (
    PUBLIC_UPLOAD_PREFIX as EVACUEE_UPLOAD_PREFIX,
    UPLOAD_DIR as EVACUEE_UPLOAD_DIR,
    ensure_upload_dir as ensure_evacuee_upload_dir,
)
from models import (
    MetricLogCreate,
    MetricLogRead,
    MetricTrendPointRead,
    PassengerObservationRead,
    PassengerObservationSummary,
    SystemAlertCreate,
    SystemAlertRead,
    TacticalStateCreate,
    TacticalStateRead,
    ZoneStatusRead,
)
from observation_storage import (
    PUBLIC_UPLOAD_PREFIX,
    UPLOAD_DIR,
    clear_observation_images,
    ensure_upload_dir,
    save_observation_image,
)
from mqtt_bridge import mqtt_bridge
from reports.shift_report import REPORT_MIME_TYPE, generate_shift_report_csv, generate_shift_report_xlsx
from tactical_state import tactical_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    ensure_upload_dir()
    ensure_evacuee_upload_dir()
    camera_manager.start_all()
    mqtt_bridge.start()
    cv_manager.start_worker()
    try:
        yield
    finally:
        cv_manager.shutdown()
        mqtt_bridge.stop()
        camera_manager.stop_all()


app = FastAPI(title="CAG Passenger Monitoring API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ensure_upload_dir()
app.mount(PUBLIC_UPLOAD_PREFIX, StaticFiles(directory=UPLOAD_DIR), name="observation-images")
ensure_evacuee_upload_dir()
app.mount(EVACUEE_UPLOAD_PREFIX, StaticFiles(directory=EVACUEE_UPLOAD_DIR), name="evacuee-images")
app.include_router(evacuee_router)
app.include_router(cv_router)

FRONTEND_DIST_PATH = settings.frontend_dist_path
FRONTEND_INDEX_PATH = FRONTEND_DIST_PATH / "index.html"
FRONTEND_ASSETS_PATH = FRONTEND_DIST_PATH / "assets"

if FRONTEND_ASSETS_PATH.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS_PATH), name="frontend-assets")


DbSession = Annotated[Session, Depends(get_db)]


def get_camera_or_404(camera_id: str) -> CameraStreamer:
    streamer = camera_manager.get(camera_id)
    if streamer is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' is not configured.")
    return streamer


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/status")
def camera_status() -> dict:
    return camera_manager.primary().status()


@app.get("/api/cameras")
def camera_statuses() -> list[dict]:
    return camera_manager.all_status()


@app.get("/api/cameras/{camera_id}/status")
def camera_status_by_id(camera_id: str) -> dict:
    return get_camera_or_404(camera_id).status()


@app.get("/api/stream")
def stream_camera() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_frame_generator(camera_manager.primary()),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/cameras/{camera_id}/stream")
def stream_camera_by_id(camera_id: str) -> StreamingResponse:
    return StreamingResponse(
        mjpeg_frame_generator(get_camera_or_404(camera_id)),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/metrics", response_model=list[MetricLogRead])
def get_metrics(db: DbSession, run_id: str | None = Query(default=None)) -> list[MetricLogRead]:
    return crud.get_latest_metrics(db, run_id=run_id, limit=10)


@app.get("/api/metrics/trends", response_model=list[MetricTrendPointRead])
def get_metric_trends(
    db: DbSession,
    run_id: str | None = Query(default=None),
    minutes: int = Query(default=60, ge=1, le=1440),
) -> list[MetricTrendPointRead]:
    return crud.get_metric_trends(db, run_id=run_id, minutes=minutes)


@app.post("/api/metrics", response_model=MetricLogRead, status_code=201)
def post_metric(payload: MetricLogCreate, db: DbSession) -> MetricLogRead:
    return crud.create_metric_log(db, payload)


@app.get("/api/zones/status", response_model=list[ZoneStatusRead])
def get_zone_status(db: DbSession, run_id: str | None = Query(default=None)) -> list[ZoneStatusRead]:
    return crud.get_zone_status(db, capacities=settings.zone_capacity_map, run_id=run_id)


@app.get("/api/reports/shift.csv")
def get_shift_report_csv(db: DbSession, run_id: str | None = Query(default=None)) -> Response:
    report = generate_shift_report_csv(db, capacities=settings.zone_capacity_map, run_id=run_id)
    return Response(
        content=report.content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{report.filename}"'},
    )


@app.get("/api/reports/shift.xlsx")
def get_shift_report_xlsx(db: DbSession, run_id: str | None = Query(default=None)) -> Response:
    report = generate_shift_report_xlsx(db, capacities=settings.zone_capacity_map, run_id=run_id)
    return Response(
        content=report.content,
        media_type=REPORT_MIME_TYPE,
        headers={"Content-Disposition": f'attachment; filename="{report.filename}"'},
    )


@app.post("/api/tactical", response_model=TacticalStateRead, status_code=201)
def post_tactical_state(payload: TacticalStateCreate) -> TacticalStateRead:
    return tactical_store.update(payload)


@app.get("/api/tactical/latest", response_model=TacticalStateRead)
def get_latest_tactical_state(
    camera_id: str | None = Query(default=None),
    run_id: str | None = Query(default=None),
) -> TacticalStateRead:
    return tactical_store.latest(camera_id=camera_id, run_id=run_id)


@app.get("/api/alerts", response_model=list[SystemAlertRead])
def get_alerts(db: DbSession, run_id: str | None = Query(default=None)) -> list[SystemAlertRead]:
    return crud.get_latest_alerts(db, run_id=run_id, limit=5)


@app.post("/api/alerts", response_model=SystemAlertRead, status_code=201)
def post_alert(payload: SystemAlertCreate, db: DbSession) -> SystemAlertRead:
    return crud.create_system_alert(db, payload)


@app.get("/api/observations/summary", response_model=PassengerObservationSummary)
def get_observation_summary(
    db: DbSession,
    run_id: str | None = Query(default=None),
) -> PassengerObservationSummary:
    return crud.get_observation_summary(db, run_id=run_id)


@app.get("/api/observations", response_model=list[PassengerObservationRead])
def get_observations(
    db: DbSession,
    gender: str | None = Query(default=None),
    min_age: float | None = Query(default=None, ge=0, le=120),
    max_age: float | None = Query(default=None, ge=0, le=120),
    camera_id: str | None = Query(default=None),
    run_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[PassengerObservationRead]:
    if min_age is not None and max_age is not None and min_age > max_age:
        raise HTTPException(status_code=400, detail="min_age cannot be greater than max_age.")

    return crud.get_latest_observations(
        db,
        gender=gender,
        min_age=min_age,
        max_age=max_age,
        camera_id=camera_id,
        run_id=run_id,
        limit=limit,
    )


@app.post("/api/observations", response_model=PassengerObservationRead, status_code=201)
async def post_observation(
    db: DbSession,
    image: UploadFile = File(...),
    age: float = Form(..., ge=0, le=120),
    gender: str = Form(..., min_length=1, max_length=32),
    camera_id: str = Form(..., min_length=1, max_length=80),
    run_id: str = Form("default", max_length=80),
    track_id: str | None = Form(default=None, max_length=120),
    age_confidence: float | None = Form(default=None, ge=0, le=1),
    gender_confidence: float | None = Form(default=None, ge=0, le=1),
    timestamp: datetime | None = Form(default=None),
) -> PassengerObservationRead:
    image_path, image_url = await save_observation_image(image)
    try:
        return crud.create_passenger_observation(
            db,
            timestamp=timestamp,
            run_id=run_id,
            camera_id=camera_id,
            track_id=track_id,
            age=age,
            gender=gender,
            age_confidence=age_confidence,
            gender_confidence=gender_confidence,
            image_path=str(image_path),
            image_url=image_url,
        )
    except Exception:
        image_path.unlink(missing_ok=True)
        raise


@app.delete("/api/observations")
def delete_observations(db: DbSession) -> dict[str, int]:
    deleted_rows = crud.clear_passenger_observations(db)
    deleted_images = clear_observation_images()
    return {"deleted_rows": deleted_rows, "deleted_images": deleted_images}


def serve_frontend_index() -> FileResponse:
    if not FRONTEND_INDEX_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend build not found. Run `npm run build` in the frontend folder.",
        )
    return FileResponse(FRONTEND_INDEX_PATH)


@app.get("/", include_in_schema=False)
def frontend_root() -> FileResponse:
    return serve_frontend_index()


@app.get("/{full_path:path}", include_in_schema=False)
def frontend_spa_fallback(full_path: str) -> FileResponse:
    reserved_paths = {"api", "assets", "health", "uploads"}
    reserved_prefixes = ("api/", "uploads/", "assets/")
    if full_path in reserved_paths or full_path.startswith(reserved_prefixes):
        raise HTTPException(status_code=404, detail="Not found")
    return serve_frontend_index()
