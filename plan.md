# FastAPI + React Passenger Monitoring Dashboard V1

## Summary

The active project now uses a decoupled architecture:

```text
backend/   FastAPI + OpenCV + SQLAlchemy 2.0 + SQLite
frontend/  React 18 + Vite + Tailwind + Lucide
```

V1 focuses on stable multi-camera MJPEG streaming, camera status, metrics persistence, alert persistence, passenger assistance observations, and a compact dark dashboard shell. Streamlit-era code is archived in `archive_v0/`.

## Current Architecture

- `backend/config.py` centralizes environment loading with `pydantic-settings`.
- `backend/camera.py` runs one OpenCV background thread per configured camera and serves latest JPEG bytes.
- `backend/database.py` configures SQLite with WAL mode.
- `backend/models.py` defines SQLAlchemy models and Pydantic v2 schemas.
- `backend/crud.py` contains all metric and alert database operations.
- `backend/main.py` exposes health, camera, stream, metrics, and alerts endpoints.
- `backend/main.py` also receives external passenger observation uploads for the assistance filter.
- `frontend/src/App.jsx` polls status, metrics, and alerts every 3 seconds.
- `frontend/src/components/VideoPlayer.jsx` mounts the selected camera stream as a native MJPEG `<img>`.
- `frontend/src/components/AssistanceView.jsx` filters uploaded age/gender observations and person crops.

## V1 Boundaries

- No Streamlit active dashboard.
- No YOLO inference.
- No tracking.
- No multi-camera fusion, but V1 can view multiple independent camera streams.
- No floorplan.
- No identity matching or face recognition.

The schema already leaves room for V2 fusion output through `run_id`, `zone_counts`, and `camera_online_count`.

## V2 Runway

V1 uses HTTP polling for metrics and alerts because it is simple and reliable.

For faster YOLO/fusion updates later, move metric delivery to WebSocket or Server-Sent Events so counts stay visually synchronized with the live video.
