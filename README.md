# CAG Passenger Monitoring Dashboard V1

FastAPI + React migration for the Passenger Monitoring Dashboard.

V1 focuses on:

- stable multi-camera MJPEG camera streaming
- SQLite-backed metrics and alerts
- saved passenger assistance observations from an external age/gender pipeline
- a compact dark React dashboard shell

Streamlit-era code is archived in `archive_v0/`.

## Requirements

- Python 3.11+ recommended
- Node.js LTS v20+
- A real Hikvision stream URL, usually RTSP:

```text
rtsp://username:password@192.168.50.192:554/Streaming/Channels/101
```

The Hikvision `http://camera-ip/` page is usually only the web management portal, not a video stream.
For multiple cameras, set `CAMERA_URLS` in `backend/.env`:

```text
CAMERA_URLS=cam_1=rtsp://username:password@192.168.50.192:554/Streaming/Channels/101,cam_2=rtsp://username:password@192.168.50.76:554/Streaming/Channels/101
PRIMARY_CAMERA_ID=cam_1
```

## Backend

```powershell
cd "C:\Users\aveng\Documents\Codex\CAG (MP)\backend"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Useful checks:

```text
http://localhost:8000/health
http://localhost:8000/api/status
http://localhost:8000/api/stream
http://localhost:8000/api/cameras
http://localhost:8000/api/cameras/cam_1/stream
```

## Frontend

```powershell
cd "C:\Users\aveng\Documents\Codex\CAG (MP)\frontend"
& "C:\Program Files\nodejs\npm.cmd" install
Copy-Item .env.example .env.local
& "C:\Program Files\nodejs\npm.cmd" run dev
```

Open:

```text
http://localhost:5173
```

## One-Command Demo Start

After installing backend and frontend dependencies once:

```powershell
.\start.ps1
```

## API

```text
GET  /health
GET  /api/status
GET  /api/stream
GET  /api/cameras
GET  /api/cameras/{camera_id}/status
GET  /api/cameras/{camera_id}/stream
GET  /api/metrics?run_id=
POST /api/metrics
GET  /api/alerts?run_id=
POST /api/alerts
GET  /api/observations?gender=&min_age=&max_age=&camera_id=&run_id=
POST /api/observations
DELETE /api/observations
```

Metrics and alerts return latest global entries when `run_id` is omitted.

Passenger observations are an assistance filter only. The dashboard stores model-provided age/gender estimates and person crop images from an external pipeline; it does not run MiVOLO, identify people, or perform face recognition.

Example observation upload:

```powershell
curl.exe -X POST "http://localhost:8000/api/observations" `
  -F "image=@C:\path\to\person_crop.jpg" `
  -F "age=42" `
  -F "gender=male" `
  -F "camera_id=cam_1" `
  -F "age_confidence=0.88" `
  -F "gender_confidence=0.94"
```
