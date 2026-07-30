# CAG Passenger Monitoring Dashboard V1.5

FastAPI + React migration for the Passenger Monitoring Dashboard.

V1.5 focuses on:

- stable multi-camera MJPEG camera streaming
- SQLite-backed metrics and alerts
- camera-keyed zone capacity status bars
- historical passenger-count trend sparklines
- tactical floor map dots from external CV telemetry
- MQTT live telemetry ingestion
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
ZONE_CAPACITIES_JSON={"cam_1":150,"cam_2":150}
```

For teammate CV telemetry over MQTT, run a broker such as Mosquitto on the machine acting as the server. In centralised laptop-server testing, this is your laptop:

```text
MQTT_ENABLED=true
MQTT_HOST=localhost
MQTT_PORT=1883
MQTT_TOPIC_METRICS=cag/metrics
MQTT_TOPIC_TACTICAL=cag/tactical
MQTT_TOPIC_ALERTS=cag/alerts
MQTT_METRIC_LOG_INTERVAL_SECONDS=1
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

This is development mode. The React app runs through Vite and calls the backend using `VITE_API_URL`.

## One-Command Demo Start

After installing backend and frontend dependencies once:

```powershell
.\start.ps1
```

This starts the backend and Vite frontend separately for development.

## Centralised Server Mode

For deployment testing, run the dashboard as a single server. Your laptop can act as the temporary server first:

```text
Your laptop:
- Mosquitto MQTT broker
- FastAPI backend
- SQLite database
- uploaded crop images
- compiled React dashboard

Friend strong PC:
- CV pipeline publishes MQTT to your laptop IP

Staff/test devices:
- browser only
```

Run:

```powershell
.\start_server.ps1
```

The script builds `frontend/dist` and starts FastAPI on:

```text
http://localhost:8000
```

Other devices on the same network should open the network URL printed by the script, for example:

```text
http://192.168.50.197:8000
```

In server mode, the frontend uses same-origin API paths so viewer devices call the same server that served the dashboard. `start_server.ps1` forces this for the production build even if `frontend/.env.local` exists for development.

If you build manually for server mode, clear `VITE_API_URL` first:

```powershell
cd "C:\Users\aveng\Documents\Codex\CAG (MP)\frontend"
$env:VITE_API_URL=" "
& "C:\Program Files\nodejs\npm.cmd" run build
```

For temporary laptop-server MQTT testing:

```text
Backend .env on your laptop:
MQTT_ENABLED=true
MQTT_HOST=localhost
MQTT_PORT=1883

Friend CV script on strong PC:
--mqtt-broker YOUR_LAPTOP_IP
--mqtt-port 1883
```

Staff devices only need access to port `8000`. The CV publisher needs access to MQTT port `1883`.

## API

```text
GET  /health
GET  /api/status
GET  /api/stream
GET  /api/cameras
GET  /api/cameras/{camera_id}/status
GET  /api/cameras/{camera_id}/stream
GET  /api/metrics?run_id=
GET  /api/metrics/trends?run_id=&minutes=60
POST /api/metrics
GET  /api/zones/status?run_id=
GET  /api/reports/shift.xlsx?run_id=
GET  /api/reports/shift.csv?run_id=
POST /api/tactical
GET  /api/tactical/latest?camera_id=&run_id=
GET  /api/alerts?run_id=
POST /api/alerts
GET  /api/observations?gender=&min_age=&max_age=&camera_id=&run_id=
GET  /api/observations/summary?run_id=
POST /api/observations
DELETE /api/observations
GET  /api/evacuees?gender=&min_age=&max_age=&camera_id=&run_id=
GET  /api/evacuees/summary?run_id=
GET  /api/evacuees/{evacuee_id}
PUT  /api/evacuees/by-master/{run_id}/{master_identity_id}
PUT  /api/evacuees/by-master/{run_id}/{master_identity_id}/views/{view_type}
GET  /api/evacuees/reid-gallery?run_id=
DELETE /api/evacuees?run_id=
```

Metrics and alerts return latest global entries when `run_id` is omitted.
Shift report exports use the latest 24 hours in Singapore time and sample the metric timeline every 5 minutes.

MQTT topics carry lightweight live telemetry:

```text
cag/metrics   -> passenger_count, zone_counts, camera_online_count
cag/tactical  -> people_count, inside_count, outside_visible_count, positions_cm, map_size_cm, outside_context_cm
cag/alerts    -> severity and message
```

The tactical map is a global fused floor map. The CV pipeline should publish `cag/tactical` with
`camera_id: "fused"` for the combined 2D plane, while per-camera counts stay inside `zone_counts`.
The dashboard camera selector changes the live video feed only; it does not change the tactical map.
The dashboard treats `people_count` as inside occupancy. Points inside the calibrated tent render as red dots,
while visible points outside the tent render as cyan context dots in a compressed outside border.

Large person crop images still use HTTP multipart uploads. MQTT is not used for gallery images or ReID embeddings.

Passenger Assistance now groups evidence by unique ReID master identity. Each evacuee can have five progressively filled gallery slots: `baseline`, `front`, `back`, `left_side`, and `right_side`. The card thumbnail prefers Front, then the sharpest side, then Baseline, then Back. Clicking it opens all five slots for manual comparison.

FastAPI owns the SQLite database and image upload storage. For deployment, start the tracker with:

```text
--reid-api-url http://127.0.0.1:8000
```

If the CV pipeline runs on another PC, replace `127.0.0.1` with the deployment server's LAN IP. The tracker then persists identity metadata and float32 gallery embeddings in SQLite through FastAPI instead of its pickle fallback. The old `--reid-db` path remains available for standalone development runs where no backend URL is configured.

Passenger evidence is an assistance filter only. The dashboard stores model-provided age/gender estimates and ReID gallery crops; it does not establish a person's real identity or perform face recognition. Staff must manually verify the images.

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

## Seed Demo Data

To test the dashboard without the external MiVOLO pipeline:

```powershell
cd "C:\Users\aveng\Documents\Codex\CAG (MP)\backend"
.\.venv\Scripts\python.exe seed_demo_data.py --reset-observations
```

This creates generated five-view identity galleries with complete and missing-angle examples, legacy person-crop placeholders, metrics, and alerts using:

```text
run_id = demo_assistance_001
```

Restart or refresh the frontend, then open the **Passenger Assistance** tab. Use the run ID filter above if you want to see only seeded records.
