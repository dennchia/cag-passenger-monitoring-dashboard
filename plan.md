# CAG Passenger Monitoring Dashboard Project Plan

## Project Summary

This project is a passenger monitoring dashboard for a CAG evacuation-like or controlled crowd-management scenario. The system is designed to help operators see camera status, live video, passenger counts, alerts, and assistance information clearly during a demo or field test.

The project has progressed through several versions:

- **V0:** Streamlit proof-of-concept dashboard using JSON demo data.
- **V1:** Active FastAPI + React dashboard with live camera streaming, SQLite persistence, and passenger assistance logs.
- **V1.5:** Planned operational visibility upgrade with zone capacity bars and historical trend sparklines.
- **V2:** Future full intelligence layer for tighter YOLO/fusion integration and faster real-time updates.

The active project now uses:

```text
backend/   FastAPI + OpenCV + SQLAlchemy 2.0 + SQLite
frontend/  React 18 + Vite + Tailwind + Lucide
archive_v0/ archived Streamlit prototype
```

## V0 - Streamlit Prototype

### Purpose

V0 was built to prove the dashboard concept before committing to a full backend/frontend architecture. It focused on the operational story: show passenger count, capacity risk, camera health, floorplan status, and demo scenarios in one dashboard.

V0 was useful because it let the team decide what information mattered most before spending time on a more complex React/FastAPI system.

### What V0 Was Supposed To Do

- Read a dashboard-ready fusion JSON file.
- Display the total fused passenger count.
- Show capacity status using normal, warning, critical, and offline states.
- Show whether the data was live or stale.
- Display zone-level occupancy inside a tent-style layout.
- Show camera status, per-camera counts, FPS, and visibility notes.
- Show system health such as fusion engine, network, and power state.
- Show alerts and event logs for demo explanation.
- Include a registered-persons tab for consented test participants only.
- Include a technical/debug view for raw JSON inspection.
- Support demo scenarios for normal, warning, critical, camera offline, registered persons, and fusion offline cases.
- Allow a stitched camera feed URL to be embedded for visual confirmation.

### What V0 Actually Built

V0 is archived in `archive_v0/`.

It included:

- `app.py` as a thin Streamlit launcher.
- `cag_dashboard/app_shell.py` for dashboard orchestration, tabs, settings, refresh loop, and view layout.
- `cag_dashboard/config.py` for paths, demo scenarios, thresholds, state colours, and theme palettes.
- `cag_dashboard/controls.py` for the top-right settings panel.
- `cag_dashboard/data.py` for safe JSON loading.
- `cag_dashboard/logic.py` for data age, capacity state, trend, ETA, alerts, camera state, and zone calculations.
- `cag_dashboard/styles.py` and `theme.py` for custom CSS, light mode, dark mode, and theme state.
- `cag_dashboard/views/overview.py` for fused count, capacity, operational brief, and alert strip.
- `cag_dashboard/views/floorplan.py` for the enhanced tent schematic.
- `cag_dashboard/views/cameras.py` for stitched feed and compact camera status tiles.
- `cag_dashboard/views/health.py` for count-over-time chart, system health, and latest events.
- `cag_dashboard/views/people.py` for registered-persons display.
- `cag_dashboard/views/technical.py` for raw JSON/debug display.
- `data/sample_fusion_output.json` as the default demo input.
- `data/scenarios/` for multiple testing states.
- `tools/mock_stitched_feed.py` for local feed testing.
- `tools/hikvision_yolo_stream.py` for RTSP + YOLO experimentation.

### V0 Data Contract

The V0 dashboard expected JSON fields such as:

- `schema_version`
- `timestamp`
- `run_id`
- `total_count`
- `capacity_limit`
- `status`
- `system_health`
- `zones`
- `cameras`
- `recognized_persons`
- `alerts`
- `event_log`
- `count_history`

This JSON contract helped shape the later database/API design in V1.

### Why V0 Was Archived

V0 was successful as a prototype, but Streamlit became limiting for the next stage:

- Layout control was difficult when the dashboard became more complex.
- Live video embedding was awkward.
- The page could feel cramped or require unwanted scrolling.
- UI polish was harder to control.
- Streamlit reruns made some interactions less smooth.
- It was not ideal for a modular frontend with future live updates.

The Streamlit version was archived rather than deleted because it still documents the original dashboard logic, style decisions, and demo scenarios.

## V1 - FastAPI + React Active Dashboard

### Purpose

V1 replaced the Streamlit dashboard with a decoupled system:

- FastAPI handles camera streaming, API routes, persistence, uploads, and backend state.
- React handles the dashboard UI, live video display, filtering, and operator interaction.

The main goal of V1 is to create a proper system foundation that is easier to extend than Streamlit.

### What V1 Is Supposed To Do

- Stream live Hikvision camera feeds through FastAPI.
- Display live MJPEG camera streams smoothly in React.
- Support multiple independent camera streams.
- Show camera connection health and resolution.
- Store metric rows in SQLite.
- Store alert rows in SQLite.
- Receive passenger assistance observations from an external age/gender pipeline.
- Store uploaded person crop images in an ignored backend upload folder.
- Let staff filter observations by age, gender, camera, and run ID.
- Show demographic summary totals for passenger observations.
- Clearly state that the assistance feature is not face recognition or identity matching.
- Keep the dashboard modular so teammate pipelines can post data later.

### Current V1 Architecture

- `backend/config.py` centralizes environment loading with `pydantic-settings`.
- `backend/camera.py` runs one OpenCV background thread per configured camera and serves latest JPEG bytes.
- `backend/database.py` configures SQLite with WAL mode.
- `backend/models.py` defines SQLAlchemy models and Pydantic v2 schemas.
- `backend/crud.py` contains database operations.
- `backend/main.py` exposes health, camera, stream, metrics, alerts, and observation endpoints.
- `frontend/src/App.jsx` polls camera status, metrics, and alerts every 3 seconds.
- `frontend/src/components/VideoPlayer.jsx` mounts selected camera streams as native MJPEG `<img>` elements.
- `frontend/src/components/OperationsStatusPills.jsx` shows compact operational health pills.
- `frontend/src/components/OperationsSidebarTabs.jsx` toggles between metrics and alerts.
- `frontend/src/components/AssistanceView.jsx` filters uploaded age/gender observations and person crops.
- `frontend/src/components/ThemeDropdown.jsx` supports light/dark theme selection.

### What V1 Actually Built

V1 currently includes:

- FastAPI backend server.
- React + Vite frontend.
- Tailwind-based dashboard layout.
- Live MJPEG stream endpoint for the primary camera.
- Per-camera MJPEG stream endpoints.
- Multi-camera status endpoint.
- Camera selector in the live video panel.
- Placeholder stream frames when a camera is offline or reconnecting.
- Automatic camera reconnect attempts.
- SQLite database with WAL mode.
- `MetricLog` model for passenger metrics.
- `SystemAlert` model for alert records.
- `PassengerObservation` model for passenger assistance logs.
- `POST /api/metrics` for future fusion pipeline metric writes.
- `POST /api/alerts` for future alert writes.
- `POST /api/observations` for external age/gender/person-crop uploads.
- `GET /api/observations` with filters for gender, min age, max age, camera ID, run ID, and limit.
- `GET /api/observations/summary` for run-level demographic summary.
- `DELETE /api/observations` to clear saved demo observations and uploaded crop files.
- Seed demo data script for testing without the external MiVOLO pipeline.
- Passenger Assistance tab with filter controls and compact observation cards.
- Age input validation for min/max age filters.
- Global summary / local filter behavior: top demographic summary follows run ID only, while bottom observation cards follow all filters.
- Light/dark dashboard theme support.

### V1 Operations Tab

The Operations tab is focused on live system monitoring.

It currently shows:

- compact backend/camera/resolution status pills
- live camera stream
- camera selector when multiple cameras are configured
- selected camera offline overlay
- metrics and alerts in sidebar mini-tabs

The Operations tab is intentionally simpler than the old V0 view because V1 first needed stable camera streaming, API separation, and persistence.

### V1 Passenger Assistance Tab

The Passenger Assistance tab supports the evacuation story where a passenger may ask staff for help finding a family member.

The external pipeline is expected to send:

- person crop image
- estimated age
- estimated gender
- camera ID
- optional run ID
- optional track ID
- optional age/gender confidence
- optional timestamp

The dashboard displays these as assistance logs. Staff can filter observations to narrow the manual search.

Important privacy boundary:

- The dashboard does not identify people.
- The dashboard does not perform face recognition.
- The dashboard does not claim that a person is a specific identity.
- A human helper must manually verify crop images.

### V1 API Surface

Current main API routes:

```text
GET    /health
GET    /api/status
GET    /api/stream
GET    /api/cameras
GET    /api/cameras/{camera_id}/status
GET    /api/cameras/{camera_id}/stream
GET    /api/metrics?run_id=
POST   /api/metrics
GET    /api/alerts?run_id=
POST   /api/alerts
GET    /api/observations?gender=&min_age=&max_age=&camera_id=&run_id=&limit=
GET    /api/observations/summary?run_id=
POST   /api/observations
DELETE /api/observations
```

### V1 Boundaries

V1 intentionally does not include:

- active Streamlit dashboard
- YOLO inference inside this repo
- MiVOLO inference inside this repo
- person tracking logic
- multi-camera fusion logic
- React floorplan visualization
- identity matching
- face recognition

Those responsibilities either belong to teammate pipelines or future versions.

## V1.5 - Operational Enhancements Roadmap

### Purpose

V1.5 upgrades the dashboard from a passive monitoring viewer into a more proactive operations tool. V1 shows live camera streams, metrics, alerts, and assistance logs; V1.5 should help operators understand crowd pressure, predict surges, document shifts, and turn passenger assistance matches into operational alerts.

### What V1.5 Should Do

V1.5 should add four operational features:

1. Zone Capacity Status Bars.
2. Historical Trend Sparklines.
3. One-Click Shift Reports.
4. Flag for Assistance Action Cards.

V1.5 also includes a practical CV integration feature:

5. Tactical Floor Map with Live Dots.

These features must not break existing V1 behavior:

- live MJPEG camera streams
- camera selector
- metrics and alerts polling
- Passenger Assistance filters
- Global Top / Local Bottom demographic summary behavior
- light/dark theme readability

### Feature 1 - Zone Capacity Status Bars

Purpose:

Convert raw passenger counts into visual capacity pressure indicators so operators can quickly see whether a camera zone is safe, nearing capacity, or critical.

Current V1.5 implementation direction:

- Treat camera IDs as zone keys for the first implementation.
- Use backend-owned capacity configuration through `.env`.
- Use the latest `MetricLog.zone_counts` row as the current occupancy source.
- Keep this feature independent of the live camera stream so camera video continues working even when metric rows are missing.
- Use status thresholds:
  - Safe: below 60%
  - Warning: 60% to below 85%
  - Critical: 85% and above
  - Unknown: capacity missing or invalid

Backend requirements:

- Add `ZONE_CAPACITIES_JSON` to backend configuration.
- Add `GET /api/zones/status?run_id=`.
- Return latest global zone status when `run_id` is omitted.
- Return run-specific zone status when `run_id` is provided.
- Return an empty list when no metric rows exist.
- Return unknown status when a zone count exists but no capacity is configured.

Expected response:

```json
[
  {
    "zone_id": "cam_1",
    "count": 82,
    "capacity": 150,
    "percent_used": 54.7,
    "status": "safe"
  }
]
```

Frontend requirements:

- Add a compact Zone Capacity panel in the Operations tab.
- Place the panel below system status pills and above the live video.
- Show zone/camera label, count, capacity, percent used, status label, and progress bar.
- Use green, amber, red, and grey status colors.
- Show a clear empty state when no zone data exists.

### Feature 2 - Historical Trend Sparklines

Purpose:

Show whether crowd size is rising, stable, or clearing over the trailing 60 minutes.

Current V1.5 implementation direction:

- `GET /api/metrics/trends?run_id=&minutes=60` queries historical `MetricLog` rows from SQLite.
- The endpoint returns timestamp-ordered passenger count points.
- If `run_id` is omitted, the backend defaults to the latest active run ID from `MetricLog`.
- The endpoint is read-only and lightweight.
- The default trend window is the trailing 60 minutes.

Frontend behavior:

- `MetricTrendSparkline` renders a minimalist axis-free SVG chart.
- The sparkline appears beside Zone Capacity Status Bars in the Operations view.
- It uses the same polling rhythm as existing Operations data.
- It shows an empty state if fewer than two trend points are available.

### Integration Feature - Tactical Floor Map With Live Dots

Purpose:

Show the latest X/Y person positions from the teammate CV homography pipeline as dots on a schematic floor map, so operators can see both the count and approximate spatial position.

Current V1.5 implementation direction:

- Keep tactical X/Y coordinates separate from `MetricLog.zone_counts`.
- Store only the latest tactical state in backend memory because it updates frequently.
- Treat the tactical map as a global fused floor map; camera selection changes video only.
- Use `POST /api/tactical` for the CV pipeline to send `camera_id: "fused"`, run ID, inside count, outside visible count, map size, outside context size, and `positions_cm`.
- Classify positions as `inside` when they are within the calibrated `map_size_cm` tent, and `outside_visible` when they are outside the tent but still within the configured outside context range.
- Keep per-camera inside counts inside `zone_counts`; capacity bars and metric totals must use inside occupancy only.
- Do not claim incoming/outgoing direction yet; outside points are awareness context only.
- Use `GET /api/tactical/latest?run_id=` for the React dashboard to read the latest global map state.
- Treat tactical data as stale when no update has arrived for 5 seconds.

Expected tactical payload:

```json
{
  "timestamp": 1700000000,
  "camera_id": "fused",
  "run_id": "field_test_001",
  "people_count": 1,
  "inside_count": 1,
  "outside_visible_count": 1,
  "positions_cm": [
    { "x": 120.5, "y": 80.2, "area": "inside" },
    { "x": -350.0, "y": 140.0, "area": "outside_visible" }
  ],
  "map_size_cm": 300,
  "outside_context_cm": 700,
  "zone_counts": { "cam_1": 1, "cam_2": 0 }
}
```

Frontend behavior:

- `TacticalMap` renders the 3m x 3m tent as the dominant central map and compresses outside visible context into a thin surrounding border.
- Inside occupancy uses red dots; outside visible context uses cyan dots with a small legend.
- The map appears in the Operations tab below status pills and above the live video.
- It polls every 1 second while Operations is active.
- The map shows waiting/stale status when data is missing or older than 5 seconds.

### Feature 3 - One-Click Shift Reports

Purpose:

Create a simple audit/reporting workflow for end-of-shift or post-demo review.

Planned backend requirements:

- Add an export endpoint for the active run.
- Active run defaults to the latest `MetricLog.run_id` unless a run ID is supplied.
- Generate CSV containing:
  - run ID
  - report generation time
  - peak crowd count
  - peak crowd timestamp
  - total alert count
  - alert count by severity
  - global demographic summary
- Keep uploaded person crop images out of the CSV.

Planned frontend requirements:

- Add an `Export Shift Summary` button in the Operations tab.
- Download the generated CSV in the browser.
- Keep the button visually secondary to live monitoring controls.
- Do not block live video or polling while export is requested.

### Feature 4 - Flag for Assistance Action Cards

Purpose:

Connect Passenger Assistance search results to the Operations alert queue, so a possible manual match can be escalated for staff attention.

Planned backend requirements:

- Add an endpoint that creates a high-priority alert from a passenger observation.
- Store the generated alert in the existing `SystemAlert` table.
- Alert message should include:
  - observation ID
  - last known camera
  - last seen timestamp
  - estimated age and gender
- Keep this as an assistance workflow action, not identity matching.

Planned frontend requirements:

- Add a `Flag Match` button to Passenger Assistance person crop cards.
- On click, send the observation ID to the backend.
- Show success/error feedback on the card or Assistance view.
- Let the new alert appear in the Operations alert feed through existing polling.

### V1.5 Implementation Order

1. Zone Capacity Status Bars - implemented first.
2. Historical Trend Sparklines - implemented second.
3. Tactical Floor Map with Live Dots - implemented as the first CV spatial integration feature.
4. One-Click Shift Reports - next planned feature.
5. Flag for Assistance Action Cards - planned after reports.

This order starts with operational visibility, then trend awareness, then reporting, then workflow escalation.

Planned future endpoints:

```text
GET /api/zones/status?run_id=
GET /api/metrics/trends?run_id=&minutes=60
POST /api/tactical
GET /api/tactical/latest?camera_id=&run_id=
GET /api/reports/shift-summary?run_id=
POST /api/observations/{observation_id}/flag
```

### V1.5 Expected Data Direction

The future fusion pipeline should eventually post metrics like:

```json
{
  "run_id": "field_test_001",
  "passenger_count": 128,
  "camera_online_count": 2,
  "zone_counts": {
    "cam_1": 82,
    "cam_2": 46
  }
}
```

The dashboard should then transform those stored metric rows into operator-friendly zone bars, trend sparklines, reports, and alerts.

## CV/Fusion Pipeline Integration Roadmap

### Purpose

This section defines what is needed to integrate the dashboard with the teammate computer vision and fusion pipeline. The dashboard should not run YOLO, MiVOLO, tracking, or multi-camera fusion directly. It should receive processed outputs, validate them, store important records, and display them clearly.

### Integration Stage 1 - Agree On Data Contracts

Before connecting systems, the team must agree on exact payload formats.

Metric/count contract:

```json
{
  "run_id": "field_test_001",
  "timestamp": "2026-06-04T14:30:00+08:00",
  "passenger_count": 128,
  "camera_online_count": 2,
  "zone_counts": {
    "cam_1": 82,
    "cam_2": 46
  }
}
```

Passenger observation contract:

```text
POST /api/observations
Content-Type: multipart/form-data
```

Required fields:

```text
image
age
gender
camera_id
```

Optional fields:

```text
run_id
track_id
age_confidence
gender_confidence
timestamp
```

Alert contract:

```json
{
  "run_id": "field_test_001",
  "timestamp": "2026-06-04T14:30:00+08:00",
  "severity": "warning",
  "message": "cam_2 visibility reduced"
}
```

Tactical map contract:

```json
{
  "timestamp": 1700000000,
  "camera_id": "cam_1",
  "run_id": "field_test_001",
  "people_count": 1,
  "positions_cm": [{ "x": 120.5, "y": 80.2 }],
  "map_size_cm": 300
}
```

Success criteria:

- Everyone agrees on field names.
- Everyone agrees on camera IDs.
- Everyone agrees on `run_id` usage.
- Everyone agrees on timestamp format.
- Everyone agrees which values are optional.

### Integration Stage 2 - Build Local API Test Harness

Before connecting the real CV pipeline, create a small test sender that simulates teammate output.

The test sender should:

- send fake metric rows
- send fake alerts
- send fake passenger observations
- send one or two sample crop images
- use the same payload format as the final pipeline

Success criteria:

- Dashboard receives metric data.
- Zone bars update.
- Tactical floor map dots update.
- Metrics panel updates.
- Alerts panel updates.
- Passenger Assistance cards appear.
- Images load from backend `image_url`.

### Integration Stage 3 - Connect CV Pipeline Using MQTT For Live Telemetry

Live telemetry integration should use MQTT so the CV pipeline can publish lightweight updates reliably without coupling itself to React or browser clients.

The CV pipeline should publish:

```text
cag/metrics
cag/tactical
cag/alerts
```

Recommended flow:

```text
CV/fusion pipeline calculates processed outputs
        ->
Publishes lightweight JSON telemetry to MQTT
        ->
FastAPI MQTT bridge subscribes and updates backend state
        ->
Dashboard displays metrics, tactical dots, and alerts through existing React components
```

HTTP remains available for:

```text
POST /api/metrics
POST /api/tactical
POST /api/alerts
```

These HTTP endpoints are useful for manual tests and fallback. Person crop observations still use:

```text
POST /api/observations
```

because images should not be sent through MQTT.

Recommended MQTT topics:

```text
cag/metrics
cag/tactical
cag/alerts
```

Success criteria:

- Friend's pipeline can publish metrics to MQTT.
- FastAPI receives MQTT messages and writes throttled metric rows to SQLite.
- Tactical floor map dots update from MQTT.
- Alerts published to MQTT appear in the Operations alert feed.
- Person crop images appear in Passenger Assistance through HTTP upload.
- Dashboard still works when optional fields are missing.

### Integration Stage 4 - Add Run/Session Coordination

Once data is flowing, standardise run/session handling.

Needed decisions:

- how a `run_id` is created
- who starts a run
- whether all teammates use the same run ID
- how old runs are reviewed
- how reports select a run

Recommended default:

```text
run_id = field_test_YYYYMMDD_001
```

Success criteria:

- Metrics, alerts, and observations from the same test share the same `run_id`.
- Dashboard can filter by run ID.
- Future shift reports can export the correct session.

### Integration Stage 5 - Keep HTTP For Heavy Uploads And Fallback

HTTP remains the correct protocol for large or file-based payloads.

HTTP should carry:

```text
- person crop images
- age/gender observations
- important uploads
- manual API testing
```

Success criteria:

- Large images are not base64 encoded into MQTT messages.
- The backend keeps `/api/observations` for multipart crop uploads.
- Manual POST tests still work during debugging.

### Integration Stage 6 - Add High-Frequency Dashboard Push If Needed

If React polling becomes too slow for the visible dashboard, upgrade the backend-to-frontend delivery path.

Recommended future architecture:

```text
CV Pipeline(s)
   -> MQTT publish

MQTT Broker
   -> FastAPI subscribes

FastAPI Backend
   -> WebSocket/SSE

React Dashboard
```

Protocol responsibilities:

```text
MQTT:
- frequent people count
- zone counts
- X/Y coordinates
- camera status
- FPS
- fusion status

HTTP POST:
- person crop images
- age/gender observations
- important uploads

WebSocket/SSE:
- live dashboard updates

SQLite:
- slower audit/history logging
```

Important rule:

```text
Do not write to SQLite every 0.1 seconds.
```

Recommended rates:

```text
CV/MQTT updates: up to 10 times per second
FastAPI in-memory latest state: up to 10 times per second
React visible update: 1-2 times per second
SQLite logging: every 1-5 seconds
```

Success criteria:

- Dashboard feels live without freezing.
- Network interruptions recover.
- SQLite does not get overloaded.
- Viewers see consistent state.

### Integration Stage 7 - Add Multi-Viewer Dashboard Support

For field use, multiple laptops may need to view the dashboard.

Recommended architecture:

```text
Strong PC / Server Laptop:
- CV pipeline
- MQTT broker
- FastAPI backend
- SQLite database
- frontend server

Viewer laptops:
- browser only
```

Success criteria:

- All viewers connect to the same backend.
- All viewers see the same counts and alerts.
- No viewer laptop runs its own database.
- No viewer laptop subscribes directly to MQTT.
- One machine remains the source of truth.

### Integration Stage 8 - Final Field Test Checklist

Before the real demo or field test, verify:

- all devices are on the same network
- backend IP address is known
- MQTT broker IP address is known
- camera IDs match across all systems
- run ID is agreed
- API endpoints are reachable
- firewall allows required ports
- MQTT port 1883 is reachable
- person crop uploads work
- metrics update correctly
- tactical dots update correctly
- alerts update correctly
- dashboard can recover from temporary disconnects
- reports/export are linked to the correct run
- no real passwords or `.env` files are committed

Success criteria:

- One command or checklist can start the system.
- Teammates know which endpoint/protocol to use.
- Dashboard displays real processed outputs.
- System remains usable if one camera or pipeline component fails.


## V2 - Future Intelligence Layer

### Purpose

V2 is the future stage where the dashboard becomes more tightly connected to the real computer vision and fusion pipeline.

V1 is mostly infrastructure and display. V2 should bring in stronger real-time intelligence once the teammate pipeline is ready.

### What V2 Should Do

- Receive fused passenger counts from the final detection/fusion pipeline.
- Support faster live metric delivery through WebSocket or Server-Sent Events.
- Bring back a React version of the spatial floorplan if real zone/fusion data is available.
- Show zone-level movement or crowding changes more clearly.
- Support stronger field-test reporting and post-session analysis.
- Continue separating dashboard display from heavy model inference.

### V2 Candidate Enhancements

- WebSocket or SSE metric stream.
- React floorplan based on zone occupancy.
- More complete run/session filtering.
- Better post-run history views.
- Fusion confidence or reliability indicators.
- Zone delta indicators.
- Critical-state timer.
- Estimated time to capacity.
- Exportable summary for final demo/reporting.

## Planning Principles

- Keep heavy CV/model processing outside the dashboard repo.
- Keep the backend responsible for streaming, persistence, and API contracts.
- Keep the frontend responsible for clear operational display and interaction.
- Keep privacy wording explicit: assistance filtering is not identification.
- Update `style.md` whenever the dashboard visual hierarchy or major layout direction changes.
- Keep `README.md` focused on current setup, run commands, and active API usage.
- Use `plan.md` as the source of truth for project progression, version scope, and roadmap.

