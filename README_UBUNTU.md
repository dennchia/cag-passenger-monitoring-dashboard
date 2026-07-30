# Ubuntu operation guide

This repository uses two separate Ubuntu Python environments:

- `.venv-cv-linux`: Torch, CUDA, YOLO, MediaPipe, TransReID, role classification, and MiVOLO.
- `backend/.venv-linux`: FastAPI, SQLite, camera previews, and MQTT ingestion.

Copied Windows virtual environments cannot run on Ubuntu.

## Initial setup

Install the NVIDIA driver, reboot, and verify both cards with `nvidia-smi`. Then run:

```bash
bash setup_ubuntu.sh
source .venv-cv-linux/bin/activate
# Install the CUDA-enabled PyTorch build selected at https://pytorch.org/get-started/locally/
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
deactivate
```

Copy `backend/.env.example` to `backend/.env` and configure `CAMERA_URLS`. Keep real RTSP credentials only in that private `.env` file.

## Normal operator startup

Run one command from the repository root:

```bash
bash start_ubuntu.sh
```

The script validates the environments and configuration, starts Mosquitto only if port 1883 is unused, builds React only when its production output is missing or stale, starts FastAPI, waits for `/health`, and opens:

```text
http://localhost:8000
```

The dashboard initially shows **Preparing computer vision**. It enables **Start Session** only after all configured models are loaded. **Stop Session** closes camera processing safely but keeps models loaded, so the next session starts faster.

Press `Ctrl+C` in the startup terminal to stop the backend, worker, and any Mosquitto process that this command created.

## Technical tester launcher

Testers can instead open the detailed Tkinter launcher:

```bash
bash launch_tracker_ubuntu.sh
```

It retains controls for camera selection, calibration, devices, model settings, thresholds, fusion, ReID, MQTT, map dimensions, grid dimensions, logging, and recording. Its camera defaults are read from `backend/.env`; credentials are redacted from the command preview.

The dashboard worker and technical launcher are independent entry points, but they intentionally cannot own the CV runtime simultaneously. Stop `start_ubuntu.sh` before using the technical launcher, or stop the technical tracker before starting the operator server.

## Frontend development

Use the separate development command when changing React code:

```bash
bash start_dev_ubuntu.sh
```

This serves Vite on `http://localhost:5173` and FastAPI on `http://localhost:8000`. Normal operators should use `start_ubuntu.sh`.

## LAN viewing and control

Read-only dashboard data is available at the LAN URL printed by the startup script. Start/Stop control is localhost-only by default.

To permit a remote operator, set both values in `backend/.env` and restart:

```text
CV_CONTROL_ALLOW_LAN=true
CV_CONTROL_TOKEN=replace-with-a-long-random-secret
```

The remote dashboard asks for this as an **Operator access code**. It is held only in browser memory and is never returned by the backend.

## Recovery

- **Computer vision unavailable:** inspect `LogEvidance/cv_service.jsonl`, verify CUDA with the command above, and check that the configured model files exist. Restart `start_ubuntu.sh` after correcting the problem.
- **MQTT unavailable:** check `ss -ltn '( sport = :1883 )'` and `LogEvidance/mosquitto-server.log`.
- **Camera cannot open:** verify the camera IP, switch port, Ethernet route, and `CAMERA_URLS`. The worker deliberately does not log RTSP credentials.
- **Port 8000 already in use:** stop the earlier backend. The startup script refuses to replace an unknown process.
- **Technical launcher reports CV already owned:** stop the operator server/worker before starting the tester tracker.

Homography files remain valid only while camera placement, resolution, and crop remain unchanged. Desktop recording may require an Xorg session rather than Wayland.
