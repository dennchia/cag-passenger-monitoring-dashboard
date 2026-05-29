# CAG Passenger Monitoring Dashboard

Fresh Streamlit rebuild for the **Real-Time Passenger Counting and Monitoring System for CAG**.

The dashboard follows the decisions in `style.md`: a light operations board centered on fused passenger count, capacity risk, active alerts, system health, and an enhanced schematic tent floorplan.

## Run

```powershell
pip install -r requirements.txt
python -m streamlit run app.py
```

## Project Structure

```text
app.py                         thin Streamlit entry point
cag_dashboard/
  app_shell.py                 dashboard page orchestration
  config.py                    paths, scenarios, thresholds, colours
  controls.py                  top-right settings panel
  data.py                      JSON loading and defaults
  logic.py                     status, trend, age, capacity, zone calculations
  styles.py                    CSS injection and theme-aware styling
  theme.py                     light/dark theme state
  components.py                shared UI helpers such as badges and tables
  views/
    header.py                  title and status row
    overview.py                count, capacity, alert strip
    floorplan.py               enhanced tent schematic
    cameras.py                 stitched feed and camera tiles
    health.py                  chart, health cards, latest events
    people.py                  registered-persons tab
    technical.py               debug/raw JSON view
tools/
  hikvision_yolo_stream.py     RTSP + YOLO stream helper
  mock_stitched_feed.py        local mock feed for testing
```

## Data

The dashboard reads `data/sample_fusion_output.json` by default and includes demo scenarios in `data/scenarios/`.

The expected input shape is documented in `plan.md`. The UI should keep working when optional fields are missing.

For live camera stitching, run the stitching server separately and paste its HTTP stream URL into **Settings > Stitched feed URL**. The recommended first prototype URL is `http://localhost:8080/stitched_feed`.

## Test A Real Hikvision Feed

Install the optional camera dependencies:

```powershell
pip install -r requirements-camera.txt
```

Set the RTSP URL in PowerShell. Keep the real password out of committed files:

```powershell
$env:CAG_CAMERA_IP = "172.20.10.5"
$env:CAG_CAMERA_USERNAME = "admin"
$env:CAG_CAMERA_PASSWORD = "your-password"
python tools/hikvision_yolo_stream.py --channel 101
```

If channel `101` times out, try `--channel 102` for the lower-quality stream.

Open `http://localhost:8080/stitched_feed` in Chrome first. If the feed works there, paste the same URL into **Dashboard > Settings > Stitched feed URL**.
