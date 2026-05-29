from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import streamlit as st

from cag_dashboard import config

def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if len(text) == 5 and text[2] == ":":
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def format_timestamp(value: Any) -> str:
    if not value:
        return "Unknown"
    text = str(value)
    timestamp = parse_timestamp(text)
    if timestamp is None:
        return text
    return timestamp.strftime("%H:%M:%S")


def seconds_since(value: Any) -> float | None:
    timestamp = parse_timestamp(value)
    if timestamp is None:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return (datetime.now(timestamp.tzinfo) - timestamp).total_seconds()


def data_age_label(age: float | None) -> str:
    if age is None:
        return "Unknown"
    if age < -2:
        return "Clock mismatch"
    if age <= config.STALE_AFTER_SECONDS:
        return "Live"
    return f"Stale: {age:.1f}s"


def normalise_status(value: Any) -> str:
    status = str(value or "normal").lower().replace("_", " ").strip()
    if status in {"alert", "error"}:
        return "critical"
    if status not in config.SEVERITY:
        return "normal"
    return status


def capacity_state(total_count: int, capacity_limit: int, warning: float, critical: float) -> tuple[float, str]:
    ratio = total_count / capacity_limit if capacity_limit else 0.0
    if ratio >= critical:
        return ratio, "critical"
    if ratio >= warning:
        return ratio, "warning"
    return ratio, "normal"


def display_state(data: dict[str, Any], demo_mode: bool, warning: float, critical: float) -> tuple[str, float | None]:
    age = 0.0 if demo_mode else seconds_since(data.get("timestamp"))
    stale = False if demo_mode else (age is None or age > config.STALE_AFTER_SECONDS)
    health = data.get("system_health", {})
    fusion_status = str(health.get("fusion_engine", "")).lower()
    source_status = normalise_status(data.get("status"))

    if stale or fusion_status == "offline" or source_status == "offline":
        return "offline", age

    _, derived_state = capacity_state(
        safe_int(data.get("total_count")),
        safe_int(data.get("capacity_limit")),
        warning,
        critical,
    )
    return max([source_status, derived_state], key=lambda item: config.SEVERITY[item]), age


def history_points(history: list[dict[str, Any]]) -> list[tuple[float, int, str]]:
    parsed: list[tuple[float, int, str]] = []
    base_minute: float | None = None
    for item in history:
        raw_ts = item.get("timestamp")
        label = str(raw_ts or "")
        total = safe_int(item.get("total_count"))
        minute: float | None = None
        timestamp = parse_timestamp(raw_ts)
        if timestamp is not None:
            minute = timestamp.timestamp() / 60
        elif len(label) == 5 and label[2] == ":":
            try:
                hour, minute_text = label.split(":")
                minute = int(hour) * 60 + int(minute_text)
            except ValueError:
                minute = None
        if minute is None:
            continue
        if base_minute is None:
            base_minute = minute
        parsed.append((minute - base_minute, total, label))
    return parsed


def trend_summary(history: list[dict[str, Any]], capacity_limit: int) -> tuple[str, str, str]:
    points = history_points(history)
    if len(points) < 2:
        return "Stable", "No trend", "Not enough history"

    previous = points[-2]
    current = points[-1]
    delta = current[1] - previous[1]
    elapsed = max(current[0] - previous[0], 1)
    rate = delta / elapsed

    if delta > 2:
        direction = "Rising"
        status = f"+{delta} since last point"
    elif delta < -2:
        direction = "Falling"
        status = f"{delta} since last point"
    else:
        direction = "Stable"
        status = "Minimal change"

    if rate > 0 and current[1] < capacity_limit:
        eta_minutes = (capacity_limit - current[1]) / rate
        eta = f"{eta_minutes:.0f} min to capacity"
    elif current[1] >= capacity_limit:
        eta = "At or above capacity"
    else:
        eta = "No ETA while stable/falling"
    return direction, status, eta


def reliability_note(data: dict[str, Any], state: str, age: float | None, demo_mode: bool) -> str:
    cameras = data.get("cameras", [])
    offline = [cam.get("camera_id", "camera") for cam in cameras if normalise_status(cam.get("status")) != "normal" and str(cam.get("status", "")).lower() != "online"]
    low_fps = [cam.get("camera_id", "camera") for cam in cameras if safe_float(cam.get("fps")) and safe_float(cam.get("fps")) < 8]
    visibility = [
        cam.get("camera_id", "camera")
        for cam in cameras
        if str(cam.get("visibility_status", "normal")).lower() not in {"normal", "unknown", ""}
    ]
    if state == "offline":
        return "Last known output. Confirm fusion engine, network, and timestamp before using the count."
    if not demo_mode and (age is None or age > config.STALE_AFTER_SECONDS):
        return "Data freshness is uncertain. Check the live JSON writer."
    if offline:
        return f"Lower confidence: {', '.join(map(str, offline))} offline."
    if low_fps:
        return f"Lower confidence: low FPS from {', '.join(map(str, low_fps))}."
    if visibility:
        return f"Visibility warning from {', '.join(map(str, visibility))}."
    return "Count reliability normal based on current camera health."


def update_critical_timer(state: str) -> str:
    now = datetime.now()
    if state == "critical":
        if "critical_started_at" not in st.session_state:
            st.session_state.critical_started_at = now
        elapsed = int((now - st.session_state.critical_started_at).total_seconds())
        if elapsed < 60:
            return f"{elapsed}s in critical"
        return f"{elapsed // 60}m {elapsed % 60}s in critical"
    st.session_state.pop("critical_started_at", None)
    return "Not critical"


def zone_deltas(zones: list[dict[str, Any]]) -> dict[str, int]:
    current = {str(zone.get("zone_id", index + 1)): safe_int(zone.get("count")) for index, zone in enumerate(zones)}
    previous = st.session_state.get("previous_zone_counts", {})
    deltas = {zone_id: count - safe_int(previous.get(zone_id)) for zone_id, count in current.items() if zone_id in previous}
    st.session_state.previous_zone_counts = current
    return deltas


def active_alert(data: dict[str, Any], state: str, total: int, capacity: int) -> tuple[str, str]:
    if state == "critical":
        return "critical", f"Critical occupancy: {total}/{capacity} capacity used. Immediate attention required."
    if state == "warning":
        return "warning", f"Warning occupancy: {total}/{capacity} capacity used."
    if state == "offline":
        return "offline", "System offline or stale. Showing last known output."

    alerts = data.get("alerts", [])
    if not alerts:
        return "normal", "No active alerts."
    ranked = {"critical": 3, "alert": 3, "warning": 2, "info": 1}
    alert = max(alerts, key=lambda item: ranked.get(str(item.get("level", "info")).lower(), 0))
    level = normalise_status(alert.get("level"))
    return level, str(alert.get("message", "Active alert"))


def zone_capacity(zone: dict[str, Any], fallback: float) -> float:
    value = zone.get("zone_capacity", fallback)
    return safe_float(value, fallback)


def zone_state(percent: float) -> str:
    if percent >= 85:
        return "critical"
    if percent >= 60:
        return "warning"
    return "normal"


def camera_colour(camera: dict[str, Any]) -> str:
    status = str(camera.get("status", "")).lower()
    return config.COLOURS["normal"] if status == "online" else config.COLOURS["critical"]


def camera_positions(total: int, tent_x: float, tent_y: float, tent_w: float, tent_h: float) -> list[tuple[float, float]]:
    positions = [
        (tent_x + 118, tent_y - 28),
        (tent_x + tent_w - 118, tent_y - 28),
        (tent_x + tent_w + 30, tent_y + tent_h / 2),
        (tent_x + tent_w - 118, tent_y + tent_h + 28),
        (tent_x + 118, tent_y + tent_h + 28),
        (tent_x - 30, tent_y + tent_h / 2),
    ]
    return positions[:total]
