from __future__ import annotations

from pathlib import Path

DATA_PATH = Path("data/sample_fusion_output.json")
SCENARIO_PATHS: dict[str, Path | None] = {
    "Current sample / field test": DATA_PATH,
    "Normal operation": Path("data/scenarios/01_normal_operation.json"),
    "Warning near capacity": Path("data/scenarios/02_warning_near_capacity.json"),
    "Critical over capacity": Path("data/scenarios/03_critical_over_capacity.json"),
    "Camera offline": Path("data/scenarios/04_camera_offline.json"),
    "Registered persons demo": Path("data/scenarios/05_registered_persons.json"),
    "Fusion system offline": Path("data/scenarios/06_fusion_system_offline.json"),
    "Custom JSON path": None,
}

STALE_AFTER_SECONDS = 5
CONFIDENCE_THRESHOLD = 0.80

BASE_COLOURS = {
    "normal": "#16803c",
    "warning": "#b7791f",
    "critical": "#b91c1c",
    "offline": "#64748b",
    "info": "#2563eb",
}

THEMES = {
    "light": {
        "panel": "#ffffff",
        "ink": "#0f172a",
        "muted": "#64748b",
        "line": "#dbe3ee",
        "surface": "#f8fafc",
        "app_bg": "linear-gradient(180deg, #f8fafc 0%, #eef3f8 100%)",
        "panel_soft": "#f8fafc",
        "grid": "#d5deea",
        "table_header": "#f8fafc",
        "privacy_bg": "#fffbeb",
        "privacy_text": "#713f12",
        "privacy_border": "#b7791f",
        "map_fill": "#f1f5f9",
        "theme_button_bg": "#ffffff",
        "theme_button_text": "#0f172a",
        "theme_button_border": "#cbd5e1",
        "theme_button_hover": "#f1f5f9",
    },
    "dark": {
        "panel": "#111827",
        "ink": "#f8fafc",
        "muted": "#a7b4c7",
        "line": "#334155",
        "surface": "#0f172a",
        "app_bg": "linear-gradient(180deg, #0b1120 0%, #111827 100%)",
        "panel_soft": "#172033",
        "grid": "#334155",
        "table_header": "#172033",
        "privacy_bg": "#2b2111",
        "privacy_text": "#fde68a",
        "privacy_border": "#d97706",
        "map_fill": "#1e293b",
        "theme_button_bg": "#111827",
        "theme_button_text": "#f8fafc",
        "theme_button_border": "#334155",
        "theme_button_hover": "#1f2937",
    },
}

COLOURS = {**BASE_COLOURS, **THEMES["light"]}
SEVERITY = {"normal": 0, "warning": 1, "critical": 2, "offline": 3}
