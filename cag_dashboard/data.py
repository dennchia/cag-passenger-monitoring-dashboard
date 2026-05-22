from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def read_dashboard_data(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Fusion output file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Fusion output file is invalid JSON: {path}") from exc

    if not isinstance(data, dict):
        raise RuntimeError("Fusion output must be a JSON object.")

    defaults = {
        "schema_version": "1.0",
        "timestamp": None,
        "run_id": "N/A",
        "total_count": 0,
        "capacity_limit": 0,
        "status": "normal",
        "system_health": {},
        "zones": [],
        "cameras": [],
        "recognized_persons": [],
        "alerts": [],
        "event_log": [],
        "count_history": [],
    }
    return {**defaults, **data}
