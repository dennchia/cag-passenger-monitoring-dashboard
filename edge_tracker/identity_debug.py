"""Opt-in event logger for diagnosing temporary cross-camera ID splits.

TEMP_IDENTITY_DEBUG: This module and every call tagged with the same marker are
temporary troubleshooting code. The logger is disabled by default and can be
removed after the identity-split investigation is complete.
"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading
import time


_enabled = False
_log_path = None
_lock = threading.Lock()
_last_emitted = {}
_write_warning_emitted = False
_base_fields = {}


def _json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        try:
            return _json_safe(to_list())
        except (TypeError, ValueError):
            pass
    scalar = getattr(value, "item", None)
    if callable(scalar):
        try:
            return _json_safe(scalar())
        except (TypeError, ValueError):
            pass
    return str(value)


def configure_identity_debug(enabled=False, log_path=None, context=None):
    """Enable the temporary logger and start a fresh JSONL trace."""

    global _enabled, _log_path, _last_emitted, _write_warning_emitted, _base_fields
    with _lock:
        _enabled = bool(enabled)
        _log_path = Path(log_path) if _enabled and log_path else None
        _last_emitted = {}
        _write_warning_emitted = False
        _base_fields = {
            str(key): _json_safe(value)
            for key, value in dict(context or {}).items()
        }
        if _log_path is not None:
            try:
                _log_path.parent.mkdir(parents=True, exist_ok=True)
                _log_path.write_text("", encoding="utf-8")
            except OSError as exc:
                print(f"[IDENTITY_DEBUG] Unable to create {_log_path}: {exc}", flush=True)
                _log_path = None

    if _enabled:
        identity_event("debug_logging_started", log_path=_log_path)


def identity_event(
    event,
    *,
    throttle_key=None,
    throttle_seconds=0.0,
    console=True,
    **fields,
):
    """Print and persist one identity decision event when debugging is enabled."""

    global _write_warning_emitted
    if not _enabled:
        return

    now_monotonic = time.monotonic()
    cache_key = None if throttle_key is None else (str(event), str(throttle_key))
    with _lock:
        if cache_key is not None:
            last_emitted = _last_emitted.get(cache_key)
            if last_emitted is not None and now_monotonic - last_emitted < float(throttle_seconds):
                return
            _last_emitted[cache_key] = now_monotonic

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": str(event),
            **_base_fields,
            **{str(key): _json_safe(value) for key, value in fields.items()},
        }
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        if console:
            print(f"[IDENTITY_DEBUG] {line}", flush=True)

        if _log_path is not None:
            try:
                with _log_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError as exc:
                if not _write_warning_emitted:
                    print(f"[IDENTITY_DEBUG] Unable to write {_log_path}: {exc}", flush=True)
                    _write_warning_emitted = True
