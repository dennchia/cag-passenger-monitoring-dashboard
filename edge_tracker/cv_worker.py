#!/usr/bin/env python3
"""Persistent model-owning worker controlled by FastAPI over JSON lines."""

from __future__ import annotations

import copy
import json
import os
import queue
import sys
import threading
import traceback
from datetime import datetime, timezone


PROTOCOL_STREAM = sys.stdout
# Heavy modules and the existing pipeline use print for useful diagnostics.
# Keep stdout exclusively machine-readable and send those diagnostics to the
# manager's stderr/log reader instead.
sys.stdout = sys.stderr
print(f"CV worker Python: {sys.executable}", flush=True)

from launch_config import build_tracker_arguments, default_launch_values
from session_lock import CvRuntimeLock


_write_lock = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit(**payload) -> None:
    message = {"type": "status", "timestamp": utc_now(), **payload}
    with _write_lock:
        PROTOCOL_STREAM.write(json.dumps(message, separators=(",", ":")) + "\n")
        PROTOCOL_STREAM.flush()


def main() -> int:
    command_queue: queue.Queue[dict] = queue.Queue()
    stop_event = threading.Event()
    shutdown_event = threading.Event()
    state_lock = threading.Lock()
    state = {"value": "offline", "run_id": None}

    def set_state(value: str, **extra) -> None:
        with state_lock:
            state["value"] = value
            if "run_id" in extra:
                state["run_id"] = extra["run_id"]
            snapshot_run_id = state["run_id"]
        emit(state=value, run_id=snapshot_run_id, **{k: v for k, v in extra.items() if k != "run_id"})

    def read_commands() -> None:
        for line in sys.stdin:
            try:
                command = json.loads(line)
            except json.JSONDecodeError:
                emit(state="failed", error="Worker received invalid JSON command.")
                continue
            name = command.get("command")
            if name == "stop":
                with state_lock:
                    current_state = state["value"]
                if current_state in {"starting", "running"}:
                    set_state("stopping")
                    stop_event.set()
                elif current_state == "ready":
                    set_state("ready")
            elif name == "shutdown":
                shutdown_event.set()
                stop_event.set()
                command_queue.put(command)
                return
            elif name == "start":
                command_queue.put(command)
            else:
                emit(state="failed", error=f"Unknown worker command: {name!r}")

        # EOF means the FastAPI manager disappeared without completing the
        # normal shutdown handshake (for example, a terminal was closed or a
        # second Ctrl+C forced Uvicorn to exit during model loading). The main
        # thread may be blocked inside native model initialization, so events
        # alone cannot guarantee timely cleanup. Exit immediately; the OS then
        # releases cameras, CUDA resources, and the runtime lock.
        os._exit(0)

    reader = threading.Thread(target=read_commands, name="cv-worker-command-reader", daemon=True)
    reader.start()

    models = None
    fatal_error = False
    try:
        with CvRuntimeLock("dashboard CV worker"):
            set_state("loading", loading_stage="Reading production configuration", error=None)
            values = default_launch_values()
            arguments = build_tracker_arguments(values)

            from main_tracker import parse_args, preload_models, run_pipeline

            args = parse_args(arguments)
            models = preload_models(
                args,
                loading_stage=lambda stage: set_state(
                    "loading", loading_stage=stage, error=None
                ),
            )
            set_state("ready", loading_stage="Complete", error=None)

            while not shutdown_event.is_set():
                command = command_queue.get()
                if command.get("command") == "shutdown":
                    break
                if command.get("command") != "start":
                    continue
                with state_lock:
                    if state["value"] != "ready":
                        current_state = state["value"]
                    else:
                        current_state = None
                if current_state is not None:
                    emit(
                        state=current_state,
                        run_id=state["run_id"],
                        error=f"Cannot start while worker is {current_state}.",
                    )
                    continue

                session_args = copy.copy(args)
                requested_run_id = str(command.get("run_id") or session_args.run_id).strip()
                session_args.run_id = requested_run_id or session_args.run_id
                stop_event.clear()
                set_state(
                    "starting",
                    run_id=session_args.run_id,
                    started_at=utc_now(),
                    stopped_at=None,
                    error=None,
                )
                try:
                    def mark_running() -> None:
                        if stop_event.is_set():
                            set_state("stopping", run_id=session_args.run_id)
                        else:
                            set_state("running", run_id=session_args.run_id, error=None)

                    run_pipeline(
                        session_args,
                        models,
                        stop_event=stop_event,
                        started_callback=mark_running,
                    )
                except Exception as exc:
                    set_state(
                        "failed",
                        run_id=session_args.run_id,
                        stopped_at=utc_now(),
                        error=str(exc),
                    )
                    traceback.print_exc(file=sys.stderr)
                    continue
                set_state(
                    "ready",
                    run_id=session_args.run_id,
                    stopped_at=utc_now(),
                    loading_stage="Complete",
                    error=None,
                )
    except Exception as exc:
        fatal_error = True
        emit(state="failed", error=str(exc), stopped_at=utc_now())
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        if models is not None:
            models.close()
        if not fatal_error:
            emit(state="offline", stopped_at=utc_now())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
