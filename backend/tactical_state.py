from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock

from models import TacticalStateCreate, TacticalStateRead


STALE_AFTER_SECONDS = 5.0


class TacticalStateStore:
    def __init__(self, stale_after_seconds: float = STALE_AFTER_SECONDS):
        self.stale_after_seconds = stale_after_seconds
        self._lock = Lock()
        self._states: dict[tuple[str, str], TacticalStateRead] = {}

    def update(self, payload: TacticalStateCreate) -> TacticalStateRead:
        now = datetime.now(timezone.utc)
        state = TacticalStateRead(
            timestamp=payload.timestamp or now,
            received_at=now,
            camera_id=payload.camera_id,
            run_id=payload.run_id,
            camera_source=payload.camera_source,
            people_count=payload.people_count,
            positions_cm=payload.positions_cm,
            map_size_cm=payload.map_size_cm,
            has_data=True,
            stale=False,
            age_seconds=0,
        )
        with self._lock:
            self._states[(payload.camera_id, payload.run_id)] = state
        return state

    def latest(self, camera_id: str | None = None, run_id: str | None = None) -> TacticalStateRead:
        with self._lock:
            candidates = [
                state
                for state in self._states.values()
                if (not camera_id or state.camera_id == camera_id) and (not run_id or state.run_id == run_id)
            ]

        if not candidates:
            return TacticalStateRead(camera_id=camera_id, run_id=run_id)

        latest_state = max(candidates, key=lambda state: state.received_at or datetime.min.replace(tzinfo=timezone.utc))
        return self._with_stale_status(latest_state)

    def _with_stale_status(self, state: TacticalStateRead) -> TacticalStateRead:
        if state.received_at is None:
            return state.model_copy(update={"stale": True, "age_seconds": None})

        age_seconds = max(0.0, (datetime.now(timezone.utc) - state.received_at).total_seconds())
        return state.model_copy(
            update={
                "age_seconds": round(age_seconds, 1),
                "stale": age_seconds > self.stale_after_seconds,
            }
        )


tactical_store = TacticalStateStore()
