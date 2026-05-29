from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, desc, select
from sqlalchemy.orm import Session

from models import MetricLog, MetricLogCreate, PassengerObservation, SystemAlert, SystemAlertCreate


def _timestamp_or_now(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    return value


def _zone_counts_to_text(value: Any | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def create_metric_log(db: Session, payload: MetricLogCreate) -> MetricLog:
    metric = MetricLog(
        timestamp=_timestamp_or_now(payload.timestamp),
        run_id=payload.run_id,
        passenger_count=payload.passenger_count,
        zone_counts=_zone_counts_to_text(payload.zone_counts),
        camera_online_count=payload.camera_online_count,
    )
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric


def get_latest_metrics(db: Session, run_id: str | None = None, limit: int = 10) -> list[MetricLog]:
    statement = select(MetricLog)
    if run_id:
        statement = statement.where(MetricLog.run_id == run_id)
    statement = statement.order_by(desc(MetricLog.timestamp), desc(MetricLog.id)).limit(limit)
    return list(db.scalars(statement).all())


def create_system_alert(db: Session, payload: SystemAlertCreate) -> SystemAlert:
    alert = SystemAlert(
        timestamp=_timestamp_or_now(payload.timestamp),
        run_id=payload.run_id,
        severity=payload.severity,
        message=payload.message,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


def get_latest_alerts(db: Session, run_id: str | None = None, limit: int = 5) -> list[SystemAlert]:
    statement = select(SystemAlert)
    if run_id:
        statement = statement.where(SystemAlert.run_id == run_id)
    statement = statement.order_by(desc(SystemAlert.timestamp), desc(SystemAlert.id)).limit(limit)
    return list(db.scalars(statement).all())


def create_passenger_observation(
    db: Session,
    *,
    timestamp: datetime | None,
    run_id: str,
    camera_id: str,
    age: float,
    gender: str,
    image_path: str,
    image_url: str,
    track_id: str | None = None,
    age_confidence: float | None = None,
    gender_confidence: float | None = None,
) -> PassengerObservation:
    observation = PassengerObservation(
        timestamp=_timestamp_or_now(timestamp),
        run_id=run_id,
        camera_id=camera_id,
        track_id=track_id,
        age=age,
        gender=gender.strip().lower(),
        age_confidence=age_confidence,
        gender_confidence=gender_confidence,
        image_path=image_path,
        image_url=image_url,
    )
    db.add(observation)
    db.commit()
    db.refresh(observation)
    return observation


def get_latest_observations(
    db: Session,
    *,
    gender: str | None = None,
    min_age: float | None = None,
    max_age: float | None = None,
    camera_id: str | None = None,
    run_id: str | None = None,
    limit: int = 50,
) -> list[PassengerObservation]:
    statement = select(PassengerObservation)
    if gender:
        statement = statement.where(PassengerObservation.gender == gender.strip().lower())
    if min_age is not None:
        statement = statement.where(PassengerObservation.age >= min_age)
    if max_age is not None:
        statement = statement.where(PassengerObservation.age <= max_age)
    if camera_id:
        statement = statement.where(PassengerObservation.camera_id == camera_id)
    if run_id:
        statement = statement.where(PassengerObservation.run_id == run_id)

    statement = statement.order_by(desc(PassengerObservation.timestamp), desc(PassengerObservation.id)).limit(limit)
    return list(db.scalars(statement).all())


def clear_passenger_observations(db: Session) -> int:
    count = len(list(db.scalars(select(PassengerObservation.id)).all()))
    db.execute(delete(PassengerObservation))
    db.commit()
    return count
