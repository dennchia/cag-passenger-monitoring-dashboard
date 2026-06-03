from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MetricLog(Base):
    __tablename__ = "metric_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    run_id: Mapped[str] = mapped_column(String(80), default="default", index=True)
    passenger_count: Mapped[int] = mapped_column(Integer)
    zone_counts: Mapped[str | None] = mapped_column(Text, nullable=True)
    camera_online_count: Mapped[int | None] = mapped_column(Integer, nullable=True)


class SystemAlert(Base):
    __tablename__ = "system_alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    run_id: Mapped[str] = mapped_column(String(80), default="default", index=True)
    severity: Mapped[str] = mapped_column(String(32), default="info", index=True)
    message: Mapped[str] = mapped_column(Text)


class PassengerObservation(Base):
    __tablename__ = "passenger_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    run_id: Mapped[str] = mapped_column(String(80), default="default", index=True)
    camera_id: Mapped[str] = mapped_column(String(80), index=True)
    track_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    age: Mapped[float] = mapped_column(Float)
    gender: Mapped[str] = mapped_column(String(32), index=True)
    age_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    gender_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    image_path: Mapped[str] = mapped_column(Text)
    image_url: Mapped[str] = mapped_column(Text)


class MetricLogCreate(BaseModel):
    passenger_count: int = Field(ge=0)
    run_id: str = "default"
    zone_counts: Any | None = None
    camera_online_count: int | None = Field(default=None, ge=0)
    timestamp: datetime | None = None


class MetricLogRead(BaseModel):
    id: int
    timestamp: datetime
    run_id: str
    passenger_count: int
    zone_counts: str | None = None
    camera_online_count: int | None = None

    model_config = {"from_attributes": True}


class SystemAlertCreate(BaseModel):
    severity: str = "info"
    message: str
    run_id: str = "default"
    timestamp: datetime | None = None


class SystemAlertRead(BaseModel):
    id: int
    timestamp: datetime
    run_id: str
    severity: str
    message: str

    model_config = {"from_attributes": True}


class PassengerObservationRead(BaseModel):
    id: int
    timestamp: datetime
    run_id: str
    camera_id: str
    track_id: str | None = None
    age: float
    gender: str
    age_confidence: float | None = None
    gender_confidence: float | None = None
    image_url: str

    model_config = {"from_attributes": True}


class PassengerObservationSummary(BaseModel):
    total_analyzed: int = 0
    males: int = 0
    females: int = 0
    unknown: int = 0
    minors: int = 0
