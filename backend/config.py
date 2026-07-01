from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    camera_url: str = Field(
        "rtsp://username:password@192.168.50.192:554/Streaming/Channels/101",
        validation_alias="CAMERA_URL",
    )
    camera_urls: str = Field("", validation_alias="CAMERA_URLS")
    primary_camera_id: str = Field("cam_1", validation_alias="PRIMARY_CAMERA_ID")
    camera_reconnect_seconds: int = Field(5, validation_alias="CAMERA_RECONNECT_SECONDS")
    camera_jpeg_quality: int = Field(80, validation_alias="CAMERA_JPEG_QUALITY")
    sqlite_db_path: str = Field("./passenger_monitoring.db", validation_alias="SQLITE_DB_PATH")
    observation_upload_dir: str = Field("./uploads/observations", validation_alias="OBSERVATION_UPLOAD_DIR")
    frontend_dist_dir: str = Field("../frontend/dist", validation_alias="FRONTEND_DIST_DIR")
    zone_capacities_json: str = Field('{"cam_1":150,"cam_2":150}', validation_alias="ZONE_CAPACITIES_JSON")
    mqtt_enabled: bool = Field(False, validation_alias="MQTT_ENABLED")
    mqtt_host: str = Field("localhost", validation_alias="MQTT_HOST")
    mqtt_port: int = Field(1883, validation_alias="MQTT_PORT")
    mqtt_username: str = Field("", validation_alias="MQTT_USERNAME")
    mqtt_password: str = Field("", validation_alias="MQTT_PASSWORD")
    mqtt_topic_metrics: str = Field("cag/metrics", validation_alias="MQTT_TOPIC_METRICS")
    mqtt_topic_tactical: str = Field("cag/tactical", validation_alias="MQTT_TOPIC_TACTICAL")
    mqtt_topic_alerts: str = Field("cag/alerts", validation_alias="MQTT_TOPIC_ALERTS")
    mqtt_metric_log_interval_seconds: float = Field(1.0, validation_alias="MQTT_METRIC_LOG_INTERVAL_SECONDS")
    cors_origins: str = Field(
        "http://localhost:5173,http://localhost:3000",
        validation_alias="CORS_ORIGINS",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def zone_capacity_map(self) -> dict[str, int]:
        try:
            raw_value = json.loads(self.zone_capacities_json)
        except json.JSONDecodeError:
            return {}

        if not isinstance(raw_value, dict):
            return {}

        capacities: dict[str, int] = {}
        for zone_id, capacity in raw_value.items():
            try:
                normalized_capacity = int(capacity)
            except (TypeError, ValueError):
                continue
            if str(zone_id).strip() and normalized_capacity > 0:
                capacities[str(zone_id).strip()] = normalized_capacity
        return capacities

    @property
    def camera_source_map(self) -> dict[str, str]:
        sources: dict[str, str] = {}
        entries = [entry.strip() for entry in self.camera_urls.split(",") if entry.strip()]

        for index, entry in enumerate(entries, start=1):
            if "=" in entry:
                camera_id, camera_url = entry.split("=", 1)
                camera_id = camera_id.strip()
            else:
                camera_id = f"cam_{index}"
                camera_url = entry

            camera_url = camera_url.strip()
            if camera_id and camera_url:
                sources[camera_id] = camera_url

        if not sources and self.camera_url:
            sources[self.primary_camera_id] = self.camera_url

        return sources

    @property
    def observation_upload_path(self) -> Path:
        configured = Path(self.observation_upload_dir)
        if configured.is_absolute():
            return configured
        return Path(__file__).resolve().parent / configured

    @property
    def frontend_dist_path(self) -> Path:
        configured = Path(self.frontend_dist_dir)
        if configured.is_absolute():
            return configured
        return Path(__file__).resolve().parent / configured


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
