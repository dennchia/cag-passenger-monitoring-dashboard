from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

import crud
from config import Settings, settings
from database import SessionLocal
from models import MetricLogCreate, SystemAlertCreate, TacticalStateCreate
from tactical_state import tactical_store

try:
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover - keeps the app bootable until requirements are installed.
    mqtt = None


logger = logging.getLogger(__name__)


class MqttBridge:
    def __init__(self, app_settings: Settings):
        self.settings = app_settings
        self.client = None
        self._lock = threading.Lock()
        self._last_metric_log_at = 0.0
        self._latest_zone_counts_by_run: dict[str, dict[str, int]] = {}
        self._latest_camera_online_count_by_run: dict[str, int] = {}

    def start(self) -> None:
        if not self.settings.mqtt_enabled:
            logger.info("MQTT bridge disabled.")
            return
        if mqtt is None:
            logger.error("MQTT bridge enabled but paho-mqtt is not installed.")
            return

        self.client = self._create_client()
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        if self.settings.mqtt_username:
            self.client.username_pw_set(
                self.settings.mqtt_username,
                self.settings.mqtt_password or None,
            )

        logger.info("Starting MQTT bridge to %s:%s.", self.settings.mqtt_host, self.settings.mqtt_port)
        self.client.connect_async(self.settings.mqtt_host, self.settings.mqtt_port, keepalive=60)
        self.client.loop_start()

    def stop(self) -> None:
        if self.client is None:
            return
        logger.info("Stopping MQTT bridge.")
        self.client.loop_stop()
        self.client.disconnect()
        self.client = None

    def _create_client(self):
        try:
            return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="cag-fastapi-bridge")
        except (AttributeError, TypeError):
            return mqtt.Client(client_id="cag-fastapi-bridge")

    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        logger.info("MQTT bridge connected with result %s.", reason_code)
        for topic in self._topics():
            client.subscribe(topic, qos=1)
            logger.info("MQTT bridge subscribed to %s.", topic)

    def _on_disconnect(self, client, userdata, reason_code, properties=None) -> None:
        logger.warning("MQTT bridge disconnected with result %s. Paho will attempt reconnect.", reason_code)

    def _on_message(self, client, userdata, message) -> None:
        topic = message.topic
        try:
            payload = json.loads(message.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring invalid MQTT JSON on %s: %s", topic, exc)
            return

        try:
            if topic == self.settings.mqtt_topic_metrics:
                self.handle_metrics_payload(payload)
            elif topic == self.settings.mqtt_topic_tactical:
                self.handle_tactical_payload(payload)
            elif topic == self.settings.mqtt_topic_alerts:
                self.handle_alert_payload(payload)
            else:
                logger.debug("Ignoring MQTT message on unhandled topic %s.", topic)
        except Exception:
            logger.exception("MQTT bridge failed while handling topic %s.", topic)

    def handle_metrics_payload(self, payload: dict[str, Any]) -> None:
        metric = self._metric_from_payload(payload)
        if metric is None:
            return

        now = time.monotonic()
        with self._lock:
            if now - self._last_metric_log_at < self.settings.mqtt_metric_log_interval_seconds:
                return
            self._last_metric_log_at = now

        with SessionLocal() as db:
            crud.create_metric_log(db, metric)

    def handle_tactical_payload(self, payload: dict[str, Any]) -> None:
        try:
            tactical_store.update(TacticalStateCreate(**payload))
        except ValidationError as exc:
            logger.warning("Ignoring invalid MQTT tactical payload: %s", exc)

    def handle_alert_payload(self, payload: dict[str, Any]) -> None:
        try:
            alert = SystemAlertCreate(**payload)
        except ValidationError as exc:
            logger.warning("Ignoring invalid MQTT alert payload: %s", exc)
            return

        with SessionLocal() as db:
            crud.create_system_alert(db, alert)

    def _metric_from_payload(self, payload: dict[str, Any]) -> MetricLogCreate | None:
        run_id = str(payload.get("run_id") or "default")
        incoming_zone_counts = self._extract_zone_counts(payload)

        with self._lock:
            if incoming_zone_counts:
                run_zone_counts = self._latest_zone_counts_by_run.setdefault(run_id, {})
                run_zone_counts.update(incoming_zone_counts)
            else:
                run_zone_counts = self._latest_zone_counts_by_run.setdefault(run_id, {})

            if payload.get("camera_online_count") is not None:
                try:
                    self._latest_camera_online_count_by_run[run_id] = max(0, int(payload["camera_online_count"]))
                except (TypeError, ValueError):
                    pass

            zone_counts = dict(run_zone_counts)
            camera_online_count = self._latest_camera_online_count_by_run.get(run_id)

        try:
            passenger_count = max(0, int(payload.get("passenger_count", 0)))
        except (TypeError, ValueError):
            passenger_count = 0

        if payload.get("passenger_count") is None and zone_counts:
            passenger_count = sum(zone_counts.values())

        try:
            return MetricLogCreate(
                passenger_count=passenger_count,
                run_id=run_id,
                zone_counts=zone_counts or payload.get("zone_counts"),
                camera_online_count=camera_online_count,
                timestamp=payload.get("timestamp") or datetime.now(timezone.utc),
            )
        except ValidationError as exc:
            logger.warning("Ignoring invalid MQTT metric payload: %s", exc)
            return None

    def _extract_zone_counts(self, payload: dict[str, Any]) -> dict[str, int]:
        zone_counts = payload.get("zone_counts")
        if not isinstance(zone_counts, dict):
            camera_id = payload.get("camera_id")
            if camera_id:
                zone_counts = {str(camera_id): payload.get("passenger_count", 0)}
            else:
                return {}

        normalized: dict[str, int] = {}
        for zone_id, value in zone_counts.items():
            try:
                count = max(0, int(value))
            except (TypeError, ValueError):
                continue
            normalized[str(zone_id)] = count
        return normalized

    def _topics(self) -> list[str]:
        return [
            self.settings.mqtt_topic_metrics,
            self.settings.mqtt_topic_tactical,
            self.settings.mqtt_topic_alerts,
        ]


mqtt_bridge = MqttBridge(settings)
