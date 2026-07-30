import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request


BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

import cv_api
from cv_api import CvSessionStart, CvStatus


def request_from(host):
    return Request({"type": "http", "method": "POST", "path": "/", "headers": [], "client": (host, 1234)})


class FakeManager:
    def __init__(self):
        self.start_calls = []
        self.stop_calls = 0

    def status(self):
        return {
            "state": "ready",
            "ready": True,
            "running": False,
            "run_id": None,
            "started_at": None,
            "stopped_at": None,
            "pid": 123,
            "loading_stage": "Complete",
            "error": None,
            "mqtt_broker_reachable": True,
        }

    def start_session(self, run_id=None):
        self.start_calls.append(run_id)

    def stop_session(self):
        self.stop_calls += 1


class CvApiTests(unittest.TestCase):
    def setUp(self):
        self.manager = FakeManager()
        self.manager_patch = patch.object(cv_api, "cv_manager", self.manager)
        self.manager_patch.start()

    def tearDown(self):
        self.manager_patch.stop()

    def test_status_contract(self):
        status = cv_api.get_cv_status(request_from("127.0.0.1"), None)
        validated = CvStatus(**status)
        self.assertTrue(validated.ready)
        self.assertTrue(validated.control_allowed)

    def test_local_start_and_stop_are_allowed(self):
        request = request_from("::1")
        cv_api.start_cv_session(CvSessionStart(run_id="field_test_1"), request, None)
        cv_api.stop_cv_session(request, None)
        self.assertEqual(self.manager.start_calls, ["field_test_1"])
        self.assertEqual(self.manager.stop_calls, 1)

    def test_remote_control_is_denied_by_default(self):
        with patch.object(cv_api.settings, "cv_control_allow_lan", False):
            with self.assertRaises(HTTPException) as context:
                cv_api.start_cv_session(
                    CvSessionStart(run_id="field_test_1"),
                    request_from("192.168.50.20"),
                    None,
                )
        self.assertEqual(context.exception.status_code, 403)

    def test_remote_control_requires_configured_token(self):
        token = SimpleNamespace(get_secret_value=lambda: "correct-token")
        with patch.object(cv_api.settings, "cv_control_allow_lan", True), patch.object(
            cv_api.settings, "cv_control_token", token
        ):
            with self.assertRaises(HTTPException) as context:
                cv_api.start_cv_session(
                    CvSessionStart(run_id="field_test_1"),
                    request_from("192.168.50.20"),
                    "wrong-token",
                )
            cv_api.start_cv_session(
                CvSessionStart(run_id="field_test_1"),
                request_from("192.168.50.20"),
                "correct-token",
            )
        self.assertEqual(context.exception.status_code, 401)


if __name__ == "__main__":
    unittest.main()
