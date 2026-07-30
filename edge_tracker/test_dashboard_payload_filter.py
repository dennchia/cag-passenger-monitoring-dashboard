import unittest
from types import SimpleNamespace

from main_tracker import build_payloads


class _Capture:
    def is_opened(self):
        return True


def person(identity_id, role, center, state="confirmed", sources=("cam_1", "cam_2")):
    return {
        "center": center,
        "sources": list(sources),
        "observations": [],
        "identity_id": identity_id,
        "temporary_group_id": None,
        "identity_state": state,
        "role": role,
    }


class DashboardPayloadFilterTests(unittest.TestCase):
    def test_only_confirmed_roles_are_published_and_only_evacuees_are_counted(self):
        contexts = [
            SimpleNamespace(camera_id="cam_1", tactical_points=[], cap=_Capture()),
            SimpleNamespace(camera_id="cam_2", tactical_points=[], cap=_Capture()),
        ]
        args = SimpleNamespace(
            camera_id="fused",
            run_id="test",
            map_size_cm=300,
            mqtt_send_map_image=False,
            mqtt_image_quality=80,
        )
        fused_people = [
            person(1, "evacuee", (100, 100)),
            person(2, "cag", (120, 120)),
            person(3, "scdf", (350, 120), sources=("cam_2",)),
            person(None, None, (140, 140), state=None, sources=("cam_1",)),
            person(4, "evacuee", (160, 160), state="provisional"),
        ]

        tactical, metrics = build_payloads(contexts, args, fused_people)

        self.assertEqual(tactical["people_count"], 1)
        self.assertEqual(metrics["passenger_count"], 1)
        self.assertEqual(len(tactical["positions_cm"]), 3)
        self.assertEqual(
            {position["role"] for position in tactical["positions_cm"]},
            {"evacuee", "cag", "scdf"},
        )
        self.assertNotIn("temporary_group_id", tactical["positions_cm"][0])
        self.assertEqual(tactical["zone_counts"], {"cam_1": 1, "cam_2": 1})


if __name__ == "__main__":
    unittest.main()
