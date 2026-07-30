import unittest
from pathlib import Path

import yaml
from ultralytics.trackers.utils.gmc import GMC

from constants import DEFAULT_TRACKER_CONFIG_PATH


class TrackerConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_path = Path(__file__).resolve().parent / DEFAULT_TRACKER_CONFIG_PATH
        cls.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    def test_default_tracker_keeps_botsort_appearance_association(self):
        self.assertEqual(self.config["tracker_type"], "botsort")
        self.assertTrue(self.config["with_reid"])
        self.assertEqual(self.config["model"], "auto")

    def test_fixed_camera_default_disables_global_motion_compensation(self):
        gmc = GMC(self.config["gmc_method"])
        self.assertIsNone(gmc.method)


if __name__ == "__main__":
    unittest.main()
