"""Guards that stop one person's crop landing in another person's gallery.

Three separate paths allowed a location-only guess to become permanent:

* an unpromoted group wrote its crops to the evidence folder immediately, so
  two people who merely walked close together shared a folder;
* the stable-location fallback confirmed a member that appearance had already
  argued against; and
* refusing a merge left the group free to be promoted into a second master for
  a person who already had one.

These tests cover all three, and pin the behaviour that must survive: a group
that appearance vouches for still gets its crops, and a member appearance never
judged is still confirmable by location alone.
"""

import unittest

import numpy as np

from reid_memory import AppearanceIdentityMemory


def make_slot(camera_id, seed=1.0, frame_index=1):
    feature = np.full(8, float(seed), dtype=np.float32)
    feature /= np.linalg.norm(feature)
    return {
        "feature": feature,
        "feature_source": "transreid",
        "feature_space_id": "test-feature-space",
        "feature_dimension": int(feature.size),
        "image_path": None,
        "digest": None,
        "captured_frame": int(frame_index),
        "captured_at": float(frame_index),
        "camera_id": camera_id,
        "sharpness": 200.0,
        "detection_confidence": 0.95,
    }


class MemoryTestCase(unittest.TestCase):
    def setUp(self):
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            enable_role_classification=False,
            enable_demographics=False,
            provisional_location_confirm_frames=3,
            start_worker=False,
        )
        # Capture evidence writes instead of spawning the writer subprocess.
        self.saved = []
        self.memory._queue_evidence_save = self.saved.append

    def tearDown(self):
        self.memory.close(drain=False)

    def make_provisional(self, identity_id=-1, member_keys=()):
        record = self.memory._new_record(identity_state="provisional")
        record["member_track_keys"] = set(member_keys)
        record["global_reid_checked_track_keys"] = set(member_keys)
        self.memory.identities[identity_id] = record
        for key in member_keys:
            self.memory.track_to_identity[key] = identity_id
        return record


class DeferredEvidenceTests(MemoryTestCase):
    """An unpromoted group is a geometric guess, not a person."""

    def test_unpromoted_group_writes_no_image(self):
        record = self.make_provisional()
        self.memory._defer_provisional_evidence_locked(
            record, {"identity_id": -1, "slot_name": "baseline_cam_1",
                     "crop": None, "output_path": "/ev/Temporary_0001/a.png"}
        )
        self.assertEqual(self.saved, [])
        self.assertEqual(len(record["deferred_evidence_tasks"]), 1)

    def test_promotion_releases_images_into_the_master_folder(self):
        record = self.make_provisional(member_keys=[("cam_1", 1)])
        record["camera_baselines"] = {"cam_1": make_slot("cam_1")}
        record["camera_baselines"]["cam_1"]["image_path"] = "/ev/Temporary_0001/a.png"
        self.memory._defer_provisional_evidence_locked(
            record, {"identity_id": -1, "slot_name": "baseline_cam_1",
                     "crop": None, "output_path": "/ev/Temporary_0001/a.png"}
        )

        with self.memory._lock:
            master_id = self.memory._promote_provisional_locked(-1, "stable_location")

        self.assertIsNotNone(master_id)
        self.assertEqual(len(self.saved), 1)
        # Re-addressed from the temporary folder to the master's folder...
        self.assertIn(f"Master_{master_id:04d}", self.saved[0]["output_path"])
        self.assertEqual(self.saved[0]["identity_id"], master_id)
        # ...and the slot must follow, or the saved digest never finds it.
        self.assertEqual(
            self.memory.identities[master_id]["camera_baselines"]["cam_1"]["image_path"],
            self.saved[0]["output_path"],
        )

    def test_group_torn_down_before_promotion_never_writes(self):
        record = self.make_provisional()
        self.memory._defer_provisional_evidence_locked(
            record, {"identity_id": -1, "slot_name": "baseline_cam_1",
                     "crop": None, "output_path": "/ev/Temporary_0001/a.png"}
        )
        self.memory.identities.pop(-1)
        self.assertEqual(self.saved, [])


class StableLocationOverrideTests(MemoryTestCase):
    """Standing in the right place cannot overrule an appearance rejection."""

    def setUp(self):
        super().setUp()
        self.key = ("cam_1", 11)
        self.record = self.memory._new_record()
        self.record["pending_member_keys"] = {self.key}
        self.record["gallery"]["baseline"] = make_slot("cam_2")
        self.memory.identities[1] = self.record
        self.memory.track_to_identity[self.key] = 1

    def _confirm(self, reason):
        with self.memory._lock:
            return self.memory._confirm_pending_members_locked(
                1, reason, appearance_confirmed=(reason == "global_reid")
            )

    def test_rejected_member_is_not_confirmed_by_location(self):
        self.record["appearance_rejected_member_keys"] = {self.key}
        self.assertFalse(self._confirm("stable_location"))
        self.assertIn(self.key, self.record["pending_member_keys"])

    def test_unjudged_member_is_still_confirmed_by_location(self):
        self.assertTrue(self._confirm("stable_location"))
        self.assertNotIn(self.key, self.record["pending_member_keys"])

    def test_challenged_member_is_not_confirmed_by_location(self):
        self.record["pending_member_keys"] = set()
        self.record["challenged_member_keys"] = {self.key}
        self.record["appearance_rejected_member_keys"] = {self.key}
        self.assertFalse(self._confirm("stable_location"))
        self.assertIn(self.key, self.record["challenged_member_keys"])

    def test_positive_reid_can_still_confirm_a_rejected_member(self):
        self.record["appearance_rejected_member_keys"] = {self.key}
        self.assertTrue(self._confirm("global_reid"))


class BlockedMergePromotionTests(MemoryTestCase):
    """Refusing a merge must not manufacture a second master instead."""

    def setUp(self):
        super().setUp()
        self.incumbent = ("cam_1", 10)
        self.member = ("cam_1", 54)
        self.memory.identities[5] = self.memory._new_record()
        self.memory.track_to_identity[self.incumbent] = 5

        self.record = self.make_provisional(identity_id=-10, member_keys=[self.member])
        self.record["camera_baselines"] = {"cam_1": make_slot("cam_1")}
        self.record["merge_blocked_by_master"] = 5

    def _promote(self):
        with self.memory._lock:
            return self.memory._promote_provisional_locked(-10, "stable_location")

    def test_no_new_master_while_the_incumbent_is_visible(self):
        self.memory.visible_track_keys_by_camera["cam_1"] = {self.incumbent, self.member}
        self.assertIsNone(self._promote())
        self.assertIn(-10, self.memory.identities)

    def test_promotion_resumes_once_the_incumbent_disappears(self):
        self.memory.visible_track_keys_by_camera["cam_1"] = {self.member}
        master_id = self._promote()
        self.assertIsNotNone(master_id)
        self.assertGreater(master_id, 0)
        self.assertNotIn("merge_blocked_by_master", self.memory.identities[master_id])

    def test_unblocked_group_promotes_normally(self):
        self.record.pop("merge_blocked_by_master")
        self.memory.visible_track_keys_by_camera["cam_1"] = {self.incumbent, self.member}
        self.assertIsNotNone(self._promote())


if __name__ == "__main__":
    unittest.main()
