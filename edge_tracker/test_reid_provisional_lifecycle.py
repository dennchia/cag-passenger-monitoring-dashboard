import unittest

import numpy as np

from constants import REID_SEMANTIC_SLOTS
from reid_memory import AppearanceIdentityMemory


class FixedExtractor:
    def __init__(self, feature):
        self.feature = np.asarray(feature, dtype=np.float32)

    def extract_many_aligned(self, crops):
        return [self.feature.copy() for _ in crops]

    @staticmethod
    def feature_space_id(_dimension):
        return "test-feature-space"


def make_slot(feature, camera_id, frame_index=1):
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
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


class ProvisionalIdentityLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            enable_role_classification=False,
            enable_demographics=False,
            provisional_location_confirm_frames=3,
            start_worker=False,
        )
        self.left_key = ("cam_1", 10)
        self.right_key = ("cam_2", 20)

    def tearDown(self):
        self.memory.close(drain=False)

    def create_pair(self):
        return self.memory.create_provisional_pair(self.left_key, self.right_key)

    def seed_angle(self, identity_id, camera_id, orientation, feature):
        with self.memory._lock:
            record = self.memory.identities[identity_id]
            camera_gallery = record["camera_views"].setdefault(
                camera_id,
                {slot_name: None for slot_name in REID_SEMANTIC_SLOTS},
            )
            camera_gallery[orientation] = make_slot(feature, camera_id)
            record.setdefault("global_reid_checked_track_keys", set()).update(
                key
                for key in record.get("member_track_keys", ())
                if key[0] == camera_id
            )

    def process_delayed_location_assigned_task(
        self,
        extractor_feature,
        task_provisional_identity_id=None,
    ):
        self.memory.close(drain=False)
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            reid_extractor=FixedExtractor(extractor_feature),
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=False,
        )
        identity_id = 7
        track_key = ("cam_2", 18)
        record = self.memory._new_record(identity_state="confirmed")
        record["gallery"]["baseline"] = make_slot((1.0, 0.0), "cam_1")
        record["location_managed"] = True
        record["member_track_keys"].add(track_key)
        record["confirmation_reason"] = "stable_location"
        self.memory.identities[identity_id] = record
        self.memory.next_identity_id = 8
        self.memory.track_to_identity[track_key] = identity_id
        self.memory.pending_intake[track_key] = {
            "submitted": True,
            "generation": 1,
        }

        checker = np.indices((64, 32)).sum(axis=0) % 2
        crop = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        task = {
            "type": "intake",
            "track_key": track_key,
            "camera_id": "cam_2",
            "frame_index": 10,
            "samples": [
                {
                    "crop": crop,
                    "frame_index": 10,
                    "camera_id": "cam_2",
                    "observed_at": 1.0,
                    "sharpness": 200.0,
                    "area": float(crop.shape[0] * crop.shape[1]),
                    "detection_confidence": 0.95,
                    "orientation": "front",
                    "map_point": (0.0, 0.0),
                }
            ],
            # This is the stale snapshot that previously prevented ID 7
            # from being compared and caused a new ID to be created.
            "excluded_identity_ids": {identity_id},
            "same_camera_peer_keys": set(),
            "generation": 1,
            "handoff_identity_id": None,
            "handoff_from_key": None,
            "provisional_identity_id": task_provisional_identity_id,
        }
        self.memory._process_intake_task(task)
        return identity_id, track_key

    def test_create_provisional_pair_uses_unnumbered_temporary_group(self):
        temporary_id = self.create_pair()

        self.assertLess(temporary_id, 0)
        self.assertEqual(self.memory.identity_state(temporary_id), "provisional")
        self.assertEqual(self.memory.lookup_track_key(self.left_key), temporary_id)
        self.assertEqual(self.memory.lookup_track_key(self.right_key), temporary_id)
        self.assertIsNone(self.memory.lookup(10, camera_id="cam_1"))
        self.assertEqual(self.memory.temporary_group(10, camera_id="cam_1"), "tmp_1")
        self.assertEqual(
            self.memory.create_provisional_pair(self.right_key, self.left_key),
            temporary_id,
        )
        self.assertEqual(self.memory.next_identity_id, 1)

    def test_camera_specific_same_angle_slots_can_coexist(self):
        identity_id = self.create_pair()
        self.seed_angle(identity_id, "cam_1", "front", (1.0, 0.0))
        self.seed_angle(identity_id, "cam_2", "front", (0.0, 1.0))

        record = self.memory.identities[identity_id]
        cam_1_front = record["camera_views"]["cam_1"]["front"]
        cam_2_front = record["camera_views"]["cam_2"]["front"]

        self.assertIsNot(cam_1_front, cam_2_front)
        np.testing.assert_array_equal(cam_1_front["feature"], np.array([1.0, 0.0]))
        np.testing.assert_array_equal(cam_2_front["feature"], np.array([0.0, 1.0]))

    def test_matching_same_angle_allocates_master_after_global_search(self):
        temporary_id = self.create_pair()
        self.seed_angle(temporary_id, "cam_1", "front", (1.0, 0.0))
        self.seed_angle(temporary_id, "cam_2", "front", (0.99, 0.10))

        with self.memory._lock:
            master_id = self.memory._evaluate_provisional_evidence_locked(temporary_id)

        self.assertEqual(master_id, 1)
        self.assertNotIn(temporary_id, self.memory.identities)
        self.assertEqual(self.memory.identity_state(master_id), "confirmed")
        self.assertEqual(self.memory.lookup_track_key(self.left_key), master_id)
        self.assertEqual(self.memory.lookup_track_key(self.right_key), master_id)
        self.assertEqual(
            self.memory.identities[master_id]["confirmation_reason"],
            "same_angle_reid",
        )
        self.assertIsNotNone(self.memory.identities[master_id]["gallery"]["baseline"])
        self.assertTrue(
            self.memory.assignment_metadata(10, camera_id="cam_1")["appearance_confirmed"]
        )

    def test_opposite_angles_are_inconclusive_and_remain_provisional(self):
        identity_id = self.create_pair()
        self.seed_angle(identity_id, "cam_1", "front", (1.0, 0.0))
        self.seed_angle(identity_id, "cam_2", "back", (1.0, 0.0))

        with self.memory._lock:
            promoted = self.memory._evaluate_provisional_evidence_locked(identity_id)

        self.assertFalse(promoted)
        self.assertEqual(self.memory.identity_state(identity_id), "provisional")
        self.assertEqual(self.memory.identities[identity_id]["reid_comparisons"], {})
        self.assertEqual(self.memory.lookup_track_key(self.left_key), identity_id)
        self.assertEqual(self.memory.lookup_track_key(self.right_key), identity_id)

    def test_high_same_angle_distance_challenges_but_does_not_split(self):
        identity_id = self.create_pair()
        self.seed_angle(identity_id, "cam_1", "left_side", (1.0, 0.0))
        self.seed_angle(identity_id, "cam_2", "left_side", (0.0, 1.0))

        with self.memory._lock:
            promoted = self.memory._evaluate_provisional_evidence_locked(identity_id)

        self.assertFalse(promoted)
        self.assertEqual(self.memory.identity_state(identity_id), "challenged")
        self.assertEqual(self.memory.lookup_track_key(self.left_key), identity_id)
        self.assertEqual(self.memory.lookup_track_key(self.right_key), identity_id)
        self.assertEqual(
            self.memory.assignment_metadata(10, camera_id="cam_1")["identity_state"],
            "challenged",
        )
        self.assertAlmostEqual(
            self.memory.identities[identity_id]["reid_comparisons"][
                "cam_1:cam_2:left_side"
            ],
            1.0,
        )

    def test_stable_location_allocates_master_only_after_both_global_checks(self):
        temporary_id = self.create_pair()
        record = self.memory.identities[temporary_id]
        record["camera_baselines"]["cam_1"] = make_slot((1.0, 0.0), "cam_1")

        state = self.memory.note_location_match(temporary_id, pair_streak=3, observations=())
        self.assertEqual(state, "provisional")

        record["camera_baselines"]["cam_2"] = make_slot((0.0, 1.0), "cam_2")
        record["global_reid_checked_track_keys"].update(
            (self.left_key, self.right_key)
        )
        state = self.memory.note_location_match(temporary_id, pair_streak=3, observations=())

        self.assertEqual(state, "confirmed")
        master_id = 1
        self.assertNotIn(temporary_id, self.memory.identities)
        self.assertEqual(self.memory.identity_state(master_id), "confirmed")
        self.assertEqual(record["confirmation_reason"], "stable_location")
        self.assertEqual(self.memory.lookup_track_key(self.left_key), master_id)
        self.assertEqual(self.memory.lookup_track_key(self.right_key), master_id)
        self.assertFalse(
            self.memory.assignment_metadata(10, camera_id="cam_1")["appearance_confirmed"]
        )

    def test_location_cannot_promote_before_global_reid_check(self):
        identity_id = self.create_pair()
        record = self.memory.identities[identity_id]
        record["camera_baselines"]["cam_1"] = make_slot((1.0, 0.0), "cam_1")
        record["camera_baselines"]["cam_2"] = make_slot((0.0, 1.0), "cam_2")

        state = self.memory.note_location_match(
            identity_id,
            pair_streak=3,
            observations=(),
        )

        self.assertEqual(state, "provisional")
        self.assertEqual(self.memory.identity_state(identity_id), "provisional")

    def test_provisional_global_reid_reuses_existing_master_for_whole_pair(self):
        self.memory.close(drain=False)
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            reid_extractor=FixedExtractor((1.0, 0.0)),
            intake_frames=1,
            intake_delay_seconds=0.0,
            blur_threshold=0.0,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=True,
        )
        existing_id = 1
        existing = self.memory._new_record(identity_state="confirmed")
        existing["gallery"]["baseline"] = make_slot((1.0, 0.0), "cam_1")
        self.memory.identities[existing_id] = existing
        self.memory.next_identity_id = 2

        provisional_id = self.create_pair()
        self.assertLess(provisional_id, 0)
        self.assertEqual(self.memory.next_identity_id, 2)

        checker = np.indices((64, 32)).sum(axis=0) % 2
        crop = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        self.memory.assign(
            10,
            crop,
            frame_index=1,
            excluded_identity_ids={existing_id},
            camera_id="cam_1",
            detection_confidence=0.95,
            orientation="front",
            observed_at=1.0,
            map_point=(0.0, 0.0),
            intake_body_complete=True,
        )
        self.assertTrue(self.memory.wait_for_idle())

        self.assertNotIn(provisional_id, self.memory.identities)
        self.assertEqual(self.memory.lookup_track_key(self.left_key), existing_id)
        self.assertEqual(self.memory.lookup_track_key(self.right_key), existing_id)
        self.assertEqual(self.memory.identity_state(existing_id), "confirmed")
        self.assertTrue(
            self.memory.assignment_metadata(10, camera_id="cam_1")[
                "appearance_confirmed"
            ]
        )

    def test_later_camera_track_attaches_provisionally_to_existing_master(self):
        identity_id = 7
        record = self.memory._new_record(identity_state="confirmed")
        record["gallery"]["baseline"] = make_slot((1.0, 0.0), "cam_1")
        self.memory.identities[identity_id] = record
        self.memory.next_identity_id = 8
        self.memory.track_to_identity[self.left_key] = identity_id
        self.memory.track_binding_metadata[self.left_key] = {
            "identity_state": "confirmed",
            "appearance_confirmed": True,
        }

        attached_id = self.memory.create_provisional_pair(self.left_key, self.right_key)

        self.assertEqual(attached_id, identity_id)
        self.assertEqual(self.memory.track_identity_state(self.left_key), "confirmed")
        self.assertEqual(self.memory.track_identity_state(self.right_key), "provisional")
        self.assertEqual(self.memory.lookup_track_key(self.right_key), identity_id)
        self.assertEqual(self.memory.next_identity_id, 8)

        record["camera_baselines"]["cam_2"] = make_slot((0.0, 1.0), "cam_2")
        self.memory.note_location_match(identity_id, pair_streak=3, observations=(
            {"camera_id": "cam_1", "local_track_id": 10, "point": (0.0, 0.0), "captured_at": 1.0},
            {"camera_id": "cam_2", "local_track_id": 20, "point": (0.0, 0.0), "captured_at": 1.0},
        ))

        self.assertEqual(self.memory.track_identity_state(self.right_key), "confirmed")
        self.assertEqual(self.memory.lookup_track_key(self.right_key), identity_id)

    def test_later_member_uses_fresh_same_angle_evidence_not_old_camera_slot(self):
        identity_id = 7
        record = self.memory._new_record(identity_state="confirmed")
        record["gallery"]["baseline"] = make_slot((1.0, 0.0), "cam_1")
        record["gallery"]["front"] = make_slot((1.0, 0.0), "cam_1")
        record["camera_views"]["cam_2"] = {
            slot_name: None for slot_name in REID_SEMANTIC_SLOTS
        }
        record["camera_views"]["cam_2"]["front"] = make_slot(
            (1.0, 0.0), "cam_2"
        )
        self.memory.identities[identity_id] = record
        self.memory.next_identity_id = 8
        self.memory.track_to_identity[self.left_key] = identity_id

        self.memory.create_provisional_pair(self.left_key, self.right_key)
        self.assertIsNone(record["camera_views"]["cam_2"]["front"])

        self.seed_angle(identity_id, "cam_2", "front", (0.99, 0.10))
        with self.memory._lock:
            confirmed = self.memory._evaluate_provisional_evidence_locked(identity_id)

        self.assertTrue(confirmed)
        self.assertEqual(self.memory.track_identity_state(self.right_key), "confirmed")
        self.assertEqual(self.memory.lookup_track_key(self.right_key), identity_id)

    def test_existing_master_history_cannot_skip_new_member_location_streak(self):
        self.memory.provisional_location_confirm_frames = 4
        identity_id = 7
        record = self.memory._new_record(identity_state="confirmed")
        record["gallery"]["baseline"] = make_slot((1.0, 0.0), "cam_1")
        record["location_match_frames"] = 99
        self.memory.identities[identity_id] = record
        self.memory.next_identity_id = 8
        self.memory.track_to_identity[self.left_key] = identity_id

        self.memory.create_provisional_pair(self.left_key, self.right_key)
        record["camera_baselines"]["cam_2"] = make_slot((0.0, 1.0), "cam_2")

        self.memory.note_location_match(identity_id, pair_streak=3, observations=(
            {"camera_id": "cam_1", "local_track_id": 10, "point": (0.0, 0.0), "captured_at": 1.0},
            {"camera_id": "cam_2", "local_track_id": 20, "point": (0.0, 0.0), "captured_at": 1.0},
        ))
        self.assertEqual(self.memory.track_identity_state(self.right_key), "provisional")

        self.memory.note_location_match(identity_id, pair_streak=4, observations=(
            {"camera_id": "cam_1", "local_track_id": 10, "point": (0.0, 0.0), "captured_at": 1.1},
            {"camera_id": "cam_2", "local_track_id": 20, "point": (0.0, 0.0), "captured_at": 1.1},
        ))
        self.assertEqual(self.memory.track_identity_state(self.right_key), "confirmed")

    def test_attached_track_intake_cannot_create_a_second_master(self):
        self.memory.close(drain=False)
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            reid_extractor=FixedExtractor((0.0, 1.0)),
            intake_frames=1,
            intake_delay_seconds=0.0,
            blur_threshold=0.0,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=True,
        )
        identity_id = 7
        record = self.memory._new_record(identity_state="confirmed")
        record["gallery"]["baseline"] = make_slot((1.0, 0.0), "cam_1")
        self.memory.identities[identity_id] = record
        self.memory.next_identity_id = 8
        self.memory.track_to_identity[self.left_key] = identity_id
        self.memory.create_provisional_pair(self.left_key, self.right_key)

        checker = np.indices((64, 32)).sum(axis=0) % 2
        crop = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        returned_id, _similarity, _reidentified = self.memory.assign(
            20,
            crop,
            frame_index=1,
            camera_id="cam_2",
            detection_confidence=0.95,
            orientation="back",
            observed_at=1.0,
            map_point=(0.0, 0.0),
            intake_body_complete=True,
        )
        self.assertTrue(self.memory.wait_for_idle())

        self.assertEqual(returned_id, identity_id)
        self.assertEqual(self.memory.lookup_track_key(self.right_key), identity_id)
        self.assertEqual(set(self.memory.identities), {identity_id})
        self.assertTrue(
            self.memory.assignment_metadata(20, camera_id="cam_2")[
                "provisional_intake_complete"
            ]
        )

    def test_revoked_provisional_binding_returns_to_intake_without_key_error(self):
        identity_id = self.create_pair()
        self.memory._physical_match_allowed_locked = lambda *_args, **_kwargs: False
        checker = np.indices((64, 32)).sum(axis=0) % 2
        crop = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)

        returned_id, _similarity, _reidentified = self.memory.assign(
            self.left_key[1],
            crop,
            frame_index=1,
            camera_id=self.left_key[0],
            detection_confidence=0.95,
            orientation="front",
            observed_at=1.0,
            map_point=(200.0, 0.0),
            intake_body_complete=True,
        )

        self.assertIsNone(returned_id)
        self.assertIsNone(self.memory.lookup_track_key(self.left_key))
        self.assertIn(identity_id, self.memory.identities)

    def test_attached_track_global_reid_confirms_the_existing_master(self):
        self.memory.close(drain=False)
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            reid_extractor=FixedExtractor((1.0, 0.0)),
            intake_frames=1,
            intake_delay_seconds=0.0,
            blur_threshold=0.0,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=True,
        )
        identity_id = 7
        record = self.memory._new_record(identity_state="confirmed")
        record["gallery"]["baseline"] = make_slot((1.0, 0.0), "cam_1")
        self.memory.identities[identity_id] = record
        self.memory.next_identity_id = 8
        self.memory.track_to_identity[self.left_key] = identity_id
        self.memory.create_provisional_pair(self.left_key, self.right_key)

        checker = np.indices((64, 32)).sum(axis=0) % 2
        crop = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        self.memory.assign(
            20,
            crop,
            frame_index=1,
            camera_id="cam_2",
            detection_confidence=0.95,
            orientation="front",
            observed_at=1.0,
            map_point=(0.0, 0.0),
            intake_body_complete=True,
        )
        self.assertTrue(self.memory.wait_for_idle())

        self.assertEqual(set(self.memory.identities), {identity_id})
        self.assertEqual(self.memory.lookup_track_key(self.right_key), identity_id)
        self.assertEqual(self.memory.track_identity_state(self.right_key), "confirmed")
        self.assertTrue(
            self.memory.assignment_metadata(20, camera_id="cam_2")[
                "appearance_confirmed"
            ]
        )
        self.assertIn("cam_2", self.memory.identities[identity_id]["camera_baselines"])
        self.assertIsNotNone(
            self.memory.identities[identity_id]["camera_views"]["cam_2"]["front"]
        )

    def test_borderline_location_match_needs_two_batches_before_commit(self):
        self.memory.close(drain=False)
        borderline_feature = (0.76, float(np.sqrt(1.0 - 0.76**2)))
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            reid_extractor=FixedExtractor(borderline_feature),
            distance_threshold=0.27,
            intake_frames=1,
            intake_delay_seconds=0.0,
            blur_threshold=0.0,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=True,
        )
        identity_id = 7
        record = self.memory._new_record(identity_state="confirmed")
        record["gallery"]["baseline"] = make_slot((1.0, 0.0), "cam_1")
        self.memory.identities[identity_id] = record
        self.memory.next_identity_id = 8
        self.memory.track_to_identity[self.left_key] = identity_id
        self.memory.create_provisional_pair(self.left_key, self.right_key)

        checker = np.indices((64, 32)).sum(axis=0) % 2
        crop = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        self.memory.assign(
            20,
            crop,
            frame_index=1,
            camera_id="cam_2",
            detection_confidence=0.95,
            orientation="front",
            observed_at=1.0,
            map_point=(0.0, 0.0),
            intake_body_complete=True,
        )
        self.assertTrue(self.memory.wait_for_idle())

        self.assertEqual(self.memory.track_identity_state(self.right_key), "provisional")
        self.assertNotIn("cam_2", record["camera_baselines"])
        self.assertIn(self.right_key, self.memory.pending_member_evidence)

        self.memory._process_provisional_semantic_task(
            {
                "type": "provisional_semantic",
                "identity_id": identity_id,
                "track_key": self.right_key,
                "slot_name": "back",
                "sample": {
                    "crop": crop,
                    "frame_index": 50,
                    "camera_id": "cam_2",
                    "observed_at": 1.5,
                    "sharpness": 200.0,
                    "area": float(crop.shape[0] * crop.shape[1]),
                    "detection_confidence": 0.95,
                    "orientation": "back",
                    "map_point": (0.0, 0.0),
                },
            }
        )
        self.assertIsNone(record["camera_views"]["cam_2"]["back"])
        self.assertIsNotNone(
            self.memory.pending_member_evidence[self.right_key]["views"]["back"]
        )

        self.memory.assign(
            20,
            crop,
            frame_index=100,
            camera_id="cam_2",
            detection_confidence=0.95,
            orientation="front",
            observed_at=2.0,
            map_point=(0.0, 0.0),
            intake_body_complete=True,
        )
        self.assertTrue(self.memory.wait_for_idle())

        self.assertEqual(self.memory.track_identity_state(self.right_key), "confirmed")
        self.assertIn("cam_2", record["camera_baselines"])
        self.assertIsNotNone(record["camera_views"]["cam_2"]["front"])
        self.assertIsNotNone(record["camera_views"]["cam_2"]["back"])
        self.assertNotIn(self.right_key, self.memory.pending_member_evidence)

    def test_borderline_global_match_does_not_create_or_bind_on_first_batch(self):
        self.memory.close(drain=False)
        borderline_feature = (0.76, float(np.sqrt(1.0 - 0.76**2)))
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            reid_extractor=FixedExtractor(borderline_feature),
            distance_threshold=0.27,
            intake_frames=1,
            intake_delay_seconds=0.0,
            blur_threshold=0.0,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=True,
        )
        existing_id = 7
        record = self.memory._new_record(identity_state="confirmed")
        record["gallery"]["baseline"] = make_slot((1.0, 0.0), "cam_1")
        self.memory.identities[existing_id] = record
        self.memory.next_identity_id = 8

        checker = np.indices((64, 32)).sum(axis=0) % 2
        crop = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        self.memory.assign(
            30,
            crop,
            frame_index=1,
            camera_id="cam_2",
            detection_confidence=0.95,
            observed_at=1.0,
            intake_body_complete=True,
        )
        self.assertTrue(self.memory.wait_for_idle())

        self.assertIsNone(self.memory.lookup_track_key(("cam_2", 30)))
        self.assertEqual(set(self.memory.identities), {existing_id})

        self.memory.assign(
            30,
            crop,
            frame_index=100,
            camera_id="cam_2",
            detection_confidence=0.95,
            observed_at=2.0,
            intake_body_complete=True,
        )
        self.assertTrue(self.memory.wait_for_idle())

        self.assertEqual(self.memory.lookup_track_key(("cam_2", 30)), existing_id)
        self.assertEqual(set(self.memory.identities), {existing_id})

    def test_delayed_intake_confirms_latest_location_assignment(self):
        identity_id, track_key = self.process_delayed_location_assigned_task(
            (1.0, 0.0)
        )

        self.assertEqual(set(self.memory.identities), {identity_id})
        self.assertEqual(self.memory.lookup_track_key(track_key), identity_id)
        self.assertEqual(self.memory.track_identity_state(track_key), "confirmed")
        self.assertTrue(
            self.memory.assignment_metadata(18, camera_id="cam_2")[
                "appearance_confirmed"
            ]
        )

    def test_delayed_intake_mismatch_challenges_without_creating_new_id(self):
        identity_id, track_key = self.process_delayed_location_assigned_task(
            (0.0, 1.0),
            task_provisional_identity_id=99,
        )

        self.assertEqual(set(self.memory.identities), {identity_id})
        self.assertEqual(self.memory.lookup_track_key(track_key), identity_id)
        self.assertEqual(self.memory.track_identity_state(track_key), "challenged")
        metadata = self.memory.assignment_metadata(18, camera_id="cam_2")
        self.assertFalse(metadata["appearance_confirmed"])
        self.assertAlmostEqual(metadata["distance"], 1.0)
        record = self.memory.identities[identity_id]
        self.assertNotIn("cam_2", record["camera_baselines"])
        self.assertNotIn("cam_2", record["camera_views"])
        self.assertNotIn(track_key, self.memory.pending_member_evidence)

    def test_location_managed_physical_gate_requires_three_bad_samples(self):
        identity_id = self.create_pair()
        self.memory.cross_camera_fusion_distance_cm = 50.0
        with self.memory._lock:
            self.memory.recent_master_observations[identity_id] = {
                "cam_1": {
                    "track_key": self.left_key,
                    "map_point": (0.0, 0.0),
                    "observed_at": 1.0,
                }
            }
            results = [
                self.memory._physical_match_allowed_locked(
                    identity_id,
                    "cam_2",
                    (100.0, 0.0),
                    1.0,
                )
                for _ in range(3)
            ]

        self.assertEqual(results, [True, True, False])


if __name__ == "__main__":
    unittest.main()
