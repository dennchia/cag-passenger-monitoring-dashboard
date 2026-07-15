import unittest

import numpy as np

from cctv_detect_humans_feet import (
    AppearanceIdentityMemory,
    calculate_anatomical_ratio,
    estimate_mediapipe_foot_point,
    estimate_virtual_foot_from_ratio,
    extrapolate_fourth_corner,
    remap_state_dict_for_timm,
    store_anatomical_ratio,
)


class FakeLandmark:
    def __init__(self, x=0.0, y=0.0, visibility=0.0):
        self.x = x
        self.y = y
        self.visibility = visibility


class FakePoseResult:
    def __init__(self, landmarks):
        self.pose_landmarks = [landmarks]


class FakeEmptyPoseResult:
    pose_landmarks = []


class FakePoseEstimator:
    def __init__(self, results):
        self.results = list(results)

    def detect(self, _crop):
        return self.results.pop(0)


def make_landmarks(crop_width, crop_height, points):
    landmarks = [FakeLandmark() for _ in range(33)]
    for index, point in points.items():
        if len(point) == 3:
            x, y, visibility = point
        else:
            x, y = point
            visibility = 0.95
        landmarks[index] = FakeLandmark(
            x=x / crop_width,
            y=y / crop_height,
            visibility=visibility,
        )
    return landmarks


class AnatomicalRatioTrackingTest(unittest.TestCase):
    def test_ratio_can_reconstruct_scaled_virtual_foot(self):
        initial_nose = np.array([100.0, 50.0])
        initial_shoulder = np.array([100.0, 70.0])
        initial_foot = np.array([100.0, 200.0])

        ratio = calculate_anatomical_ratio(initial_nose, initial_shoulder, initial_foot)

        current_nose = np.array([250.0, 80.0])
        current_shoulder = np.array([250.0, 90.0])
        virtual_foot = estimate_virtual_foot_from_ratio(current_nose, current_shoulder, ratio)

        self.assertAlmostEqual(ratio, 20.0 / 130.0)
        self.assertTrue(np.allclose(virtual_foot, np.array([250.0, 155.0])))

    def test_virtual_foot_projects_perpendicular_to_shoulder_line(self):
        nose = np.array([100.0, 50.0])
        shoulder = np.array([100.0, 70.0])
        left_shoulder = np.array([90.0, 75.0])
        right_shoulder = np.array([110.0, 65.0])
        ratio = 20.0 / 130.0

        virtual_foot = estimate_virtual_foot_from_ratio(
            nose,
            shoulder,
            ratio,
            left_shoulder=left_shoulder,
            right_shoulder=right_shoulder,
        )

        shoulder_vector = right_shoulder - left_shoulder
        projected_vector = virtual_foot - shoulder
        self.assertGreater(virtual_foot[1], shoulder[1])
        self.assertNotAlmostEqual(virtual_foot[0], shoulder[0])
        self.assertAlmostEqual(float(np.dot(shoulder_vector, projected_vector)), 0.0)

    def test_invalid_short_anchor_does_not_update_ratio(self):
        nose = np.array([100.0, 50.0])
        shoulder = np.array([100.0, 52.0])
        foot = np.array([100.0, 200.0])

        self.assertIsNone(calculate_anatomical_ratio(nose, shoulder, foot))

    def test_ratio_above_strict_upper_bound_is_rejected(self):
        nose = np.array([100.0, 50.0])
        shoulder = np.array([100.0, 70.0])
        foot = np.array([100.0, 150.0])

        self.assertIsNone(calculate_anatomical_ratio(nose, shoulder, foot))

    def test_ratio_storage_uses_ema_instead_of_overwrite(self):
        memory = {}
        store_anatomical_ratio(memory, 1, 0.10)
        store_anatomical_ratio(memory, 1, 0.20)

        self.assertAlmostEqual(memory[1], 0.11)

    def test_mediapipe_ratio_update_uses_slow_ema_when_head_straight_and_feet_clear(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        memory = {5: 0.10}
        last_foot_memory = {}
        box = np.array([0.0, 0.0, 100.0, 220.0])
        landmarks = make_landmarks(
            crop_width=112,
            crop_height=237,
            points={
                0: (50.0, 50.0),
                2: (45.0, 47.0),
                5: (55.0, 47.0),
                11: (50.0, 70.0),
                12: (50.0, 70.0),
                27: (50.0, 190.0),
                29: (50.0, 200.0),
            },
        )
        estimator = FakePoseEstimator([FakePoseResult(landmarks)])

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=5,
            frame_index=1,
        )

        self.assertEqual(method, "mediapipe")
        self.assertTrue(np.allclose(foot, np.array([50.0, 200.0])))
        self.assertAlmostEqual(memory[5], 0.10 * 0.98 + (20.0 / 130.0) * 0.02)

    def test_low_confidence_feet_do_not_update_ratio_memory(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        memory = {6: 0.10}
        last_foot_memory = {}
        box = np.array([0.0, 0.0, 100.0, 220.0])
        landmarks = make_landmarks(
            crop_width=112,
            crop_height=237,
            points={
                0: (50.0, 50.0),
                2: (45.0, 47.0),
                5: (55.0, 47.0),
                11: (50.0, 70.0),
                12: (50.0, 70.0),
                27: (50.0, 190.0, 0.60),
                29: (50.0, 200.0, 0.60),
            },
        )
        estimator = FakePoseEstimator([FakePoseResult(landmarks)])

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=6,
            frame_index=1,
        )

        self.assertEqual(method, "mediapipe")
        self.assertTrue(np.allclose(foot, np.array([50.0, 200.0])))
        self.assertAlmostEqual(memory[6], 0.10)

    def test_transreid_state_dict_keys_are_remapped_for_timm(self):
        state_dict = {
            "base.cls_token": np.zeros((1, 1, 1), dtype=np.float32),
            "base.patch_embed.proj.weight": np.zeros((1, 1, 1, 1), dtype=np.float32),
            "classifier.weight": np.zeros((1, 1), dtype=np.float32),
        }

        remapped = remap_state_dict_for_timm(state_dict)

        self.assertIn("cls_token", remapped)
        self.assertIn("patch_embed.proj.weight", remapped)
        self.assertIn("classifier.weight", remapped)
        self.assertNotIn("base.cls_token", remapped)

    def test_mediapipe_clear_view_stores_ratio_then_occlusion_uses_track_memory(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        memory = {}
        last_foot_memory = {}
        box = np.array([0.0, 0.0, 100.0, 200.0])

        clear_view_landmarks = make_landmarks(
            crop_width=112,
            crop_height=216,
            points={
                0: (50.0, 50.0),
                2: (45.0, 47.0),
                5: (55.0, 47.0),
                11: (50.0, 70.0),
                12: (50.0, 70.0),
                27: (50.0, 190.0),
                29: (50.0, 200.0),
            },
        )
        occluded_landmarks = make_landmarks(
            crop_width=112,
            crop_height=216,
            points={
                0: (50.0, 80.0),
                11: (50.0, 90.0),
                12: (50.0, 90.0),
            },
        )
        estimator = FakePoseEstimator([
            FakePoseResult(clear_view_landmarks),
            FakePoseResult(occluded_landmarks),
        ])

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=42,
            frame_index=1,
        )
        self.assertEqual(method, "mediapipe")
        self.assertTrue(np.allclose(foot, np.array([50.0, 200.0])))
        self.assertAlmostEqual(memory[42], 20.0 / 130.0)
        self.assertTrue(np.allclose(last_foot_memory[42]["point"], np.array([50.0, 200.0])))

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=42,
            frame_index=2,
        )
        self.assertEqual(method, "anatomical_ratio")
        self.assertTrue(np.allclose(foot, np.array([50.0, 155.0])))

    def test_pose_dropout_reuses_recent_foot_for_same_track_id(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        memory = {}
        last_foot_memory = {}
        box = np.array([0.0, 0.0, 100.0, 200.0])

        clear_view_landmarks = make_landmarks(
            crop_width=112,
            crop_height=216,
            points={
                0: (50.0, 50.0),
                11: (50.0, 70.0),
                12: (50.0, 70.0),
                27: (50.0, 190.0),
                29: (50.0, 200.0),
            },
        )
        estimator = FakePoseEstimator([
            FakePoseResult(clear_view_landmarks),
            FakeEmptyPoseResult(),
        ])

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=7,
            frame_index=10,
            pose_dropout_ttl_frames=30,
        )
        self.assertEqual(method, "mediapipe")
        self.assertTrue(np.allclose(foot, np.array([50.0, 200.0])))

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=7,
            frame_index=20,
            pose_dropout_ttl_frames=30,
        )
        self.assertEqual(method, "last_seen")
        self.assertTrue(np.allclose(foot, np.array([50.0, 200.0])))

    def test_missing_ankle_does_not_count_as_real_foot(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        memory = {9: 20.0 / 130.0}
        last_foot_memory = {}
        box = np.array([0.0, 0.0, 100.0, 200.0])
        landmarks = make_landmarks(
            crop_width=112,
            crop_height=216,
            points={
                0: (50.0, 80.0),
                11: (50.0, 90.0),
                12: (50.0, 90.0),
                29: (50.0, 190.0),
                31: (50.0, 200.0),
            },
        )
        estimator = FakePoseEstimator([FakePoseResult(landmarks)])

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=9,
            frame_index=1,
        )

        self.assertEqual(method, "anatomical_ratio")
        self.assertTrue(np.allclose(foot, np.array([50.0, 155.0])))

    def test_missing_ankle_does_not_fallback_to_box_bottom(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        memory = {}
        last_foot_memory = {}
        box = np.array([0.0, 0.0, 100.0, 200.0])
        landmarks = make_landmarks(
            crop_width=112,
            crop_height=216,
            points={
                0: (50.0, 80.0),
                11: (50.0, 90.0),
                12: (50.0, 90.0),
            },
        )
        estimator = FakePoseEstimator([FakePoseResult(landmarks)])

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=99,
            frame_index=1,
        )

        self.assertIsNone(foot)
        self.assertEqual(method, "no_visible_ankle")
        self.assertNotIn(99, memory)

    def test_looking_down_uses_virtual_foot_and_freezes_ratio_memory(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        memory = {8: 20.0 / 130.0}
        last_foot_memory = {}
        box = np.array([0.0, 0.0, 100.0, 220.0])
        landmarks = make_landmarks(
            crop_width=112,
            crop_height=237,
            points={
                0: (50.0, 80.0),
                2: (45.0, 50.0),
                5: (55.0, 50.0),
                11: (50.0, 100.0),
                12: (50.0, 100.0),
                27: (50.0, 190.0),
                29: (50.0, 210.0),
            },
        )
        estimator = FakePoseEstimator([FakePoseResult(landmarks)])

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=8,
            frame_index=1,
        )

        self.assertEqual(method, "anatomical_ratio")
        self.assertTrue(np.allclose(foot, np.array([50.0, 230.0])))
        self.assertAlmostEqual(memory[8], 20.0 / 130.0)

    def test_looking_down_uses_saved_head_up_anchor_instead_of_shrunken_anchor(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        ratio_memory = {12: 20.0 / 130.0}
        anchor_memory = {12: 20.0}
        last_foot_memory = {}
        box = np.array([0.0, 0.0, 100.0, 220.0])
        landmarks = make_landmarks(
            crop_width=112,
            crop_height=237,
            points={
                0: (50.0, 94.0),
                2: (45.0, 70.0),
                5: (55.0, 70.0),
                11: (50.0, 100.0),
                12: (50.0, 100.0),
                27: (50.0, 190.0),
                29: (50.0, 210.0),
            },
        )
        estimator = FakePoseEstimator([FakePoseResult(landmarks)])

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=ratio_memory,
            anatomical_anchor_memory=anchor_memory,
            last_foot_memory=last_foot_memory,
            track_id=12,
            frame_index=1,
        )

        self.assertEqual(method, "anatomical_ratio")
        self.assertTrue(np.allclose(foot, np.array([50.0, 230.0])))
        self.assertAlmostEqual(anchor_memory[12], 20.0)

    def test_impossible_jump_holds_previous_foot_point(self):
        frame = np.zeros((500, 500, 3), dtype=np.uint8)
        memory = {}
        last_foot_memory = {10: {"point": np.array([50.0, 200.0]), "frame_index": 1}}
        box = np.array([280.0, 0.0, 380.0, 240.0])
        landmarks = make_landmarks(
            crop_width=124,
            crop_height=259,
            points={
                0: (50.0, 50.0),
                2: (45.0, 47.0),
                5: (55.0, 47.0),
                11: (50.0, 70.0),
                12: (50.0, 70.0),
                27: (50.0, 190.0),
                29: (50.0, 200.0),
            },
        )
        estimator = FakePoseEstimator([FakePoseResult(landmarks)])

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=10,
            frame_index=2,
            max_foot_jump_pixels_per_frame=80.0,
        )

        self.assertEqual(method, "physics_hold")
        self.assertTrue(np.allclose(foot, np.array([50.0, 200.0])))
        self.assertTrue(np.allclose(last_foot_memory[10]["point"], np.array([50.0, 200.0])))

    def test_ratio_is_stored_by_identity_when_track_changes(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        memory = {}
        last_foot_memory = {}
        box = np.array([0.0, 0.0, 100.0, 200.0])

        clear_view_landmarks = make_landmarks(
            crop_width=112,
            crop_height=216,
            points={
                0: (50.0, 50.0),
                2: (45.0, 47.0),
                5: (55.0, 47.0),
                11: (50.0, 70.0),
                12: (50.0, 70.0),
                27: (50.0, 190.0),
                29: (50.0, 200.0),
            },
        )
        occluded_landmarks = make_landmarks(
            crop_width=112,
            crop_height=216,
            points={
                0: (50.0, 80.0),
                11: (50.0, 90.0),
                12: (50.0, 90.0),
            },
        )
        estimator = FakePoseEstimator([
            FakePoseResult(clear_view_landmarks),
            FakePoseResult(occluded_landmarks),
        ])
        appearance_memory = AppearanceIdentityMemory(similarity_threshold=0.7, ttl_frames=100)
        crop_a = np.full((120, 60, 3), (20, 80, 200), dtype=np.uint8)
        crop_b = np.full((120, 60, 3), (20, 82, 198), dtype=np.uint8)

        identity_a, _, _ = appearance_memory.assign(1, crop_a, 1)
        identity_b, _, _ = appearance_memory.assign(99, crop_b, 5)
        self.assertEqual(identity_a, identity_b)

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=1,
            identity_id=identity_a,
            frame_index=1,
        )
        self.assertEqual(method, "mediapipe")
        self.assertTrue(np.allclose(foot, np.array([50.0, 200.0])))
        self.assertAlmostEqual(memory[identity_a], 20.0 / 130.0)

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=99,
            identity_id=identity_b,
            frame_index=2,
        )
        self.assertEqual(method, "anatomical_ratio")
        self.assertTrue(np.allclose(foot, np.array([50.0, 155.0])))


    def test_appearance_memory_reuses_stable_id_for_similar_crop_after_track_change(self):
        memory = AppearanceIdentityMemory(similarity_threshold=0.7, ttl_frames=100)
        crop_a = np.full((120, 60, 3), (20, 80, 200), dtype=np.uint8)
        crop_b = np.full((120, 60, 3), (20, 82, 198), dtype=np.uint8)

        identity_a, _, reidentified_a = memory.assign(1, crop_a, 1)
        identity_b, similarity, reidentified_b = memory.assign(99, crop_b, 20)

        self.assertEqual(identity_a, identity_b)
        self.assertFalse(reidentified_a)
        self.assertTrue(reidentified_b)
        self.assertGreaterEqual(similarity, 0.7)

    def test_extrapolate_fourth_corner_uses_visible_edge_lines(self):
        known_corners = {
            "top_left": (10.0, 10.0),
            "top_right": (110.0, 10.0),
            "bottom_right": (120.0, 90.0),
        }
        edge_points = {
            "edge_a": (60.0, 90.0),
            "edge_b": (8.0, 50.0),
        }

        corner = extrapolate_fourth_corner(known_corners, edge_points, missing_corner="bottom_left")

        self.assertTrue(np.allclose(corner, np.array([6.0, 90.0])))

    def test_extrapolate_fourth_corner_accepts_edge_points_in_either_order(self):
        known_corners = {
            "top_left": (10.0, 10.0),
            "top_right": (110.0, 10.0),
            "bottom_right": (120.0, 90.0),
        }
        edge_points = {
            "edge_a": (8.0, 50.0),
            "edge_b": (60.0, 90.0),
        }

        corner = extrapolate_fourth_corner(known_corners, edge_points, missing_corner="bottom_left")

        self.assertTrue(np.allclose(corner, np.array([6.0, 90.0])))

    def test_extrapolate_fourth_corner_uses_explicit_clockwise_edges(self):
        known_corners = {
            "top_left": (10.0, 10.0),
            "top_right": (110.0, 10.0),
            "bottom_left": (10.0, 90.0),
        }
        edge_points = {
            "edge_from_prev": (112.5, 55.0),
            "edge_from_next": (62.5, 95.0),
        }

        corner = extrapolate_fourth_corner(
            known_corners,
            edge_points,
            missing_corner="bottom_right",
            allow_swapped_edges=False,
        )

        self.assertTrue(np.allclose(corner, np.array([115.0, 100.0])))

    def test_extrapolate_fourth_corner_falls_back_to_parallelogram_math(self):
        known_corners = {
            "top_left": (10.0, 10.0),
            "top_right": (110.0, 10.0),
            "bottom_right": (110.0, 90.0),
        }

        corner = extrapolate_fourth_corner(known_corners, missing_corner="bottom_left")

        self.assertTrue(np.allclose(corner, np.array([10.0, 90.0])))

if __name__ == "__main__":
    unittest.main()

