import unittest

from main_tracker import fuse_camera_points


def observation(camera, track, identity, point, captured_at=10.0, confirmed=True):
    return {
        "camera_id": camera,
        "local_track_id": track,
        "identity_id": identity,
        "reid_confirmed": confirmed,
        "point": point,
        "captured_at": captured_at,
    }


class CrossCameraAssociationTests(unittest.TestCase):
    def test_non_finite_observations_are_discarded(self):
        fused = fuse_camera_points(
            {
                "cam_1": [observation("cam_1", 1, 1, (float("nan"), 0.0), 1.0)],
                "cam_2": [observation("cam_2", 2, 1, (1.0, 1.0), 1.0)],
            },
            max_distance_cm=50.0,
            require_reid=True,
        )
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["sources"], ["cam_2"])

    def test_same_confirmed_master_and_physics_merge(self):
        fused = fuse_camera_points(
            {
                "cam_1": [observation("cam_1", 1, 8, (100.0, 100.0))],
                "cam_2": [observation("cam_2", 7, 8, (106.0, 103.0), captured_at=10.1)],
            },
            max_distance_cm=50.0,
            max_skew_seconds=0.35,
            require_reid=True,
        )
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["identity_id"], 8)
        self.assertEqual(set(fused[0]["sources"]), {"cam_1", "cam_2"})

    def test_reid_mismatch_vetoes_identical_homography_position(self):
        fused = fuse_camera_points(
            {
                "cam_1": [observation("cam_1", 1, 8, (100.0, 100.0))],
                "cam_2": [observation("cam_2", 7, 9, (100.0, 100.0))],
            },
            max_distance_cm=50.0,
            require_reid=True,
        )
        self.assertEqual(len(fused), 2)

    def test_unknown_appearance_does_not_merge_when_reid_is_required(self):
        fused = fuse_camera_points(
            {
                "cam_1": [observation("cam_1", 1, None, (100.0, 100.0), confirmed=False)],
                "cam_2": [observation("cam_2", 7, None, (101.0, 100.0), confirmed=False)],
            },
            max_distance_cm=50.0,
            require_reid=True,
        )
        self.assertEqual(len(fused), 2)

    def test_matching_reid_cannot_override_impossible_space_or_time(self):
        too_far = fuse_camera_points(
            {
                "cam_1": [observation("cam_1", 1, 8, (0.0, 0.0))],
                "cam_2": [observation("cam_2", 7, 8, (200.0, 0.0))],
            },
            max_distance_cm=50.0,
            require_reid=True,
        )
        too_old = fuse_camera_points(
            {
                "cam_1": [observation("cam_1", 1, 8, (0.0, 0.0), captured_at=10.0)],
                "cam_2": [observation("cam_2", 7, 8, (1.0, 0.0), captured_at=11.0)],
            },
            max_distance_cm=50.0,
            max_skew_seconds=0.35,
            require_reid=True,
        )
        self.assertEqual(len(too_far), 2)
        self.assertEqual(len(too_old), 2)

    def test_two_by_two_assignment_is_one_to_one(self):
        fused = fuse_camera_points(
            {
                "cam_1": [
                    observation("cam_1", 1, None, (0.0, 0.0), confirmed=False),
                    observation("cam_1", 2, None, (10.0, 0.0), confirmed=False),
                ],
                "cam_2": [
                    observation("cam_2", 11, None, (9.0, 0.0), confirmed=False),
                    observation("cam_2", 12, None, (1.0, 0.0), confirmed=False),
                ],
            },
            max_distance_cm=20.0,
            require_reid=False,
        )
        self.assertEqual(len(fused), 2)
        source_track_pairs = {
            tuple(sorted((item["camera_id"], item["local_track_id"]) for item in person["observations"]))
            for person in fused
        }
        self.assertIn((("cam_1", 1), ("cam_2", 12)), source_track_pairs)
        self.assertIn((("cam_1", 2), ("cam_2", 11)), source_track_pairs)


if __name__ == "__main__":
    unittest.main()
