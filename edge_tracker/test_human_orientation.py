import unittest
from types import SimpleNamespace

from pose_engine import get_human_orientation


def landmark(x=0.0, y=0.0, z=0.0, visibility=1.0):
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def pose(left_shoulder, right_shoulder, left_hip, right_hip):
    landmarks = [landmark() for _ in range(33)]
    landmarks[11] = left_shoulder
    landmarks[12] = right_shoulder
    landmarks[23] = left_hip
    landmarks[24] = right_hip
    return landmarks


class HumanOrientationTests(unittest.TestCase):
    def test_front_and_back_use_strict_parallel_gate(self):
        front = pose(
            landmark(0.8, 0.2), landmark(0.2, 0.2),
            landmark(0.7, 0.5), landmark(0.3, 0.5),
        )
        back = pose(
            landmark(0.2, 0.2), landmark(0.8, 0.2),
            landmark(0.3, 0.5), landmark(0.7, 0.5),
        )
        self.assertEqual(get_human_orientation(front), "front")
        self.assertEqual(get_human_orientation(back), "back")

    def test_side_views_require_narrow_shoulders_before_depth_is_trusted(self):
        left = pose(
            landmark(0.48, 0.2, -0.9), landmark(0.52, 0.2, 0.0),
            landmark(0.48, 0.5), landmark(0.52, 0.5),
        )
        right = pose(
            landmark(0.48, 0.2, 0.9), landmark(0.52, 0.2, 0.0),
            landmark(0.48, 0.5), landmark(0.52, 0.5),
        )
        self.assertEqual(get_human_orientation(left), "left_side")
        self.assertEqual(get_human_orientation(right), "right_side")

    def test_transition_and_low_visibility_are_rejected(self):
        transition = pose(
            landmark(0.35, 0.2, -0.9), landmark(0.65, 0.2, 0.0),
            landmark(0.4, 0.5), landmark(0.6, 0.5),
        )
        hidden = pose(
            landmark(0.8, 0.2, visibility=0.2), landmark(0.2, 0.2),
            landmark(0.7, 0.5), landmark(0.3, 0.5),
        )
        self.assertIsNone(get_human_orientation(transition))
        self.assertIsNone(get_human_orientation(hidden))

    def test_zero_depth_extreme_side_view_is_ambiguous(self):
        ambiguous = pose(
            landmark(0.501, 0.2, 0.0), landmark(0.499, 0.2, 0.0),
            landmark(0.50, 0.5), landmark(0.50, 0.5),
        )
        self.assertIsNone(get_human_orientation(ambiguous))


if __name__ == "__main__":
    unittest.main()
