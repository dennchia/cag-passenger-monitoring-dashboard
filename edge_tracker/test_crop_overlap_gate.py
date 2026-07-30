"""Regression tests for excluding multi-person ReID gallery crops."""

import unittest
from unittest.mock import patch

import numpy as np

from constants import DEFAULT_REID_CROP_MAX_INTRUDER_AREA_RATIO
from pose_engine import get_standing_points


class _ArrayValue:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Boxes:
    def __init__(self, boxes, track_ids):
        self.xyxy = _ArrayValue(np.asarray(boxes, dtype=float))
        self.id = _ArrayValue(np.asarray(track_ids, dtype=int))
        self.conf = _ArrayValue(np.full(len(boxes), 0.95, dtype=float))

    def __len__(self):
        return len(self.xyxy.value)


class _Keypoints:
    def __init__(self, boxes):
        xy = np.zeros((len(boxes), 17, 2), dtype=float)
        conf = np.ones((len(boxes), 17), dtype=float)
        for index, (x1, _y1, x2, y2) in enumerate(boxes):
            ankle = ((float(x1) + float(x2)) / 2.0, float(y2) - 2.0)
            xy[index, 15] = ankle
            xy[index, 16] = ankle
        self.xy = _ArrayValue(xy)
        self.conf = _ArrayValue(conf)


class _Result:
    def __init__(self, boxes, track_ids):
        self.boxes = _Boxes(boxes, track_ids)
        self.keypoints = _Keypoints(boxes)


class _AppearanceMemory:
    def __init__(self, suppressed_track_ids=()):
        self.suppressed_track_ids = set(suppressed_track_ids)
        self.assigned_crops = {}

    def observe_tracks(self, *_args, **_kwargs):
        return set()

    def is_track_suppressed(self, track_id, camera_id=None):
        del camera_id
        return int(track_id) in self.suppressed_track_ids

    def lookup(self, track_id, camera_id=None):
        del track_id, camera_id
        return None

    def temporary_group(self, track_id, camera_id=None):
        del track_id, camera_id
        return None

    def assign(self, track_id, crop, frame_index, **_kwargs):
        del frame_index
        self.assigned_crops[int(track_id)] = crop
        return None, 0.0, False

    def assignment_metadata(self, track_id, camera_id=None):
        del track_id, camera_id
        return {}

    def pending_count(self, track_id, camera_id=None):
        del track_id, camera_id
        return 0

    def required_intake_count(self):
        return 5


class CropOverlapGateTests(unittest.TestCase):
    TARGET_BOX = (20.0, 20.0, 80.0, 80.0)

    def _run(self, boxes, suppressed_track_ids=()):
        track_ids = list(range(1, len(boxes) + 1))
        memory = _AppearanceMemory(suppressed_track_ids=suppressed_track_ids)
        standing_points = get_standing_points(
            _Result(boxes, track_ids),
            np.zeros((200, 200, 3), dtype=np.uint8),
            appearance_memory=memory,
            camera_id="cam_1",
            frame_index=100,
            observation_time=10.0,
            use_mediapipe_feet=False,
            map_projector=lambda point: point,
            map_size_cm=200,
        )
        return memory, standing_points

    def test_clean_isolated_detection_still_produces_crop(self):
        memory, _points = self._run([self.TARGET_BOX])
        crop = memory.assigned_crops[1]
        self.assertIsNotNone(crop)
        self.assertGreater(crop.size, 0)

    def test_large_intruder_overlap_rejects_crop(self):
        intruder_box = (40.0, 30.0, 80.0, 70.0)
        with patch("pose_engine.identity_event") as event:
            memory, _points = self._run([self.TARGET_BOX, intruder_box])

        self.assertIsNone(memory.assigned_crops[1])
        rejection = next(
            call
            for call in event.call_args_list
            if call.args == ("reid_crop_rejected_overlap",)
            and call.kwargs.get("track_key") == ("cam_1", 1)
        )
        self.assertEqual(rejection.kwargs["intruder_track_id"], 2)
        self.assertGreater(
            rejection.kwargs["intruder_area_ratio"],
            DEFAULT_REID_CROP_MAX_INTRUDER_AREA_RATIO,
        )
        self.assertEqual(rejection.kwargs["throttle_seconds"], 1.0)

    def test_tiny_sliver_below_threshold_keeps_crop(self):
        sliver_box = (75.0, 20.0, 85.0, 30.0)
        memory, _points = self._run([self.TARGET_BOX, sliver_box])
        self.assertIsNotNone(memory.assigned_crops[1])

    def test_exact_threshold_keeps_crop(self):
        # The target's actual clamped crop is 63 x 68 pixels. This intruder's
        # intersection is 642.6 pixels, exactly 15 percent of that crop.
        threshold_box = (71.0, 18.0, 81.0, 82.26)
        memory, _points = self._run([self.TARGET_BOX, threshold_box])
        self.assertIsNotNone(memory.assigned_crops[1])

    def test_suppressed_shadow_does_not_block_canonical_crop(self):
        shadow_box = (21.0, 21.0, 79.0, 79.0)
        memory, _points = self._run(
            [self.TARGET_BOX, shadow_box],
            suppressed_track_ids={2},
        )
        self.assertIsNotNone(memory.assigned_crops[1])

    def test_real_intruder_still_blocks_when_shadow_is_present(self):
        shadow_box = (21.0, 21.0, 79.0, 79.0)
        real_intruder_box = (40.0, 30.0, 80.0, 70.0)
        with patch("pose_engine.identity_event") as event:
            memory, _points = self._run(
                [self.TARGET_BOX, shadow_box, real_intruder_box],
                suppressed_track_ids={2},
            )

        self.assertIsNone(memory.assigned_crops[1])
        rejection = next(
            call
            for call in event.call_args_list
            if call.args == ("reid_crop_rejected_overlap",)
            and call.kwargs.get("track_key") == ("cam_1", 1)
        )
        self.assertEqual(rejection.kwargs["intruder_track_id"], 3)

    def test_identical_duplicate_box_does_not_block_crop(self):
        memory, _points = self._run([self.TARGET_BOX, self.TARGET_BOX])
        self.assertIsNotNone(memory.assigned_crops[1])

    def test_rejected_crop_preserves_standing_point_and_map_presence(self):
        intruder_box = (40.0, 30.0, 80.0, 70.0)
        memory, points = self._run([self.TARGET_BOX, intruder_box])

        self.assertIsNone(memory.assigned_crops[1])
        self.assertEqual(points[0]["point"], (50, 78))
        self.assertTrue(points[0]["inside_tactical_map"])
        self.assertFalse(points[0]["suppressed"])


if __name__ == "__main__":
    unittest.main()
