import cv2
import numpy as np
try:
    import mediapipe as mp
except ImportError:
    mp = None
try:
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision
except ImportError:
    BaseOptions = None
    vision = None

from constants import *
from core_math import (
    calculate_anatomical_anchor_pixels,
    calculate_anatomical_ratio,
    estimate_virtual_foot_from_ratio,
    get_anatomical_anchor_from_memory,
    get_anatomical_ratio_from_memory,
    recall_recent_foot_point,
    reject_impossible_foot_jump,
    remember_foot_point,
    store_anatomical_anchor,
    store_anatomical_ratio,
)
from reid_memory import crop_person


def visible_mediapipe_point(landmarks, index, crop_width, crop_height, offset_x, offset_y):
    landmark = landmarks[index]
    if landmark.visibility < MIN_MEDIAPIPE_VISIBILITY:
        return None
    return np.array([
        offset_x + landmark.x * crop_width,
        offset_y + landmark.y * crop_height,
    ], dtype=float)

def mediapipe_landmark_visibility(landmarks, index):
    if landmarks is None or index >= len(landmarks):
        return 0.0
    return float(getattr(landmarks[index], "visibility", 0.0))

def mediapipe_point_with_min_visibility(landmarks, index, crop_width, crop_height, offset_x, offset_y, min_visibility):
    if mediapipe_landmark_visibility(landmarks, index) < min_visibility:
        return None
    landmark = landmarks[index]
    return np.array([
        offset_x + landmark.x * crop_width,
        offset_y + landmark.y * crop_height,
    ], dtype=float)

def mean_available_points(points):
    visible_points = [point for point in points if point is not None]
    if not visible_points:
        return None
    return np.mean(np.array(visible_points), axis=0)

def estimate_head_pitch(landmarks, crop_width, crop_height, offset_x, offset_y):
    nose = visible_mediapipe_point(landmarks, MEDIAPIPE_NOSE, crop_width, crop_height, offset_x, offset_y)
    left_eye = visible_mediapipe_point(landmarks, MEDIAPIPE_LEFT_EYE, crop_width, crop_height, offset_x, offset_y)
    right_eye = visible_mediapipe_point(landmarks, MEDIAPIPE_RIGHT_EYE, crop_width, crop_height, offset_x, offset_y)
    left_shoulder = visible_mediapipe_point(landmarks, MEDIAPIPE_LEFT_SHOULDER, crop_width, crop_height, offset_x, offset_y)
    right_shoulder = visible_mediapipe_point(landmarks, MEDIAPIPE_RIGHT_SHOULDER, crop_width, crop_height, offset_x, offset_y)

    eye_center = mean_available_points([left_eye, right_eye])
    shoulder = mean_available_points([left_shoulder, right_shoulder])
    if nose is None or eye_center is None or shoulder is None:
        return "unknown"

    anchor_pixels = float(np.linalg.norm(shoulder - nose))
    if anchor_pixels < MIN_ANATOMICAL_ANCHOR_PIXELS:
        return "unknown"

    nose_below_eyes = float(nose[1] - eye_center[1])
    down_threshold = max(MIN_HEAD_DOWN_PIXELS, anchor_pixels * HEAD_DOWN_ANCHOR_FRACTION)
    if nose_below_eyes > down_threshold:
        return "looking_down"
    return "looking_straight"

def mediapipe_point_to_image(landmark, crop_width, crop_height, offset_x, offset_y):
    return (
        int(offset_x + landmark.x * crop_width),
        int(offset_y + landmark.y * crop_height),
    )

def draw_mediapipe_skeleton(annotated_frame, landmarks, crop_width, crop_height, offset_x, offset_y):
    if vision is None:
        return

    for connection in vision.PoseLandmarksConnections.POSE_LANDMARKS:
        start = landmarks[connection.start]
        end = landmarks[connection.end]
        if start.visibility < MIN_MEDIAPIPE_VISIBILITY or end.visibility < MIN_MEDIAPIPE_VISIBILITY:
            continue

        start_point = mediapipe_point_to_image(start, crop_width, crop_height, offset_x, offset_y)
        end_point = mediapipe_point_to_image(end, crop_width, crop_height, offset_x, offset_y)
        cv2.line(annotated_frame, start_point, end_point, (0, 220, 0), 2)

    for landmark in landmarks:
        if landmark.visibility < MIN_MEDIAPIPE_VISIBILITY:
            continue

        point = mediapipe_point_to_image(landmark, crop_width, crop_height, offset_x, offset_y)
        cv2.circle(annotated_frame, point, 3, (0, 255, 255), -1)

def extract_mediapipe_body_points(landmarks, crop_width, crop_height, offset_x, offset_y):
    nose = visible_mediapipe_point(landmarks, MEDIAPIPE_NOSE, crop_width, crop_height, offset_x, offset_y)
    left_shoulder = visible_mediapipe_point(landmarks, MEDIAPIPE_LEFT_SHOULDER, crop_width, crop_height, offset_x, offset_y)
    right_shoulder = visible_mediapipe_point(landmarks, MEDIAPIPE_RIGHT_SHOULDER, crop_width, crop_height, offset_x, offset_y)
    shoulder = mean_available_points([left_shoulder, right_shoulder])

    left_ankle = visible_mediapipe_point(landmarks, MEDIAPIPE_LEFT_ANKLE, crop_width, crop_height, offset_x, offset_y)
    right_ankle = visible_mediapipe_point(landmarks, MEDIAPIPE_RIGHT_ANKLE, crop_width, crop_height, offset_x, offset_y)
    left_heel = visible_mediapipe_point(landmarks, MEDIAPIPE_LEFT_HEEL, crop_width, crop_height, offset_x, offset_y)
    left_toe = visible_mediapipe_point(landmarks, MEDIAPIPE_LEFT_FOOT_INDEX, crop_width, crop_height, offset_x, offset_y)
    right_heel = visible_mediapipe_point(landmarks, MEDIAPIPE_RIGHT_HEEL, crop_width, crop_height, offset_x, offset_y)
    right_toe = visible_mediapipe_point(landmarks, MEDIAPIPE_RIGHT_FOOT_INDEX, crop_width, crop_height, offset_x, offset_y)

    foot_points = []
    strict_foot_points = []
    for ankle, heel, toe in ((left_ankle, left_heel, left_toe), (right_ankle, right_heel, right_toe)):
        if ankle is None:
            continue
        sole_point = mean_available_points([heel, toe])
        if sole_point is not None:
            foot_points.append(sole_point)
        else:
            foot_points.append(ankle)

    for ankle_index, heel_index, toe_index in (
        (MEDIAPIPE_LEFT_ANKLE, MEDIAPIPE_LEFT_HEEL, MEDIAPIPE_LEFT_FOOT_INDEX),
        (MEDIAPIPE_RIGHT_ANKLE, MEDIAPIPE_RIGHT_HEEL, MEDIAPIPE_RIGHT_FOOT_INDEX),
    ):
        ankle = mediapipe_point_with_min_visibility(
            landmarks, ankle_index, crop_width, crop_height, offset_x, offset_y, MIN_INITIAL_FOOT_VISIBILITY
        )
        heel = mediapipe_point_with_min_visibility(
            landmarks, heel_index, crop_width, crop_height, offset_x, offset_y, MIN_INITIAL_FOOT_VISIBILITY
        )
        toe = mediapipe_point_with_min_visibility(
            landmarks, toe_index, crop_width, crop_height, offset_x, offset_y, MIN_INITIAL_FOOT_VISIBILITY
        )
        if ankle is None:
            continue
        sole_point = mean_available_points([heel, toe])
        if sole_point is not None:
            strict_foot_points.append(sole_point)
        else:
            strict_foot_points.append(ankle)

    return (
        nose,
        shoulder,
        mean_available_points(foot_points),
        mean_available_points(strict_foot_points),
        left_shoulder,
        right_shoulder,
    )

def estimate_mediapipe_foot_point(
    frame,
    box,
    pose_estimator,
    anatomical_ratio_memory=None,
    anatomical_anchor_memory=None,
    last_foot_memory=None,
    track_id=None,
    identity_id=None,
    frame_index=None,
    pose_dropout_ttl_frames=DEFAULT_POSE_DROPOUT_TTL_FRAMES,
    ratio_ema_alpha=DEFAULT_ANATOMICAL_RATIO_EMA_ALPHA,
    max_foot_jump_pixels_per_frame=DEFAULT_MAX_FOOT_JUMP_PIXELS_PER_FRAME,
    annotated_frame=None,
    pose_debug=None,
):
    if pose_estimator is None:
        return None, "no_pose"

    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    box_width = x2 - x1
    box_height = y2 - y1
    padding_x = box_width * 0.12
    padding_y = box_height * 0.08

    crop_x1 = max(0, int(x1 - padding_x))
    crop_y1 = max(0, int(y1 - padding_y))
    crop_x2 = min(frame_width, int(x2 + padding_x))
    crop_y2 = min(frame_height, int(y2 + padding_y))

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        last_foot = recall_recent_foot_point(last_foot_memory, track_id, frame_index, pose_dropout_ttl_frames, identity_id=identity_id)
        if last_foot is not None:
            return last_foot, "last_seen"
        return None, "invalid_crop"

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    result = pose_estimator.detect(crop)
    if not result.pose_landmarks:
        last_foot = recall_recent_foot_point(last_foot_memory, track_id, frame_index, pose_dropout_ttl_frames, identity_id=identity_id)
        if last_foot is not None:
            return last_foot, "last_seen"
        return None, "no_visible_ankle"

    landmarks = result.pose_landmarks[0]
    crop_height, crop_width = crop.shape[:2]
    if annotated_frame is not None:
        draw_mediapipe_skeleton(annotated_frame, landmarks, crop_width, crop_height, crop_x1, crop_y1)

    nose, shoulder, foot_point, strict_foot_point, left_shoulder, right_shoulder = extract_mediapipe_body_points(
        landmarks,
        crop_width,
        crop_height,
        crop_x1,
        crop_y1,
    )
    head_pitch = estimate_head_pitch(landmarks, crop_width, crop_height, crop_x1, crop_y1)
    if pose_debug is not None:
        pose_debug["head_pitch"] = head_pitch
    head_allows_ratio_update = head_pitch == "looking_straight"

    stored_ratio = get_anatomical_ratio_from_memory(anatomical_ratio_memory, track_id, identity_id=identity_id)
    stored_anchor = get_anatomical_anchor_from_memory(anatomical_anchor_memory, track_id, identity_id=identity_id)
    if head_pitch == "looking_down" and stored_ratio is not None:
        virtual_foot = estimate_virtual_foot_from_ratio(
            nose,
            shoulder,
            stored_ratio,
            anchor_pixels_override=stored_anchor,
            left_shoulder=left_shoulder,
            right_shoulder=right_shoulder,
        )
        if virtual_foot is not None:
            held_foot = reject_impossible_foot_jump(
                last_foot_memory,
                track_id,
                virtual_foot,
                frame_index,
                max_foot_jump_pixels_per_frame,
                identity_id=identity_id,
            )
            if held_foot is not None:
                return held_foot, "physics_hold"
            remember_foot_point(last_foot_memory, track_id, virtual_foot, frame_index, identity_id=identity_id)
            return virtual_foot, "anatomical_ratio"

    if foot_point is not None:
        held_foot = reject_impossible_foot_jump(
            last_foot_memory,
            track_id,
            foot_point,
            frame_index,
            max_foot_jump_pixels_per_frame,
            identity_id=identity_id,
        )
        if held_foot is not None:
            return held_foot, "physics_hold"
        if strict_foot_point is not None and head_allows_ratio_update:
            ratio = calculate_anatomical_ratio(nose, shoulder, strict_foot_point)
            anchor_pixels = calculate_anatomical_anchor_pixels(nose, shoulder)
            if ratio is not None:
                store_anatomical_ratio(
                    anatomical_ratio_memory,
                    track_id,
                    ratio,
                    identity_id=identity_id,
                    ema_alpha=ratio_ema_alpha,
                )
            if anchor_pixels is not None:
                store_anatomical_anchor(
                    anatomical_anchor_memory,
                    track_id,
                    anchor_pixels,
                    identity_id=identity_id,
                    ema_alpha=ratio_ema_alpha,
                )
        remember_foot_point(last_foot_memory, track_id, foot_point, frame_index, identity_id=identity_id)
        return foot_point, "mediapipe"

    if stored_ratio is not None:
        virtual_foot = estimate_virtual_foot_from_ratio(
            nose,
            shoulder,
            stored_ratio,
            anchor_pixels_override=stored_anchor if head_pitch == "looking_down" else None,
            left_shoulder=left_shoulder,
            right_shoulder=right_shoulder,
        )
        if virtual_foot is not None:
            held_foot = reject_impossible_foot_jump(
                last_foot_memory,
                track_id,
                virtual_foot,
                frame_index,
                max_foot_jump_pixels_per_frame,
                identity_id=identity_id,
            )
            if held_foot is not None:
                return held_foot, "physics_hold"
            remember_foot_point(last_foot_memory, track_id, virtual_foot, frame_index, identity_id=identity_id)
            return virtual_foot, "anatomical_ratio"

    last_foot = recall_recent_foot_point(last_foot_memory, track_id, frame_index, pose_dropout_ttl_frames, identity_id=identity_id)
    if last_foot is not None:
        return last_foot, "last_seen"

    return None, "no_visible_ankle"

def estimate_yolo_pose_ankle_point(index, keypoint_xy, keypoint_conf):
    if keypoint_xy is None or index >= len(keypoint_xy):
        return None

    left_ankle = keypoint_xy[index][LEFT_ANKLE_KEYPOINT_INDEX]
    right_ankle = keypoint_xy[index][RIGHT_ANKLE_KEYPOINT_INDEX]
    ankles = []

    if keypoint_conf is None or keypoint_conf[index][LEFT_ANKLE_KEYPOINT_INDEX] >= MIN_ANKLE_CONFIDENCE:
        if not np.allclose(left_ankle, 0):
            ankles.append(left_ankle)

    if keypoint_conf is None or keypoint_conf[index][RIGHT_ANKLE_KEYPOINT_INDEX] >= MIN_ANKLE_CONFIDENCE:
        if not np.allclose(right_ankle, 0):
            ankles.append(right_ankle)

    if ankles:
        return np.mean(np.array(ankles), axis=0)

    return None

def estimate_box_bottom_point(box):
    x1, y1, x2, y2 = map(float, box)
    return np.array([(x1 + x2) / 2, y2], dtype=float)

def get_standing_points(
    result,
    frame,
    pose_estimator=None,
    anatomical_ratio_memory=None,
    anatomical_anchor_memory=None,
    last_foot_memory=None,
    frame_index=None,
    pose_dropout_ttl_frames=DEFAULT_POSE_DROPOUT_TTL_FRAMES,
    annotated_frame=None,
    appearance_memory=None,
):
    boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.empty((0, 4), dtype=float)
    track_ids = result.boxes.id.cpu().numpy().astype(int) if getattr(result.boxes, "id", None) is not None else None
    keypoints = getattr(result, "keypoints", None)
    keypoint_xy = None
    keypoint_conf = None

    if keypoints is not None and keypoints.xy is not None:
        keypoint_xy = keypoints.xy.cpu().numpy()
        if keypoints.conf is not None:
            keypoint_conf = keypoints.conf.cpu().numpy()

    standing_points = []
    for index, box in enumerate(boxes):
        track_id = int(track_ids[index]) if track_ids is not None and index < len(track_ids) else None
        identity_id = None
        reid_similarity = 0.0
        reidentified = False
        reid_intake_count = 0
        reid_intake_required = 0
        pose_debug = {}
        if appearance_memory is not None and track_id is not None and frame_index is not None:
            identity_id, reid_similarity, reidentified = appearance_memory.assign(track_id, crop_person(frame, box), frame_index)
            reid_intake_count = appearance_memory.pending_count(track_id)
            reid_intake_required = appearance_memory.required_intake_count()

        if pose_estimator is not None:
            point, method = estimate_mediapipe_foot_point(
                frame,
                box,
                pose_estimator,
                anatomical_ratio_memory=anatomical_ratio_memory,
                anatomical_anchor_memory=anatomical_anchor_memory,
                last_foot_memory=last_foot_memory,
                track_id=track_id,
                identity_id=identity_id,
                frame_index=frame_index,
                pose_dropout_ttl_frames=pose_dropout_ttl_frames,
                annotated_frame=annotated_frame,
                pose_debug=pose_debug,
            )
        else:
            point = estimate_yolo_pose_ankle_point(index, keypoint_xy, keypoint_conf)
            method = "yolo_pose"
            if point is None:
                point = None
                method = "no_visible_ankle"

        ratio = None
        if anatomical_ratio_memory is not None:
            ratio = get_anatomical_ratio_from_memory(anatomical_ratio_memory, track_id, identity_id=identity_id)

        if point is None:
            standing_points.append({
                "point": None,
                "track_id": track_id,
                "identity_id": identity_id,
                "reid_similarity": reid_similarity,
                "reidentified": reidentified,
                "reid_intake_count": reid_intake_count,
                "reid_intake_required": reid_intake_required,
                "method": method,
                "ratio": ratio,
                "head_pitch": pose_debug.get("head_pitch"),
            })
        else:
            standing_points.append({
                "point": (int(point[0]), int(point[1])),
                "track_id": track_id,
                "identity_id": identity_id,
                "reid_similarity": reid_similarity,
                "reidentified": reidentified,
                "reid_intake_count": reid_intake_count,
                "reid_intake_required": reid_intake_required,
                "method": method,
                "ratio": ratio,
                "head_pitch": pose_debug.get("head_pitch"),
            })

    return standing_points

class MediaPipePoseEstimator:
    def __init__(self, model_path):
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.35,
            min_pose_presence_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, bgr_image):
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        return self.landmarker.detect(mp_image)

    def close(self):
        self.landmarker.close()

def create_mediapipe_pose_estimator(enabled, model_path):
    if not enabled:
        return None
    if mp is None or BaseOptions is None or vision is None:
        print("MediaPipe is not installed. Falling back to YOLO/keypoint/box foot estimation.")
        return None
    if not model_path.exists():
        print(f"MediaPipe model file not found: {model_path}. Falling back to YOLO/keypoint/box foot estimation.")
        return None

    return MediaPipePoseEstimator(model_path)

