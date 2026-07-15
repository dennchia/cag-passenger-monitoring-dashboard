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


def get_human_orientation(landmarks):
    """Return one of the four semantic gallery views for clear body poses.

    The calibrated screen-space shoulder ratio first determines whether the
    subject can genuinely be side-on. MediaPipe Z depth is only trusted once
    that gate passes, which avoids classifying a wide front/back pose as a
    side view.
    """
    if not landmarks or len(landmarks) <= 24:
        return None

    left_shoulder = landmarks[MEDIAPIPE_LEFT_SHOULDER]
    right_shoulder = landmarks[MEDIAPIPE_RIGHT_SHOULDER]
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    if (
        left_shoulder.visibility < MIN_MEDIAPIPE_VISIBILITY
        or right_shoulder.visibility < MIN_MEDIAPIPE_VISIBILITY
        or left_hip.visibility < MIN_MEDIAPIPE_VISIBILITY
        or right_hip.visibility < MIN_MEDIAPIPE_VISIBILITY
    ):
        return None

    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    average_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
    average_hip_y = (left_hip.y + right_hip.y) / 2.0
    torso_height = max(0.01, abs(average_hip_y - average_shoulder_y))
    shoulder_ratio = shoulder_width / torso_height
    depth_difference = left_shoulder.z - right_shoulder.z

    if shoulder_ratio < 0.50:
        if depth_difference < -0.75:
            return "left_side"
        if depth_difference > 0.75:
            return "right_side"
        if shoulder_ratio < 0.15:
            if abs(depth_difference) < 0.05:
                return None
            return "left_side" if depth_difference < 0 else "right_side"

    if shoulder_ratio < 1.80:
        return None
    return "front" if left_shoulder.x > right_shoulder.x else "back"

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
    if pose_debug is not None:
        pose_debug["orientation"] = get_human_orientation(landmarks)
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
    camera_id=None,
    observation_time=None,
    use_mediapipe_feet=True,
    map_projector=None,
):
    boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.empty((0, 4), dtype=float)
    track_ids = result.boxes.id.cpu().numpy().astype(int) if getattr(result.boxes, "id", None) is not None else None
    keypoints = getattr(result, "keypoints", None)
    keypoint_xy = None
    keypoint_conf = None
    detection_confidences = None

    if keypoints is not None and keypoints.xy is not None:
        keypoint_xy = keypoints.xy.cpu().numpy()
        if keypoints.conf is not None:
            keypoint_conf = keypoints.conf.cpu().numpy()
    if getattr(result.boxes, "conf", None) is not None:
        detection_confidences = result.boxes.conf.cpu().numpy()

    # Crops are cheap to collect. TransReID itself is deliberately deferred
    # to AppearanceIdentityMemory until an unmapped local track has completed
    # its rapid intake burst. Mapped tracks never run appearance inference.
    # Observe every box together before processing individuals so the memory
    # can identify a newly spawned, near-identical ByteTrack shadow relative
    # to the already established same-camera track.
    person_crops = [crop_person(frame, box) for box in boxes]
    active_identity_ids = set()
    if appearance_memory is not None:
        active_identity_ids = appearance_memory.observe_tracks(
            () if track_ids is None else track_ids,
            boxes,
            frame_index=frame_index,
            camera_id=camera_id,
            observed_at=observation_time,
        )

    standing_points = []
    for index, box in enumerate(boxes):
        track_id = int(track_ids[index]) if track_ids is not None and index < len(track_ids) else None
        identity_id = None
        reid_similarity = 0.0
        reidentified = False
        assignment_metadata = {}
        reid_intake_count = 0
        reid_intake_required = 0
        pose_debug = {}
        detection_confidence = (
            float(detection_confidences[index])
            if detection_confidences is not None and index < len(detection_confidences)
            else None
        )
        if appearance_memory is not None and track_id is not None:
            identity_id = appearance_memory.lookup(track_id, camera_id=camera_id)

        map_point = None
        # Association physics must use the current raw detection, not a held
        # or smoothed foot point that could conceal a tracker-ID teleport.
        association_image_point = estimate_box_bottom_point(box)
        if association_image_point is not None and map_projector is not None:
            try:
                projected = map_projector((float(association_image_point[0]), float(association_image_point[1])))
                if projected is not None:
                    map_point = (float(projected[0]), float(projected[1]))
            except Exception:
                map_point = None

        track_suppressed = bool(
            appearance_memory is not None
            and track_id is not None
            and appearance_memory.is_track_suppressed(track_id, camera_id=camera_id)
        )
        if track_suppressed:
            # Never spend pose work or publish a tactical-map point for an
            # unresolved duplicate. ``assign`` is still called so the memory
            # can, after a short geometry-only probation, collect one bounded
            # five-crop appearance check. One-frame ghosts return before they
            # collect anything, and a verified duplicate never repeats it.
            if frame_index is not None:
                appearance_memory.assign(
                    track_id,
                    person_crops[index],
                    frame_index,
                    excluded_identity_ids=active_identity_ids,
                    camera_id=camera_id,
                    detection_confidence=detection_confidence,
                    observed_at=observation_time,
                    map_point=map_point,
                )
            reid_intake_count = appearance_memory.pending_count(track_id, camera_id=camera_id)
            standing_points.append({
                "point": None,
                "track_id": track_id,
                "identity_id": None,
                "reid_similarity": 0.0,
                "reidentified": False,
                "reid_confirmed": False,
                "query_feature_space_id": None,
                "matched_slot": None,
                "reid_intake_count": reid_intake_count,
                "reid_intake_required": appearance_memory.required_intake_count(),
                "method": "shadow_suppressed",
                "ratio": None,
                "head_pitch": None,
                "orientation": None,
                "suppressed": True,
            })
            continue

        if pose_estimator is not None and use_mediapipe_feet:
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

            # When MediaPipe is present only for semantic ReID, run it after
            # every cheap missing-slot/confidence/sharpness/cooldown gate has
            # passed. This avoids reintroducing per-person pose inference on
            # every frame merely to route gallery orientations.
            if (
                pose_estimator is not None
                and appearance_memory is not None
                and track_id is not None
                and frame_index is not None
                and appearance_memory.semantic_probe_due(
                    track_id,
                    person_crops[index],
                    frame_index,
                    detection_confidence,
                    camera_id=camera_id,
                )
            ):
                semantic_result = pose_estimator.detect(person_crops[index])
                if semantic_result.pose_landmarks:
                    pose_debug["orientation"] = get_human_orientation(semantic_result.pose_landmarks[0])

        if appearance_memory is not None and track_id is not None and frame_index is not None:
            identity_id, reid_similarity, reidentified = appearance_memory.assign(
                track_id,
                person_crops[index],
                frame_index,
                excluded_identity_ids=active_identity_ids,
                camera_id=camera_id,
                detection_confidence=detection_confidence,
                orientation=pose_debug.get("orientation"),
                observed_at=observation_time,
                map_point=map_point,
            )
            if identity_id is not None:
                # The reservation is scoped to this camera. A matching track
                # in another camera may legitimately share the master ID.
                active_identity_ids.add(identity_id)
            assignment_metadata = appearance_memory.assignment_metadata(
                track_id,
                camera_id=camera_id,
            )
            reid_intake_count = appearance_memory.pending_count(track_id, camera_id=camera_id)
            reid_intake_required = appearance_memory.required_intake_count()

        ratio = None
        if anatomical_ratio_memory is not None:
            ratio = get_anatomical_ratio_from_memory(anatomical_ratio_memory, track_id, identity_id=identity_id)

        identity_metadata = appearance_memory.identity_metadata(identity_id) if appearance_memory is not None and identity_id is not None else {}

        if point is None:
            standing_points.append({
                "point": None,
                "track_id": track_id,
                "identity_id": identity_id,
                "reid_similarity": reid_similarity,
                "reidentified": reidentified,
                "reid_confirmed": bool(assignment_metadata.get("appearance_confirmed", False)),
                "query_feature_space_id": assignment_metadata.get("query_feature_space_id"),
                "matched_slot": assignment_metadata.get("matched_slot"),
                "reid_intake_count": reid_intake_count,
                "reid_intake_required": reid_intake_required,
                "method": method,
                "ratio": ratio,
                "head_pitch": pose_debug.get("head_pitch"),
                "orientation": pose_debug.get("orientation"),
                "suppressed": False,
                **identity_metadata,
            })
        else:
            standing_points.append({
                "point": (int(point[0]), int(point[1])),
                "track_id": track_id,
                "identity_id": identity_id,
                "reid_similarity": reid_similarity,
                "reidentified": reidentified,
                "reid_confirmed": bool(assignment_metadata.get("appearance_confirmed", False)),
                "query_feature_space_id": assignment_metadata.get("query_feature_space_id"),
                "matched_slot": assignment_metadata.get("matched_slot"),
                "reid_intake_count": reid_intake_count,
                "reid_intake_required": reid_intake_required,
                "method": method,
                "ratio": ratio,
                "head_pitch": pose_debug.get("head_pitch"),
                "orientation": pose_debug.get("orientation"),
                "suppressed": False,
                **identity_metadata,
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

