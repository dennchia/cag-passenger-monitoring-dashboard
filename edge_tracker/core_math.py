import cv2
import numpy as np

from constants import (
    DEFAULT_MAP_POSITION_EMA_ALPHA,
    DEFAULT_MAX_PERSON_SPEED_MPS,
    MAX_ANATOMICAL_RATIO,
    MIN_ANATOMICAL_ANCHOR_PIXELS,
    MIN_ANATOMICAL_FULL_BODY_PIXELS,
    MIN_ANATOMICAL_RATIO,
)


def camera_point_to_map(point, homography):
    source = np.array([[[point[0], point[1]]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(source, homography)[0][0]
    return float(mapped[0]), float(mapped[1])


def distance_cm(point_a, point_b):
    return float(np.linalg.norm(np.array(point_a, dtype=float) - np.array(point_b, dtype=float)))


def calculate_anatomical_ratio(nose_point, shoulder_point, foot_point):
    if nose_point is None or shoulder_point is None or foot_point is None:
        return None

    anchor_pixels = float(np.linalg.norm(shoulder_point - nose_point))
    full_body_pixels = float(abs(foot_point[1] - shoulder_point[1]))

    if anchor_pixels < MIN_ANATOMICAL_ANCHOR_PIXELS or full_body_pixels < MIN_ANATOMICAL_FULL_BODY_PIXELS:
        return None

    ratio = anchor_pixels / full_body_pixels
    if not MIN_ANATOMICAL_RATIO <= ratio <= MAX_ANATOMICAL_RATIO:
        return None

    return ratio


def estimate_virtual_foot_from_ratio(
    nose_point,
    shoulder_point,
    anatomical_ratio,
    anchor_pixels_override=None,
    left_shoulder=None,
    right_shoulder=None,
):
    if nose_point is None or shoulder_point is None or anatomical_ratio is None or anatomical_ratio <= 0:
        return None

    if anchor_pixels_override is None:
        anchor_pixels = float(np.linalg.norm(shoulder_point - nose_point))
    else:
        anchor_pixels = float(anchor_pixels_override)
    if anchor_pixels < MIN_ANATOMICAL_ANCHOR_PIXELS:
        return None

    full_body_pixels = anchor_pixels / anatomical_ratio
    if full_body_pixels < MIN_ANATOMICAL_FULL_BODY_PIXELS:
        return None

    if left_shoulder is None or right_shoulder is None:
        return np.array([shoulder_point[0], shoulder_point[1] + full_body_pixels], dtype=float)

    shoulder_vector = np.array(right_shoulder, dtype=float) - np.array(left_shoulder, dtype=float)
    shoulder_width = float(np.linalg.norm(shoulder_vector))
    if shoulder_width < MIN_ANATOMICAL_ANCHOR_PIXELS:
        return np.array([shoulder_point[0], shoulder_point[1] + full_body_pixels], dtype=float)

    body_direction = np.array([-shoulder_vector[1], shoulder_vector[0]], dtype=float)
    body_direction /= float(np.linalg.norm(body_direction))
    if body_direction[1] < 0:
        body_direction = -body_direction

    return np.array(shoulder_point, dtype=float) + body_direction * full_body_pixels


def calculate_anatomical_anchor_pixels(nose_point, shoulder_point):
    if nose_point is None or shoulder_point is None:
        return None
    anchor_pixels = float(np.linalg.norm(shoulder_point - nose_point))
    if anchor_pixels < MIN_ANATOMICAL_ANCHOR_PIXELS:
        return None
    return anchor_pixels


def resolve_memory_key(track_id, identity_id=None):
    if identity_id is not None:
        return ("identity", int(identity_id))
    if track_id is None:
        return None
    return ("track", int(track_id))


def remember_foot_point(last_foot_memory, track_id, foot_point, frame_index, identity_id=None):
    memory_key = resolve_memory_key(track_id, identity_id)
    if last_foot_memory is None or memory_key is None or foot_point is None or frame_index is None:
        return

    last_foot_memory[memory_key] = {
        "point": np.array(foot_point, dtype=float),
        "frame_index": int(frame_index),
    }


def recall_recent_foot_point(last_foot_memory, track_id, frame_index, max_age_frames, identity_id=None):
    memory_key = resolve_memory_key(track_id, identity_id)
    if last_foot_memory is None or memory_key is None or frame_index is None:
        return None

    cached = last_foot_memory.get(memory_key)
    if cached is None:
        return None

    age = int(frame_index) - int(cached["frame_index"])
    if age < 0 or age > max_age_frames:
        return None

    return np.array(cached["point"], dtype=float)


def reject_impossible_foot_jump(last_foot_memory, track_id, foot_point, frame_index, max_jump_pixels_per_frame, identity_id=None):
    memory_key = resolve_memory_key(track_id, identity_id)
    if (
        last_foot_memory is None
        or memory_key is None
        or foot_point is None
        or frame_index is None
        or max_jump_pixels_per_frame is None
        or max_jump_pixels_per_frame <= 0
    ):
        return None

    cached = last_foot_memory.get(memory_key)
    if cached is None:
        return None

    frame_delta = int(frame_index) - int(cached["frame_index"])
    if frame_delta <= 0:
        return None

    max_jump = float(max_jump_pixels_per_frame) * frame_delta
    jump = float(np.linalg.norm(np.array(foot_point, dtype=float) - np.array(cached["point"], dtype=float)))
    if jump <= max_jump:
        return None

    return np.array(cached["point"], dtype=float)


def get_anatomical_ratio_from_memory(anatomical_ratio_memory, track_id, identity_id=None):
    if anatomical_ratio_memory is None:
        return None

    memory_key = resolve_memory_key(track_id, identity_id)
    if memory_key is not None and memory_key in anatomical_ratio_memory:
        return anatomical_ratio_memory[memory_key]

    if track_id is not None and track_id in anatomical_ratio_memory:
        return anatomical_ratio_memory[track_id]

    return None


def get_anatomical_anchor_from_memory(anatomical_anchor_memory, track_id, identity_id=None):
    if anatomical_anchor_memory is None:
        return None

    memory_key = resolve_memory_key(track_id, identity_id)
    if memory_key is not None and memory_key in anatomical_anchor_memory:
        return anatomical_anchor_memory[memory_key]

    if track_id is not None and track_id in anatomical_anchor_memory:
        return anatomical_anchor_memory[track_id]

    return None


def store_anatomical_ratio(anatomical_ratio_memory, track_id, ratio, identity_id=None, ema_alpha=0.1):
    if anatomical_ratio_memory is None or ratio is None:
        return

    memory_key = resolve_memory_key(track_id, identity_id)
    if memory_key is None:
        return

    previous_ratio = anatomical_ratio_memory.get(memory_key)
    if previous_ratio is None:
        anatomical_ratio_memory[memory_key] = float(ratio)
        return

    updated_ratio = (1.0 - ema_alpha) * float(previous_ratio) + ema_alpha * float(ratio)
    anatomical_ratio_memory[memory_key] = updated_ratio


def store_anatomical_anchor(anatomical_anchor_memory, track_id, anchor_pixels, identity_id=None, ema_alpha=0.02):
    if anatomical_anchor_memory is None or anchor_pixels is None:
        return

    memory_key = resolve_memory_key(track_id, identity_id)
    if memory_key is None:
        return

    previous_anchor = anatomical_anchor_memory.get(memory_key)
    if previous_anchor is None:
        anatomical_anchor_memory[memory_key] = float(anchor_pixels)
        return

    updated_anchor = (1.0 - ema_alpha) * float(previous_anchor) + ema_alpha * float(anchor_pixels)
    anatomical_anchor_memory[memory_key] = updated_anchor


def _as_point(point):
    array = np.array(point, dtype=float)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"Expected a finite 2D point, got {point!r}")
    return array


def _line_from_points(point_a, point_b):
    a = _as_point(point_a)
    b = _as_point(point_b)
    if float(np.linalg.norm(b - a)) < 1e-6:
        raise ValueError("A line needs two distinct points.")
    return np.cross(np.array([a[0], a[1], 1.0]), np.array([b[0], b[1], 1.0]))


def _intersect_lines(line_a, line_b, parallel_epsilon=1e-9):
    point = np.cross(line_a, line_b)
    if abs(float(point[2])) <= parallel_epsilon:
        return None
    return point[:2] / point[2]


def extrapolate_fourth_corner(known_corners, edge_points=None, missing_corner="bottom_left", allow_swapped_edges=True):
    """Estimate the hidden rectangle corner from three corners plus optional edge cues.

    known_corners must contain top_left, top_right, bottom_right, bottom_left
    except for the missing corner.

    For deterministic calibration, pass edge_points with "edge_from_prev" and
    "edge_from_next". They mean: one point on the line from the clockwise
    previous corner into the hidden corner, and one point on the line from the
    clockwise next corner into the hidden corner. Legacy "edge_a"/"edge_b"
    inputs can still be tried in both assignments when allow_swapped_edges is
    True.
    """
    order = ["top_left", "top_right", "bottom_right", "bottom_left"]
    if missing_corner not in order:
        raise ValueError(f"missing_corner must be one of {order}")

    points = {name: (_as_point(value) if value is not None else None) for name, value in known_corners.items()}
    if missing_corner in points and points[missing_corner] is not None:
        return points[missing_corner]

    missing_index = order.index(missing_corner)
    prev_name = order[(missing_index - 1) % 4]
    next_name = order[(missing_index + 1) % 4]
    opposite_name = order[(missing_index + 2) % 4]

    for name in (prev_name, next_name, opposite_name):
        if points.get(name) is None:
            raise ValueError(f"known_corners is missing required corner: {name}")

    edge_points = edge_points or {}
    edge_from_prev = edge_points.get(f"{missing_corner}_edge_from_prev", edge_points.get("edge_from_prev"))
    edge_from_next = edge_points.get(f"{missing_corner}_edge_from_next", edge_points.get("edge_from_next"))
    edge_a = edge_points.get(f"{missing_corner}_edge_a", edge_points.get("edge_a"))
    edge_b = edge_points.get(f"{missing_corner}_edge_b", edge_points.get("edge_b"))

    fallback = points[prev_name] + points[next_name] - points[opposite_name]
    if edge_from_prev is not None and edge_from_next is not None:
        line_from_prev = _line_from_points(points[prev_name], edge_from_prev)
        line_from_next = _line_from_points(points[next_name], edge_from_next)
        intersection = _intersect_lines(line_from_prev, line_from_next)
        if intersection is not None and np.all(np.isfinite(intersection)):
            return intersection.astype(float)

    if edge_a is not None and edge_b is not None:
        candidates = []
        edge_assignments = [(edge_a, edge_b)]
        if allow_swapped_edges:
            edge_assignments.append((edge_b, edge_a))
        for prev_edge, next_edge in edge_assignments:
            line_from_prev = _line_from_points(points[prev_name], prev_edge)
            line_from_next = _line_from_points(points[next_name], next_edge)
            intersection = _intersect_lines(line_from_prev, line_from_next)
            if intersection is not None and np.all(np.isfinite(intersection)):
                candidates.append(intersection.astype(float))
        if candidates:
            return min(candidates, key=lambda point: float(np.linalg.norm(point - fallback)))

    # Fallback for a fronto-parallel or already-rectified rectangle.
    return fallback


def update_map_motion(memory, memory_key, map_point_cm, timestamp, max_speed_mps=DEFAULT_MAX_PERSON_SPEED_MPS, ema_alpha=DEFAULT_MAP_POSITION_EMA_ALPHA):
    if memory is None or memory_key is None or map_point_cm is None or timestamp is None:
        return map_point_cm, None, "no_id"

    raw_point = np.array(map_point_cm, dtype=float)
    previous = memory.get(memory_key)
    if previous is None:
        memory[memory_key] = {"point": raw_point, "timestamp": float(timestamp), "speed_mps": 0.0}
        return tuple(raw_point), 0.0, "new"

    previous_point = np.array(previous["point"], dtype=float)
    dt = float(timestamp) - float(previous["timestamp"])
    if dt <= 1e-6:
        return tuple(previous_point), previous.get("speed_mps"), "same_time"

    raw_speed_mps = distance_cm(raw_point, previous_point) / 100.0 / dt
    if raw_speed_mps > max_speed_mps:
        memory[memory_key] = {"point": previous_point, "timestamp": float(timestamp), "speed_mps": raw_speed_mps}
        return tuple(previous_point), raw_speed_mps, "speed_hold"

    smoothed_point = (1.0 - ema_alpha) * previous_point + ema_alpha * raw_point
    smoothed_speed_mps = distance_cm(smoothed_point, previous_point) / 100.0 / dt
    memory[memory_key] = {"point": smoothed_point, "timestamp": float(timestamp), "speed_mps": smoothed_speed_mps}
    return tuple(smoothed_point), smoothed_speed_mps, "smooth"
