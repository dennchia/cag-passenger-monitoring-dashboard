from datetime import datetime, timezone #the standard format like yyyy-mm-ddThh:mm:ssZ is used for backend timestamping, so we need these to get the current time in UTC and format it correctly for the backend payload
try:
    from ultralytics import YOLO #human detection model 
except ImportError:
    YOLO = None
import base64 #in case I need to send out the image 
import cv2 #for the homography transformation
import argparse #for parsing command line arguments to make the script more flexible and configurable without changing the code
import json #mainly for saving and loading the homography matrix and also for formatting the MQTT and backend payloads
import time #to push data for every 0.5 seconds to mqtt and backend (line 259)
import urllib.error #to catch backend connection errors
import urllib.parse #to construct URL properly when combining base URL and path
import urllib.request #to send HTTP POST requests to the backend with the latest metrics and counts
try:
    import paho.mqtt.client as mqtt #for MQTT communication to publish the tactical data and metrics to a broker, which can then be consumed by other services or dashboards in real-time
except ImportError:
    mqtt = None
from pathlib import Path #checking url like homogaphy_matrix.json exists or else and also for saving the homography matrix in a clean way(Line 210 219)

import numpy as np
from constants import *
from core_math import (
    camera_point_to_map,
    distance_cm,
    extrapolate_fourth_corner,
    resolve_memory_key,
    update_map_motion,
)
from camera_stream import CameraContext, LiveCamera, resize_to_fit
from pose_engine import (
    create_mediapipe_pose_estimator,
    get_standing_points,
)
from reid_memory import (
    AppearanceIdentityMemory,
    TransReIDFeatureExtractor,
)

# import yolo11n for human detection(from cv enginner mikail) cv2 for tranformation functions and numpy for the geometric coordinate matrices



# from cv engineer

#used to downgrade the resolution for display purposes without affecting the original frame used for detection and homography mapping, which can be computationally expensive at higher resolutions. This way we can maintain a smooth display while still processing the full-resolution frames for accurate detection and mapping.

def save_homography(path, matrix, image_points, map_size_cm):
    payload = {
        "matrix": matrix.tolist(),
        "image_points": image_points.tolist(),
        "map_size_cm": map_size_cm,
        "point_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
#This function saves the homography matrix along with the original image points and map size to a JSON file. This allows for easy loading and reference in future runs without needing to recalibrate every time, as long as the camera setup remains unchanged. The additional metadata about point order can help ensure that the calibration points are interpreted correctly when loading the homography later on.

def load_homography(path):
    payload = json.loads(path.read_text(encoding="utf-8"))#encoding must be same as saved
    return np.array(payload["matrix"], dtype=np.float32), int(payload.get("map_size_cm", 300))


def collect_calibration_points(frame, map_size_cm, matrix_path, missing_corner=None):
    clicked_points = []
    display_frame, scale = resize_to_fit(frame)
    corner_order = ["top_left", "top_right", "bottom_right", "bottom_left"]
    corner_labels = {
        "top_left": "TL",
        "top_right": "TR",
        "bottom_right": "BR",
        "bottom_left": "BL",
    }
    clockwise_click_specs = {
        "top_left": [
            ("edge_from_next", "top edge near TL"),
            ("top_right", "TR"),
            ("bottom_right", "BR"),
            ("bottom_left", "BL"),
            ("edge_from_prev", "left edge near TL"),
        ],
        "top_right": [
            ("top_left", "TL"),
            ("edge_from_prev", "top edge near TR"),
            ("edge_from_next", "right edge near TR"),
            ("bottom_right", "BR"),
            ("bottom_left", "BL"),
        ],
        "bottom_right": [
            ("top_left", "TL"),
            ("top_right", "TR"),
            ("edge_from_prev", "right edge near BR"),
            ("edge_from_next", "bottom edge near BR"),
            ("bottom_left", "BL"),
        ],
        "bottom_left": [
            ("top_left", "TL"),
            ("top_right", "TR"),
            ("bottom_right", "BR"),
            ("edge_from_prev", "bottom edge near BL"),
            ("edge_from_next", "left edge near BL"),
        ],
    }
    if missing_corner is not None and missing_corner not in corner_order:
        raise ValueError(f"Missing corner must be one of: {', '.join(corner_order)}")

    click_specs = clockwise_click_specs.get(missing_corner)
    required_clicks = 5 if missing_corner else 4
    window_name = "Calibration: smart missing corner" if missing_corner else "Calibration: click TL, TR, BR, BL then press Enter"

    def build_missing_corner_points():
        known_corners = {}
        edge_points = {}
        for point, (point_key, _label) in zip(clicked_points, click_specs):
            if point_key.startswith("edge_"):
                edge_points[point_key] = point
            else:
                known_corners[point_key] = point
        missing_point = extrapolate_fourth_corner(
            known_corners,
            edge_points,
            missing_corner=missing_corner,
            allow_swapped_edges=False,
        )
        return known_corners, missing_point

    def mouse_callback(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or len(clicked_points) >= required_clicks:
            return

        clicked_points.append([x / scale, y / scale])

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        preview = display_frame.copy()
        for index, (point_x, point_y) in enumerate(clicked_points):
            display_x = int(point_x * scale)
            display_y = int(point_y * scale)
            cv2.circle(preview, (display_x, display_y), 6, (0, 0, 255), -1)
            cv2.putText(
                preview,
                str(index + 1),
                (display_x + 8, display_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        preview_points = clicked_points
        if missing_corner and len(clicked_points) == required_clicks:
            try:
                known_corners, missing_point = build_missing_corner_points()
                preview_points = [known_corners.get(corner, missing_point) for corner in corner_order]
            except ValueError:
                preview_points = clicked_points

        if len(preview_points) == 4:
            display_points = [[point_x * scale, point_y * scale] for point_x, point_y in preview_points]
            pts = np.array(display_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(preview, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        if missing_corner:
            next_label = click_specs[len(clicked_points)][1] if len(clicked_points) < required_clicks else "Enter to save"
            full_order = " -> ".join(label for _key, label in click_specs)
            instruction = f"Clockwise order: {full_order}. Next: {next_label}. Enter=save, R=reset, Q=quit"
        else:
            instruction = "Click 4 floor corners in order: TL, TR, BR, BL. Enter=save, R=reset, Q=quit"

        cv2.putText(
            preview,
            instruction,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.imshow(window_name, preview)

        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(clicked_points) == required_clicks:
            break
        if key == ord("r"):
            clicked_points.clear()
        if key == ord("q"):
            cv2.destroyWindow(window_name)
            raise SystemExit("Calibration cancelled.")

    cv2.destroyWindow(window_name)

    if missing_corner:
        known_corners, missing_point = build_missing_corner_points()
        ordered_points = [known_corners.get(corner, missing_point) for corner in corner_order]
        image_points = np.array(ordered_points, dtype=np.float32)
    else:
        image_points = np.array(clicked_points, dtype=np.float32)
    map_points = np.array(
        [[0, 0], [map_size_cm, 0], [map_size_cm, map_size_cm], [0, map_size_cm]],
        dtype=np.float32,
    )

    matrix, _ = cv2.findHomography(image_points, map_points)
    if matrix is None:
        raise RuntimeError("Unable to calculate homography from selected points.")

    save_homography(matrix_path, matrix, image_points, map_size_cm)
    return matrix.astype(np.float32)


































































def create_tactical_map(points_cm, map_size_cm, title="Tactical map", color=(0, 160, 0)):
    canvas = np.full((TACTICAL_MAP_SIZE, TACTICAL_MAP_SIZE, 3), 245, dtype=np.uint8)
    scale = TACTICAL_MAP_SIZE / map_size_cm

    cv2.rectangle(canvas, (0, 0), (TACTICAL_MAP_SIZE - 1, TACTICAL_MAP_SIZE - 1), (40, 40, 40), 2)

    for meter in range(1, int(map_size_cm / 100) + 1):
        pos = int(meter * 100 * scale)
        cv2.line(canvas, (pos, 0), (pos, TACTICAL_MAP_SIZE), (210, 210, 210), 1)
        cv2.line(canvas, (0, pos), (TACTICAL_MAP_SIZE, pos), (210, 210, 210), 1)

    for index, (map_x, map_y) in enumerate(points_cm, start=1):
        pixel_x = int(round(map_x * scale))
        pixel_y = int(round(map_y * scale))
        in_zone = 0 <= map_x <= map_size_cm and 0 <= map_y <= map_size_cm
        point_color = color if in_zone else (0, 0, 255)
        pixel_x = max(0, min(TACTICAL_MAP_SIZE - 1, pixel_x))
        pixel_y = max(0, min(TACTICAL_MAP_SIZE - 1, pixel_y))
        cv2.circle(canvas, (pixel_x, pixel_y), 9, point_color, -1)
        cv2.putText(
            canvas,
            str(index),
            (pixel_x + 10, pixel_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            point_color,
            2,
        )

    cv2.putText(canvas, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    return canvas


def create_mqtt_client(host, port, client_id="tactical-publisher", username=None, password=None):
    try:
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION_2, client_id=client_id)
        except (AttributeError, TypeError):
            client = mqtt.Client(client_id=client_id)
        client.reconnect_delay_set(min_delay=1, max_delay=10)
        client.on_connect = lambda _client, _userdata, _flags, reason_code, *_args: print(
            f"MQTT connected to {host}:{port} with result {reason_code}"
        )
        client.on_disconnect = lambda _client, _userdata, reason_code, *_args: print(
            f"MQTT disconnected with result {reason_code}. Will try to reconnect before publishing."
        )
        if username is not None and password is not None:
            client.username_pw_set(username, password)
        client.connect(host, port, keepalive=60)
        client.loop_start()
        return client
    except Exception as e:
        print(f"MQTT connection failed: {e}. Continuing without MQTT.")
        return None


def ensure_mqtt_connected(client):
    if client is None:
        return False

    try:
        if hasattr(client, "is_connected") and client.is_connected():
            return True
        print("MQTT client is disconnected. Attempting reconnect...")
        client.reconnect()
        return True
    except Exception as e:
        print(f"MQTT reconnect failed: {e}")
        return False


def publish_mqtt_json(client, topic, payload, qos=1):
    result = client.publish(topic, json.dumps(payload), qos=qos)
    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        raise RuntimeError(f"publish to {topic} failed with rc={result.rc}")
    return result


def encode_image_to_base64(image, quality=80):
    success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return None
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def post_json(url, payload, timeout=5):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Detect people feet and project them to a tactical map.")
    parser.add_argument("--source", default=DEFAULT_RTSP_URL, help="Camera/video source. Use 0 for webcam.")
    parser.add_argument("--source-2", default=None, help="Optional second camera/video source.")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model path.")
    parser.add_argument("--use-mediapipe-feet", action="store_true", help="Use MediaPipe heel/toe landmarks inside each YOLO person box.")
    parser.add_argument("--mediapipe-model", default=DEFAULT_MEDIAPIPE_MODEL_PATH, help="MediaPipe pose landmarker .task model path.")
    parser.add_argument("--matrix", default="homography_matrix.json", help="Saved homography file for camera 1.")
    parser.add_argument("--matrix-2", default=None, help="Optional saved homography file for camera 2.")
    parser.add_argument("--setup", action="store_true", help="Force the 4-click homography setup for available cameras.")
    parser.add_argument("--missing-corner", choices=["top_left", "top_right", "bottom_right", "bottom_left"], default=None, help="Camera 1 hidden calibration corner. Click the other 3 corners plus 2 points on the hidden corner edges.")
    parser.add_argument("--missing-corner-2", choices=["top_left", "top_right", "bottom_right", "bottom_left"], default=None, help="Camera 2 hidden calibration corner.")
    parser.add_argument("--map-size-cm", type=int, default=300, help="Tactical map side length in centimeters.")
    parser.add_argument("--outside-context-cm", type=int, default=700, help="Outside visible context range around the map, in centimeters.")
    parser.add_argument("--conf", type=float, default=0.4, help="YOLO confidence threshold.")
    parser.add_argument("--fusion-distance-cm", type=float, default=DEFAULT_FUSION_DISTANCE_CM, help="Maximum distance for two camera detections to count as the same person.")
    parser.add_argument("--pose-dropout-ttl-frames", type=int, default=DEFAULT_POSE_DROPOUT_TTL_FRAMES, help="Frames to keep a tracked person using last known foot point when pose landmarks disappear.")
    parser.add_argument("--use-appearance-reid", action="store_true", help="Use crop appearance memory to keep stable IDs when ByteTrack changes IDs.")
    parser.add_argument("--reid-checkpoint", default="transreid_msmt17.pth", help="Path to a TransReID checkpoint for appearance feature extraction.")
    parser.add_argument("--fastreid-root", default="fast-reid", help="Path to the extracted fast-reid folder used by the TransReID checkpoint.")
    parser.add_argument("--reid-db", default="evacuee_database.pkl", help="Persistent ReID gallery database file.")
    parser.add_argument("--no-persistent-reid-db", action="store_true", help="Keep ReID identities in memory only for this run.")
    parser.add_argument("--reid-intake-frames", type=int, default=5, help="Rapid crops averaged into one denoised baseline fingerprint for a new track.")
    parser.add_argument("--reid-gallery-update-interval-frames", type=int, default=30, help="Frames between long-term ReID gallery angle updates for the same tracked person.")
    parser.add_argument("--reid-evidence-dir", default="angle_evidence", help="Folder for labeled ReID baseline and gallery angle crop snapshots.")
    parser.add_argument("--no-reid-evidence", action="store_true", help="Disable saving labeled ReID crop evidence snapshots.")
    parser.add_argument("--reid-similarity-threshold", type=float, default=DEFAULT_REID_SIMILARITY_THRESHOLD, help="Appearance similarity needed to reuse an old stable ID.")
    parser.add_argument("--reid-memory-ttl-frames", type=int, default=DEFAULT_REID_MEMORY_TTL_FRAMES, help="Frames to remember old appearance IDs for re-identification.")
    parser.add_argument("--run-id", default="default", help="Run identifier for backend tracking.")
    parser.add_argument("--camera-id", default="cam_1", help="Camera identifier for backend tracking.")
    parser.add_argument("--camera-id-2", default="cam_2", help="Optional second camera identifier.")
    parser.add_argument("--mqtt-broker", default=None, help="MQTT broker hostname or IP address.")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port.")
    parser.add_argument("--mqtt-topic", default="cag/tactical", help="MQTT topic to publish tactical data.")
    parser.add_argument("--mqtt-metrics-topic", default="cag/metrics", help="MQTT topic to publish metric/count data.")
    parser.add_argument("--mqtt-publish-interval", type=float, default=0.5, help="Seconds between MQTT publishes.")
    parser.add_argument("--mqtt-client-id", default="tactical-publisher", help="MQTT client identifier.")
    parser.add_argument("--mqtt-username", default=None, help="MQTT username if broker requires authentication.")
    parser.add_argument("--mqtt-password", default=None, help="MQTT password if broker requires authentication.")
    parser.add_argument("--mqtt-send-map-image", action="store_true", help="Send tactical map image in MQTT payload as base64.")
    parser.add_argument("--mqtt-image-quality", type=int, default=80, help="JPEG quality for MQTT map image (1-100).")
    parser.add_argument("--backend-url", default=None, help="HTTP backend base URL for POST updates.")
    parser.add_argument("--backend-path", default="/api/metrics", help="Backend API path for POST updates.")
    parser.add_argument("--http-timeout", type=int, default=5, help="Timeout for backend HTTP POST requests.")
    return parser.parse_args()




def build_camera_contexts(args):
    contexts = []
    camera_sources = [args.source]
    camera_ids = [args.camera_id]
    matrix_paths = [Path(args.matrix)]

    if args.source_2:
        camera_sources.append(args.source_2)
        camera_ids.append(args.camera_id_2)
        matrix_paths.append(Path(args.matrix_2 if args.matrix_2 else f"{Path(args.matrix).stem}_2{Path(args.matrix).suffix}"))

    for camera_id, source, matrix_path in zip(camera_ids, camera_sources, matrix_paths):
        source_value = int(source) if str(source).isdigit() else source
        contexts.append(CameraContext(camera_id, source_value, matrix_path, args.map_size_cm))

    if contexts:
        contexts[0].missing_corner = args.missing_corner
    if len(contexts) > 1:
        contexts[1].missing_corner = args.missing_corner_2

    return contexts


def ensure_homographies(contexts, setup_force):
    for context in contexts:
        if context.cap is None:
            continue

        success, first_frame = context.cap.read()
        if not success or first_frame is None:
            raise RuntimeError(f"Unable to read first frame for camera {context.camera_id}")

        if context.matrix_path.exists() and not setup_force:
            context.homography, context.map_size_cm = load_homography(context.matrix_path)
            print(f"Loaded homography for {context.camera_id} from {context.matrix_path}")
        else:
            print(f"Calibrating homography for {context.camera_id}...")
            context.homography = collect_calibration_points(
                first_frame,
                context.map_size_cm,
                context.matrix_path,
                missing_corner=context.missing_corner,
            )
            print(f"Saved homography for {context.camera_id} to {context.matrix_path}")







def process_camera_frame(context, conf, pose_estimator=None, pose_dropout_ttl_frames=DEFAULT_POSE_DROPOUT_TTL_FRAMES):
    context.tactical_points = []
    speed_debug_lines = []
    frame_timestamp = time.monotonic()
    success, frame = context.cap.read()
    context.raw_frame = frame if success else None

    if not success or frame is None:
        context.annotated_frame = None
        return False

    context.frame_index += 1
    results = context.model.track(frame, classes=[0], conf=conf, persist=True, tracker="bytetrack.yaml", verbose=False)
    result = results[0]
    annotated_frame = frame.copy() if pose_estimator is not None else result.plot()

    for index, standing_point in enumerate(get_standing_points(
        result,
        frame,
        pose_estimator,
        anatomical_ratio_memory=context.anatomical_ratio_memory,
        anatomical_anchor_memory=context.anatomical_anchor_memory,
        last_foot_memory=context.last_foot_memory,
        frame_index=context.frame_index,
        pose_dropout_ttl_frames=pose_dropout_ttl_frames,
        annotated_frame=annotated_frame,
        appearance_memory=context.appearance_memory,
    )):
        point = standing_point["point"]
        speed_mps = None
        motion_status = None
        if point is not None:
            feet_x, feet_y = point
            raw_map_x, raw_map_y = camera_point_to_map((feet_x, feet_y), context.homography)
            motion_key = resolve_memory_key(standing_point.get("track_id"), standing_point.get("identity_id"))
            (map_x, map_y), speed_mps, motion_status = update_map_motion(
                context.map_motion_memory,
                motion_key,
                (raw_map_x, raw_map_y),
                frame_timestamp,
            )
            context.tactical_points.append((map_x, map_y))

            cv2.circle(annotated_frame, (feet_x, feet_y), radius=8, color=(0, 0, 255), thickness=-1)
            label_anchor = (feet_x + 10, feet_y - 10)
            label = f"({map_x:.0f}cm, {map_y:.0f}cm)"
            person_label = f"ID {standing_point['identity_id']}" if standing_point.get("identity_id") is not None else f"T{standing_point['track_id']}"
            speed_text = "n/a" if speed_mps is None else f"{speed_mps:.2f}m/s"
            speed_debug_lines.append(f"{person_label}: {speed_text} {motion_status} {standing_point['method']}")
        else:
            label_anchor = (20, 40 + index * 22)
            label = "(no visible foot)"

        if standing_point.get("identity_id") is not None:
            label = f"ID {standing_point['identity_id']} {label}"
            if standing_point.get("reidentified"):
                label = f"{label} reid={standing_point['reid_similarity']:.2f}"
        elif standing_point.get("reid_intake_required", 0) > 1 and standing_point.get("reid_intake_count", 0) > 0:
            label = f"ANALYZING ({standing_point['reid_intake_count']}/{standing_point['reid_intake_required']}) {label}"
        elif standing_point["track_id"] is not None:
            label = f"T{standing_point['track_id']} {label}"
        ratio = standing_point.get("ratio")
        if speed_mps is not None:
            label = f"{label} v={speed_mps:.2f}m/s"
            if motion_status == "speed_hold":
                label = f"{label} HOLD"
        head_pitch = standing_point.get("head_pitch")
        if head_pitch == "looking_straight":
            label = f"{label} head=up"
        elif head_pitch == "looking_down":
            label = f"{label} head=down"
        elif head_pitch == "unknown":
            label = f"{label} head=?"
        if ratio is not None:
            label = f"{label} r={ratio:.3f}"
        if standing_point["method"] == "anatomical_ratio":
            label = f"{label} calc"
        elif standing_point["method"] == "last_seen":
            label = f"{label} last"
        elif standing_point["method"] == "physics_hold":
            label = f"{label} physics"
        elif standing_point["method"] == "no_visible_ankle":
            label = f"{label} no-ankle"
        cv2.putText(
            annotated_frame,
            label,
            label_anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    if speed_debug_lines:
        overlay_height = 26 + 22 * len(speed_debug_lines)
        y1 = max(0, annotated_frame.shape[0] - overlay_height)
        cv2.rectangle(annotated_frame, (0, y1), (annotated_frame.shape[1], annotated_frame.shape[0]), (0, 0, 0), -1)
        for line_index, line in enumerate(speed_debug_lines):
            cv2.putText(
                annotated_frame,
                line,
                (12, y1 + 24 + line_index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

    context.annotated_frame = annotated_frame
    return True




def fuse_camera_points(camera_points, max_distance_cm):
    fused_people = []

    for camera_id, points_cm in camera_points.items():
        for point in points_cm:
            best_person = None
            best_distance = None

            for person in fused_people:
                if camera_id in person["sources"]:
                    continue

                current_distance = distance_cm(point, person["center"])
                if current_distance <= max_distance_cm and (best_distance is None or current_distance < best_distance):
                    best_person = person
                    best_distance = current_distance

            if best_person is None:
                fused_people.append({
                    "center": (float(point[0]), float(point[1])),
                    "points": [(float(point[0]), float(point[1]))],
                    "sources": [camera_id],
                })
            else:
                best_person["points"].append((float(point[0]), float(point[1])))
                best_person["sources"].append(camera_id)
                best_person["center"] = tuple(np.mean(np.array(best_person["points"]), axis=0))

    return fused_people


def create_combined_tactical_map(fused_people, map_size_cm):
    canvas = np.full((TACTICAL_MAP_SIZE, TACTICAL_MAP_SIZE, 3), 245, dtype=np.uint8)
    scale = TACTICAL_MAP_SIZE / map_size_cm

    cv2.rectangle(canvas, (0, 0), (TACTICAL_MAP_SIZE - 1, TACTICAL_MAP_SIZE - 1), (40, 40, 40), 2)
    for meter in range(1, int(map_size_cm / 100) + 1):
        pos = int(meter * 100 * scale)
        cv2.line(canvas, (pos, 0), (pos, TACTICAL_MAP_SIZE), (210, 210, 210), 1)
        cv2.line(canvas, (0, pos), (TACTICAL_MAP_SIZE, pos), (210, 210, 210), 1)

    for person_index, person in enumerate(fused_people, start=1):
        map_x, map_y = person["center"]
        pixel_x = int(round(map_x * scale))
        pixel_y = int(round(map_y * scale))
        in_zone = 0 <= map_x <= map_size_cm and 0 <= map_y <= map_size_cm
        point_color = (0, 160, 0) if len(person["sources"]) > 1 else (0, 120, 255)
        if not in_zone:
            point_color = (0, 0, 255)

        pixel_x = max(0, min(TACTICAL_MAP_SIZE - 1, pixel_x))
        pixel_y = max(0, min(TACTICAL_MAP_SIZE - 1, pixel_y))
        cv2.circle(canvas, (pixel_x, pixel_y), 11, point_color, -1)
        cv2.putText(
            canvas,
            f"P{person_index}",
            (pixel_x + 10, pixel_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            point_color,
            2,
        )
        cv2.putText(
            canvas,
            "+".join(person["sources"]),
            (pixel_x + 10, pixel_y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            point_color,
            1,
        )

    cv2.putText(canvas, "Combined fused map", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    return canvas


def classify_map_point(point, map_size_cm, outside_context_cm):
    try:
        x, y = point
        x = float(x)
        y = float(y)
        map_size_cm = float(map_size_cm)
        outside_context_cm = float(outside_context_cm)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(x) or not np.isfinite(y):
        return None
    if 0 <= x <= map_size_cm and 0 <= y <= map_size_cm:
        return "inside"
    if -outside_context_cm <= x <= map_size_cm + outside_context_cm and -outside_context_cm <= y <= map_size_cm + outside_context_cm:
        return "outside_visible"
    return None


def build_payloads(contexts, args, fused_people, combined_map=None):
    map_size_cm = max(context.map_size_cm for context in contexts) if contexts else args.map_size_cm
    inside_count = 0
    outside_visible_count = 0
    positions = []
    zone_counts = {}

    for context in contexts:
        zone_counts[context.camera_id] = sum(
            1
            for point in context.tactical_points
            if classify_map_point(point, context.map_size_cm, args.outside_context_cm) == "inside"
        )

    for person_index, person in enumerate(fused_people, start=1):
        x, y = person["center"]
        area = classify_map_point((x, y), map_size_cm, args.outside_context_cm)
        if area is None:
            continue
        if area == "inside":
            inside_count += 1
        elif area == "outside_visible":
            outside_visible_count += 1
        positions.append({
            "person_id": f"P{person_index}",
            "sources": person["sources"],
            "x": round(float(x), 1),
            "y": round(float(y), 1),
            "area": area,
        })

    tactical_payload = {
        "timestamp": int(time.time()),
        "camera_id": "fused",
        "run_id": args.run_id,
        "people_count": inside_count,
        "inside_count": inside_count,
        "outside_visible_count": outside_visible_count,
        "total_visible_count": inside_count + outside_visible_count,
        "positions_cm": positions,
        "map_size_cm": map_size_cm,
        "outside_context_cm": args.outside_context_cm,
        "zone_counts": zone_counts,
        "camera_online_count": sum(1 for context in contexts if context.cap.is_opened()),
    }

    if combined_map is not None and args.mqtt_send_map_image:
        image_b64 = encode_image_to_base64(combined_map, args.mqtt_image_quality)
        if image_b64 is not None:
            tactical_payload["tactical_map_image"] = image_b64

    metric_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "passenger_count": inside_count,
        "zone_counts": zone_counts,
        "camera_online_count": tactical_payload["camera_online_count"],
    }

    return tactical_payload, metric_payload


def main():
    args = parse_args()
    contexts = build_camera_contexts(args)

    pose_estimator = create_mediapipe_pose_estimator(args.use_mediapipe_feet, Path(args.mediapipe_model))

    reid_extractor = None
    if args.use_appearance_reid:
        reid_extractor = TransReIDFeatureExtractor(Path(args.reid_checkpoint), device="cuda", fastreid_root=args.fastreid_root)
        if not reid_extractor.is_available():
            print("Appearance ReID will use color histograms only. IDs may change more easily after a person disappears.")

    for context in contexts:
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO is not installed. Install ultralytics or disable camera processing.")
        context.model = YOLO(args.model)
        if args.use_appearance_reid:
            context.appearance_memory = AppearanceIdentityMemory(
                similarity_threshold=args.reid_similarity_threshold,
                ttl_frames=args.reid_memory_ttl_frames,
                ema_alpha=DEFAULT_REID_EMA_ALPHA,
                reid_extractor=reid_extractor,
                db_path=None if args.no_persistent_reid_db else args.reid_db,
                intake_frames=args.reid_intake_frames,
                gallery_update_interval_frames=args.reid_gallery_update_interval_frames,
                evidence_dir=None if args.no_reid_evidence else args.reid_evidence_dir,
            )
        context.cap = LiveCamera(context.source)
        if not context.cap.is_opened():
            raise RuntimeError(f"Unable to open video source {context.source} for camera {context.camera_id}")

    ensure_homographies(contexts, args.setup)

    mqtt_client = None
    if args.mqtt_broker:
        print(f"Attempting to connect to MQTT broker at {args.mqtt_broker}:{args.mqtt_port}...")
        mqtt_client = create_mqtt_client(
            args.mqtt_broker,
            args.mqtt_port,
            client_id=args.mqtt_client_id,
            username=args.mqtt_username,
            password=args.mqtt_password,
        )

    backend_post_url = None
    if args.backend_url:
        backend_post_url = urllib.parse.urljoin(args.backend_url.rstrip("/") + "/", args.backend_path.lstrip("/"))

    last_mqtt_publish = 0.0

    while True:
        active_camera = False
        for context in contexts:
            processed = process_camera_frame(context, args.conf, pose_estimator, args.pose_dropout_ttl_frames)
            active_camera = active_camera or processed

        if not active_camera:
            print("No camera frames available. Exiting.")
            break

        camera_points = {context.camera_id: context.tactical_points for context in contexts}
        map_size_cm = max(context.map_size_cm for context in contexts)
        fused_people = fuse_camera_points(camera_points, args.fusion_distance_cm)
        combined_map = create_combined_tactical_map(fused_people, map_size_cm)

        for context in contexts:
            if context.annotated_frame is not None:
                display_frame, _ = resize_to_fit(context.annotated_frame)
                cv2.imshow(f"Camera {context.camera_id}", display_frame)
                tactical_map = create_tactical_map(
                    context.tactical_points,
                    map_size_cm,
                    title=f"{context.camera_id} tactical map",
                )
                cv2.imshow(f"Map {context.camera_id}", tactical_map)

        cv2.imshow("Combined tactical map", combined_map)

        if mqtt_client is not None:
            now = time.monotonic()
            if now - last_mqtt_publish >= args.mqtt_publish_interval:
                last_mqtt_publish = now
                tactical_payload, metric_payload = build_payloads(contexts, args, fused_people, combined_map)
                try:
                    if ensure_mqtt_connected(mqtt_client):
                        publish_mqtt_json(mqtt_client, args.mqtt_topic, tactical_payload, qos=1)
                        publish_mqtt_json(mqtt_client, args.mqtt_metrics_topic, metric_payload, qos=1)
                except Exception as e:
                    print(f"MQTT publish failed: {e}")

        if backend_post_url:
            tactical_payload, _ = build_payloads(contexts, args, fused_people, combined_map)
            try:
                post_json(backend_post_url, tactical_payload, timeout=args.http_timeout)
            except urllib.error.URLError as exc:
                print(f"Backend POST failed: {exc}")
            except Exception as exc:
                print(f"Unexpected backend POST error: {exc}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            print("Recalibrating all cameras...")
            ensure_homographies(contexts, setup_force=True)

    for context in contexts:
        if context.cap is not None:
            context.cap.release()

    if pose_estimator is not None:
        pose_estimator.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()






