import torch

# Fixed-size camera frames benefit from cuDNN's convolution autotuner.  It is
# safe to enable this here because the tracker does not train or change input
# tensor shapes during a run.
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# This will print True if your GPU is ready, or False if it is stuck on CPU
print("Is CUDA available?:", torch.cuda.is_available())

# This will print the actual name of your laptop's graphics card (e.g., RTX 4060)
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

import concurrent.futures #to run each camera's YOLO+pose+ReID pipeline in parallel instead of one after another
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
                1.4,
                (0, 0, 255),
                3,
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
            1.2,
            (255, 255, 255),
            3,
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
            1.0,
            point_color,
            3,
        )

    cv2.putText(canvas, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    return canvas


def create_mqtt_client(host, port, client_id="tactical-publisher", username=None, password=None):
    try:
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION_2, client_id=client_id)
        except (AttributeError, TypeError):
            client = mqtt.Client(client_id=client_id)
        if username is not None and password is not None:
            client.username_pw_set(username, password)
        client.connect(host, port, keepalive=60)
        client.loop_start()
        print(f"MQTT connected to {host}:{port}")
        return client
    except Exception as e:
        print(f"MQTT connection failed: {e}. Continuing without MQTT.")
        return None


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
    parser.add_argument("--conf", type=float, default=0.4, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=DEFAULT_YOLO_NMS_IOU, help="YOLO NMS IoU threshold. Lower values suppress overlapping duplicate boxes more aggressively.")
    parser.add_argument("--tracker-config", default=DEFAULT_TRACKER_CONFIG_PATH, help="Project ByteTrack YAML path.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size. Lower values improve FPS at some accuracy cost.")
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FP16 YOLO inference on CUDA (disable with --no-half if needed).",
    )
    parser.add_argument("--device", type=str, default="0", help="Device to run YOLO on (e.g., 0, 1, cpu)")
    parser.add_argument("--fusion-distance-cm", type=float, default=DEFAULT_FUSION_DISTANCE_CM, help="Maximum distance for two camera detections to count as the same person.")
    parser.add_argument("--cross-camera-max-skew-seconds", type=float, default=DEFAULT_CROSS_CAMERA_MAX_SKEW_SECONDS, help="Maximum capture-time difference for cross-camera association.")
    parser.add_argument("--pose-dropout-ttl-frames", type=int, default=DEFAULT_POSE_DROPOUT_TTL_FRAMES, help="Frames to keep a tracked person using last known foot point when pose landmarks disappear.")
    parser.add_argument("--use-appearance-reid", action="store_true", help="Use crop appearance memory to keep stable IDs when ByteTrack changes IDs.")
    parser.add_argument("--reid-checkpoint", default="transreid_msmt17.pth", help="Path to a TransReID checkpoint for appearance feature extraction.")
    parser.add_argument("--fastreid-root", default="fast-reid", help="Path to the extracted fast-reid folder used by the TransReID checkpoint.")
    parser.add_argument("--reid-db", default="evacuee_database_v7.pkl", help="Persistent ReID gallery database file.")
    parser.add_argument("--no-persistent-reid-db", action="store_true", help="Keep ReID identities in memory only for this run.")
    parser.add_argument("--reid-intake-frames", type=int, default=5, help="Rapid crops averaged into the temporary matching query; the best crop and its own vector become baseline.")
    parser.add_argument("--reid-gallery-update-interval-frames", type=int, default=DEFAULT_REID_SEMANTIC_COOLDOWN_FRAMES, help="Frames to wait after successfully queuing a missing semantic gallery view.")
    parser.add_argument("--reid-evidence-dir", default="angle_evidence_v7", help="Folder for raw ReID baseline and semantic-view crop snapshots.")
    parser.add_argument("--no-reid-evidence", action="store_true", help="Disable saving labeled ReID crop evidence snapshots.")
    parser.add_argument("--reid-similarity-threshold", type=float, default=DEFAULT_REID_SIMILARITY_THRESHOLD, help="Appearance similarity needed to reuse an old stable ID.")
    parser.add_argument("--reid-distance-threshold", type=float, default=DEFAULT_REID_DISTANCE_THRESHOLD, help="Strict cosine-distance threshold for ReID; a match must be below this value.")
    parser.add_argument("--reid-memory-ttl-frames", type=int, default=DEFAULT_REID_MEMORY_TTL_FRAMES, help="Frames to retain stale local tracker bindings; persistent master galleries do not expire.")
    parser.add_argument("--reid-intake-delay-seconds", type=float, default=DEFAULT_REID_INTAKE_DELAY_SECONDS, help="Delay after first sighting before collecting the five-crop intake burst.")
    parser.add_argument("--reid-intake-timeout-seconds", type=float, default=DEFAULT_REID_INTAKE_TIMEOUT_SECONDS, help="After this delay, accept the best available non-sharp intake frames rather than waiting forever.")
    parser.add_argument("--reid-blur-threshold", type=float, default=DEFAULT_REID_BLUR_THRESHOLD, help="Minimum variance-of-Laplacian score for a clear ReID crop.")
    parser.add_argument("--reid-semantic-confidence", type=float, default=DEFAULT_REID_SEMANTIC_CONFIDENCE, help="YOLO confidence required before filling a semantic gallery slot.")
    parser.add_argument("--reid-semantic-retry-frames", type=int, default=DEFAULT_REID_SEMANTIC_RETRY_FRAMES, help="Frames to wait after a semantic crop fails a quality/orientation gate.")
    parser.add_argument("--reid-intake-retry-frames", type=int, default=DEFAULT_REID_INTAKE_RETRY_FRAMES, help="Initial frame backoff after a failed five-crop TransReID batch.")
    parser.add_argument("--reid-role-checkpoint", default=DEFAULT_REID_ROLE_CHECKPOINT, help="MobileNetV2 CAG/evacuee/SCDF role checkpoint.")
    parser.add_argument("--reid-role-confidence", type=float, default=DEFAULT_REID_ROLE_CONFIDENCE, help="Minimum confidence required before assigning a CAG/SCDF role.")
    parser.add_argument("--no-reid-role-classification", action="store_true", help="Disable the MobileNet role gate and treat new identities as evacuees.")
    parser.add_argument("--no-demographics", action="store_true", help="Disable background MiVOLO age/gender analysis for new evacuees.")
    parser.add_argument("--reid-device", type=str, default="cuda:0", help="Device to run ReID on (e.g., cuda:0, cuda:1, cpu)")
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







def process_camera_frame(
    context,
    conf,
    device_id,
    pose_dropout_ttl_frames=DEFAULT_POSE_DROPOUT_TTL_FRAMES,
    imgsz=640,
    half=True,
    nms_iou=DEFAULT_YOLO_NMS_IOU,
    tracker_config=DEFAULT_TRACKER_CONFIG_PATH,
):
    speed_debug_lines = []
    processing_timestamp = time.monotonic()
    if hasattr(context.cap, "read_with_metadata"):
        success, frame, captured_at, capture_sequence = context.cap.read_with_metadata()
    else:
        success, frame = context.cap.read()
        captured_at, capture_sequence = time.monotonic(), None
    context.raw_frame = frame if success else None

    if not success or frame is None:
        context.tactical_points = []
        context.tactical_observations = []
        context.annotated_frame = None
        return False

    if capture_sequence is not None and capture_sequence == context.last_capture_sequence:
        return True
    context.last_capture_sequence = capture_sequence
    context.tactical_points = []
    context.tactical_observations = []
    frame_timestamp = time.monotonic() if captured_at is None else float(captured_at)

    context.frame_index += 1

    # --- FPS tracking (EMA-smoothed so the readout doesn't jitter frame to frame) ---
    if context._last_frame_time is not None:
        frame_delta = processing_timestamp - context._last_frame_time
        if frame_delta > 1e-6:
            instantaneous_fps = 1.0 / frame_delta
            if context.fps <= 0.0:
                context.fps = instantaneous_fps
            else:
                context.fps = (1.0 - FPS_EMA_ALPHA) * context.fps + FPS_EMA_ALPHA * instantaneous_fps
    context._last_frame_time = processing_timestamp

    use_half = bool(half and str(device_id).lower() != "cpu" and torch.cuda.is_available())
    results = context.model.track(
        frame,
        classes=[0],
        conf=conf,
        iou=nms_iou,
        imgsz=imgsz,
        half=use_half,
        persist=True,
        tracker=tracker_config,
        verbose=False,
        device=device_id,
    )
    result = results[0]
    annotated_frame = frame.copy()
    standing_points = get_standing_points(
        result,
        frame,
        context.pose_estimator,
        anatomical_ratio_memory=context.anatomical_ratio_memory,
        anatomical_anchor_memory=context.anatomical_anchor_memory,
        last_foot_memory=context.last_foot_memory,
        frame_index=context.frame_index,
        pose_dropout_ttl_frames=pose_dropout_ttl_frames,
        annotated_frame=annotated_frame,
        appearance_memory=context.appearance_memory,
        camera_id=context.camera_id,
        observation_time=frame_timestamp,
        use_mediapipe_feet=context.use_mediapipe_feet,
        map_projector=lambda image_point: camera_point_to_map(image_point, context.homography),
    )

    # Preserve Ultralytics' YOLO-pose skeleton for accepted detections, but
    # never call plot() on the unfiltered result because that would paint a
    # rejected shadow before the application could hide it.
    if not context.use_mediapipe_feet:
        accepted_indices = [
            index
            for index, standing_point in enumerate(standing_points)
            if not standing_point.get("suppressed")
        ]
        if accepted_indices:
            annotated_frame = result[accepted_indices].plot(img=frame.copy())

    cv2.putText(
        annotated_frame,
        f"FPS: {context.fps:.1f}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (0, 255, 0),
        3,
    )

    for index, standing_point in enumerate(standing_points):
        if standing_point.get("suppressed"):
            continue
        point = standing_point["point"]
        speed_mps = None
        motion_status = None
        role = standing_point.get("role")
        if role == "scdf":
            label_color = (0, 165, 255)
        elif role == "cag":
            label_color = (0, 255, 255)
        elif role == "evacuee":
            label_color = (255, 150, 0)
        else:
            label_color = (255, 255, 255) if standing_point.get("identity_id") is None else (0, 0, 255)
        if index < len(result.boxes):
            box_values = result.boxes.xyxy[index].detach().cpu().numpy().astype(int)
            box_x1, box_y1, box_x2, box_y2 = box_values.tolist()
            cv2.rectangle(annotated_frame, (box_x1, box_y1), (box_x2, box_y2), label_color, 2)
        if point is not None:
            feet_x, feet_y = point
            raw_map_x, raw_map_y = camera_point_to_map((feet_x, feet_y), context.homography)
            motion_key = (
                ("identity", int(standing_point["identity_id"]))
                if standing_point.get("identity_id") is not None
                else ("track", context.camera_id, standing_point.get("track_id"))
            )
            (map_x, map_y), speed_mps, motion_status = update_map_motion(
                context.map_motion_memory,
                motion_key,
                (raw_map_x, raw_map_y),
                frame_timestamp,
            )
            context.tactical_points.append((map_x, map_y))
            context.tactical_observations.append({
                "camera_id": context.camera_id,
                "local_track_id": standing_point.get("track_id"),
                "identity_id": standing_point.get("identity_id"),
                "reid_confirmed": (
                    standing_point.get("identity_id") is not None
                    and bool(standing_point.get("reid_confirmed"))
                ),
                "point": (float(map_x), float(map_y)),
                "captured_at": frame_timestamp,
                "frame_index": context.frame_index,
                "role": standing_point.get("role"),
            })

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
            if role in ("cag", "scdf"):
                label = f"{label} {role.upper()}"
            elif role == "evacuee":
                gender = standing_point.get("gender", "Unknown")
                age = standing_point.get("age", "Unknown")
                label = f"{label} {gender}/{age} ({standing_point.get('gallery_filled', 0)}/5)"
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
            1.2,
            label_color,
            3,
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
                1.2,
                (255, 255, 255),
                3,
            )

    context.annotated_frame = annotated_frame
    return True




def _best_one_to_one_pairs(left, right, candidate_costs):
    """Maximize valid pair count, then minimize total spatial cost.

    Camera support is currently limited to two streams. The dynamic program
    avoids the order-dependent greedy failure where two shoulder-to-shoulder
    people are paired incorrectly. For unusually large crowds, it falls back
    to a deterministic global edge sort to avoid exponential state growth.
    """
    if not left or not right or not candidate_costs:
        return []
    if len(right) > len(left):
        swapped = {(right_index, left_index): cost for (left_index, right_index), cost in candidate_costs.items()}
        return [(right_index, left_index) for left_index, right_index in _best_one_to_one_pairs(right, left, swapped)]
    if len(right) > 18:
        used_left, used_right, pairs = set(), set(), []
        for (left_index, right_index), _cost in sorted(candidate_costs.items(), key=lambda item: item[1]):
            if left_index not in used_left and right_index not in used_right:
                used_left.add(left_index)
                used_right.add(right_index)
                pairs.append((left_index, right_index))
        return pairs

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def solve(left_index, used_right_mask):
        if left_index >= len(left):
            return 0, 0.0, ()
        best = solve(left_index + 1, used_right_mask)
        for right_index in range(len(right)):
            bit = 1 << right_index
            cost = candidate_costs.get((left_index, right_index))
            if cost is None or used_right_mask & bit:
                continue
            matched_count, total_cost, pairs = solve(left_index + 1, used_right_mask | bit)
            candidate = (matched_count + 1, total_cost + cost, ((left_index, right_index),) + pairs)
            if candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
                best = candidate
        return best

    return list(solve(0, 0)[2])


def fuse_camera_points(
    camera_observations,
    max_distance_cm,
    max_skew_seconds=DEFAULT_CROSS_CAMERA_MAX_SKEW_SECONDS,
    require_reid=False,
):
    """Fuse two camera observation sets with appearance as a hard veto.

    Homography and capture time only nominate a pair. When ReID is enabled,
    both observations must already resolve to the same shared master ID;
    unknown or different identities remain separate rather than being merged
    merely because two people are physically close.
    """
    camera_ids = list(camera_observations)
    normalized = {}
    for camera_id, observations in camera_observations.items():
        normalized[camera_id] = []
        for observation in observations:
            if isinstance(observation, dict):
                candidate = dict(observation)
            else:
                candidate = {
                    "camera_id": camera_id,
                    "local_track_id": None,
                    "identity_id": None,
                    "reid_confirmed": False,
                    "point": tuple(observation),
                    "captured_at": 0.0,
                }
            try:
                point = np.asarray(candidate.get("point"), dtype=float).reshape(-1)
                captured_at = float(candidate.get("captured_at", 0.0))
            except (TypeError, ValueError):
                continue
            if point.size != 2 or not np.all(np.isfinite(point)) or not np.isfinite(captured_at):
                continue
            candidate["point"] = (float(point[0]), float(point[1]))
            candidate["captured_at"] = captured_at
            normalized[camera_id].append(candidate)

    def singleton(observation):
        point = observation["point"]
        return {
            "center": (float(point[0]), float(point[1])),
            "points": [(float(point[0]), float(point[1]))],
            "sources": [observation["camera_id"]],
            "observations": [observation],
            "identity_id": observation.get("identity_id"),
        }

    if len(camera_ids) < 2:
        return [singleton(observation) for observations in normalized.values() for observation in observations]

    left = normalized[camera_ids[0]]
    right = normalized[camera_ids[1]]
    candidate_costs = {}
    for left_index, left_observation in enumerate(left):
        for right_index, right_observation in enumerate(right):
            spatial_distance = distance_cm(left_observation["point"], right_observation["point"])
            time_skew = abs(float(left_observation.get("captured_at", 0.0)) - float(right_observation.get("captured_at", 0.0)))
            if spatial_distance > float(max_distance_cm) or time_skew > float(max_skew_seconds):
                continue
            if require_reid:
                left_identity = left_observation.get("identity_id")
                right_identity = right_observation.get("identity_id")
                if (
                    not left_observation.get("reid_confirmed")
                    or not right_observation.get("reid_confirmed")
                    or left_identity is None
                    or left_identity != right_identity
                ):
                    continue
            time_tiebreak = time_skew / max(float(max_skew_seconds), 1e-9)
            candidate_costs[(left_index, right_index)] = spatial_distance + time_tiebreak * 1e-3

    pairs = _best_one_to_one_pairs(left, right, candidate_costs)
    paired_left = {left_index for left_index, _ in pairs}
    paired_right = {right_index for _, right_index in pairs}
    fused_people = []
    for left_index, right_index in pairs:
        observations = [left[left_index], right[right_index]]
        points = [tuple(observation["point"]) for observation in observations]
        identities = {observation.get("identity_id") for observation in observations}
        fused_people.append({
            "center": tuple(np.mean(np.asarray(points, dtype=float), axis=0)),
            "points": points,
            "sources": [observation["camera_id"] for observation in observations],
            "observations": observations,
            "identity_id": identities.pop() if len(identities) == 1 else None,
        })
    fused_people.extend(singleton(observation) for index, observation in enumerate(left) if index not in paired_left)
    fused_people.extend(singleton(observation) for index, observation in enumerate(right) if index not in paired_right)
    for camera_id in camera_ids[2:]:
        fused_people.extend(singleton(observation) for observation in normalized[camera_id])
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
        person_label = f"ID {person['identity_id']}" if person.get("identity_id") is not None else f"P{person_index}"
        cv2.putText(
            canvas,
            person_label,
            (pixel_x + 10, pixel_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            point_color,
            3,
        )
        cv2.putText(
            canvas,
            "+".join(person["sources"]),
            (pixel_x + 10, pixel_y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.84,
            point_color,
            2,
        )

    cv2.putText(canvas, "Combined fused map", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (40, 40, 40), 3)
    return canvas


def build_payloads(contexts, args, fused_people, combined_map=None):
    total_count = len(fused_people)
    positions = []
    zone_counts = {}

    for context in contexts:
        zone_counts[context.camera_id] = len(context.tactical_points)

    for person_index, person in enumerate(fused_people, start=1):
        x, y = person["center"]
        stable_id = person.get("identity_id")
        positions.append({
            "person_id": f"ID_{stable_id}" if stable_id is not None else f"P{person_index}",
            "master_id": stable_id,
            "sources": person["sources"],
            "source_tracks": [
                {
                    "camera_id": observation.get("camera_id"),
                    "local_track_id": observation.get("local_track_id"),
                }
                for observation in person.get("observations", ())
            ],
            "x": round(float(x), 1),
            "y": round(float(y), 1),
        })

    tactical_payload = {
        "timestamp": int(time.time()),
        "camera_id": args.camera_id,
        "run_id": args.run_id,
        "people_count": total_count,
        "positions_cm": positions,
        "map_size_cm": args.map_size_cm,
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
        "passenger_count": total_count,
        "zone_counts": zone_counts,
        "camera_online_count": tactical_payload["camera_online_count"],
    }

    return tactical_payload, metric_payload


def main():
    args = parse_args()
    if not 0.0 < args.iou <= 1.0:
        raise ValueError("--iou must be greater than 0 and at most 1.")
    tracker_config_path = Path(args.tracker_config).expanduser()
    if not tracker_config_path.is_absolute():
        tracker_config_path = Path(__file__).resolve().parent / tracker_config_path
    if not tracker_config_path.is_file():
        raise FileNotFoundError(f"ByteTrack configuration not found: {tracker_config_path}")
    args.tracker_config = str(tracker_config_path)
    contexts = build_camera_contexts(args)

    reid_extractor = None
    shared_appearance_memory = None
    if args.use_appearance_reid:
        reid_extractor = TransReIDFeatureExtractor(Path(args.reid_checkpoint), device=args.reid_device, fastreid_root=args.fastreid_root)
        if not reid_extractor.is_available():
            print("Appearance ReID will use color histograms only. IDs may change more easily after a person disappears.")
        shared_appearance_memory = AppearanceIdentityMemory(
            similarity_threshold=args.reid_similarity_threshold,
            distance_threshold=args.reid_distance_threshold,
            ttl_frames=args.reid_memory_ttl_frames,
            ema_alpha=DEFAULT_REID_EMA_ALPHA,
            reid_extractor=reid_extractor,
            db_path=None if args.no_persistent_reid_db else args.reid_db,
            intake_frames=args.reid_intake_frames,
            gallery_update_interval_frames=args.reid_gallery_update_interval_frames,
            evidence_dir=None if args.no_reid_evidence else args.reid_evidence_dir,
            intake_delay_seconds=args.reid_intake_delay_seconds,
            intake_timeout_seconds=args.reid_intake_timeout_seconds,
            blur_threshold=args.reid_blur_threshold,
            semantic_confidence_threshold=args.reid_semantic_confidence,
            semantic_retry_frames=args.reid_semantic_retry_frames,
            intake_retry_frames=args.reid_intake_retry_frames,
            role_checkpoint=args.reid_role_checkpoint,
            role_confidence_threshold=args.reid_role_confidence,
            enable_role_classification=not args.no_reid_role_classification,
            enable_demographics=not args.no_demographics,
            demographics_device=args.reid_device,
            cross_camera_fusion_distance_cm=args.fusion_distance_cm,
            cross_camera_max_skew_seconds=args.cross_camera_max_skew_seconds,
        )

    for context in contexts:
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO is not installed. Install ultralytics or disable camera processing.")
        context.model = YOLO(args.model)
        # One MediaPipe pose-landmarker instance per camera (not shared)
        # since cameras below now run concurrently in worker threads and
        # the MediaPipe Tasks API isn't documented as safe for concurrent
        # detect() calls on a single shared instance.
        context.use_mediapipe_feet = bool(args.use_mediapipe_feet)
        context.pose_estimator = create_mediapipe_pose_estimator(
            args.use_mediapipe_feet or args.use_appearance_reid,
            Path(args.mediapipe_model),
        )
        context.appearance_memory = shared_appearance_memory
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

    # Note: two independent YOLO model instances (one per camera context)
    # sharing the same physical GPU will still serialize their actual CUDA
    # kernels -- but running them from separate threads lets each camera's
    # CPU-side work (frame decode, MediaPipe fallback, drawing, ReID
    # post-processing) overlap with the *other* camera's GPU work instead of
    # blocking behind it, which is where the real wall-clock savings come
    # from with two cameras.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(contexts)), thread_name_prefix="camera") as camera_executor:
        while True:
            futures = [
                camera_executor.submit(
                    process_camera_frame,
                    context,
                    args.conf,
                    args.device,
                    args.pose_dropout_ttl_frames,
                    args.imgsz,
                    args.half,
                    args.iou,
                    args.tracker_config,
                )
                for context in contexts
            ]
            # Collect every result before calling any().  The previous
            # generator expression short-circuited after the first active
            # camera and allowed the other camera's context to be read while
            # its worker was still updating it.
            statuses = [future.result() for future in futures]
            active_camera = any(statuses)

            if not active_camera:
                print("No camera frames available. Exiting.")
                break

            camera_observations = {context.camera_id: context.tactical_observations for context in contexts}
            map_size_cm = max(context.map_size_cm for context in contexts)
            fused_people = fuse_camera_points(
                camera_observations,
                args.fusion_distance_cm,
                max_skew_seconds=args.cross_camera_max_skew_seconds,
                require_reid=args.use_appearance_reid,
            )
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
                        mqtt_client.publish(args.mqtt_topic, json.dumps(tactical_payload), qos=1)
                        mqtt_client.publish(args.mqtt_metrics_topic, json.dumps(metric_payload), qos=1)
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

    for context in contexts:
        if context.pose_estimator is not None:
            context.pose_estimator.close()

    if shared_appearance_memory is not None:
        shared_appearance_memory.close(drain=True)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()






