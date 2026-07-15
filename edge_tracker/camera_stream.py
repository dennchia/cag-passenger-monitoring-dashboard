import threading
import time

import cv2

from constants import DISPLAY_SIZE


class LiveCamera:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.lock = threading.Lock()
        self.running = False
        self.ret = False
        self.frame = None
        self.captured_at = None
        self.sequence = 0

        if self.cap.isOpened():
            self.ret, self.frame = self.cap.read()
            if self.ret and self.frame is not None:
                self.captured_at = time.monotonic()
                self.sequence = 1
            self.running = True
            self.thread = threading.Thread(target=self.update, daemon=True) #make it a daemon thread will automatically exit and run independently
            self.thread.start()

    def is_opened(self):
        return self.cap.isOpened()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
                if ret and frame is not None:
                    self.captured_at = time.monotonic()
                    self.sequence += 1

            if not ret:
                self.running = False

    def read(self):
        with self.lock:
            if self.frame is None:
                return self.ret, None

            return self.ret, self.frame.copy()

    def read_with_metadata(self):
        with self.lock:
            if self.frame is None:
                return self.ret, None, self.captured_at, self.sequence
            return self.ret, self.frame.copy(), self.captured_at, self.sequence

    def release(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
        self.cap.release()

def resize_to_fit(frame, max_size=DISPLAY_SIZE):
    max_width, max_height = max_size
    height, width = frame.shape[:2] # get the original dimensions of the frame
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return frame.copy(), scale

    resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
    return resized, scale

class CameraContext:
    def __init__(self, camera_id, source, matrix_path, map_size_cm):
        self.camera_id = camera_id
        self.source = source
        self.matrix_path = matrix_path
        self.map_size_cm = map_size_cm
        self.missing_corner = None
        self.homography = None
        self.cap = None
        self.annotated_frame = None
        self.tactical_points = []
        self.tactical_observations = []
        self.raw_frame = None
        self.anatomical_ratio_memory = {}
        self.anatomical_anchor_memory = {}
        self.last_foot_memory = {}
        self.map_motion_memory = {}
        self.frame_index = 0
        self.last_capture_sequence = None
        self.appearance_memory = None
        self.model = None
        self.fps = 0.0
        self._last_frame_time = None
        # Each camera gets its own MediaPipe pose-landmarker instance rather
        # than sharing one globally -- cameras now run concurrently in worker
        # threads, and the MediaPipe Tasks API is not documented as safe for
        # concurrent detect() calls on a single shared instance.
        self.pose_estimator = None
        self.use_mediapipe_feet = False

