import hashlib
import copy
import json
import os
import pickle
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None
try:
    import timm
except ImportError:
    timm = None

from constants import (
    DEFAULT_REID_BLUR_THRESHOLD,
    DEFAULT_REID_CROP_BOTTOM_PADDING,
    DEFAULT_REID_CROP_SIDE_PADDING,
    DEFAULT_REID_CROP_TOP_PADDING,
    DEFAULT_REID_DISTANCE_THRESHOLD,
    DEFAULT_REID_EMA_ALPHA,
    DEFAULT_REID_INTAKE_DELAY_SECONDS,
    DEFAULT_REID_INTAKE_RETRY_FRAMES,
    DEFAULT_REID_INTAKE_TIMEOUT_SECONDS,
    DEFAULT_REID_MAX_RETRY_FRAMES,
    DEFAULT_REID_MEMORY_TTL_FRAMES,
    DEFAULT_REID_QUEUE_SIZE,
    DEFAULT_REID_ROLE_CHECKPOINT,
    DEFAULT_REID_ROLE_CONFIDENCE,
    DEFAULT_REID_SEMANTIC_CONFIDENCE,
    DEFAULT_REID_SEMANTIC_COOLDOWN_FRAMES,
    DEFAULT_REID_SEMANTIC_RETRY_FRAMES,
    DEFAULT_REID_SHADOW_CENTER_DISTANCE_RATIO,
    DEFAULT_REID_SHADOW_CONTAINMENT_THRESHOLD,
    DEFAULT_REID_SHADOW_IOU_THRESHOLD,
    DEFAULT_REID_SHADOW_PROBATION_FRAMES,
    DEFAULT_REID_SHADOW_SEPARATION_FRAMES,
    DEFAULT_REID_SIMILARITY_THRESHOLD,
    REID_GALLERY_SLOTS,
    REID_SEMANTIC_SLOTS,
)


def _sha256_file(path):
    path = Path(path)
    if not path.exists() or not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clamp_box_to_frame(box, frame):
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    x1 = max(0, min(frame_width - 1, int(x1)))
    y1 = max(0, min(frame_height - 1, int(y1)))
    x2 = max(0, min(frame_width, int(x2)))
    y2 = max(0, min(frame_height, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def crop_person(
    frame,
    box,
    side_padding=DEFAULT_REID_CROP_SIDE_PADDING,
    top_padding=DEFAULT_REID_CROP_TOP_PADDING,
    bottom_padding=DEFAULT_REID_CROP_BOTTOM_PADDING,
):
    x1, y1, x2, y2 = map(float, box)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    padded_box = (
        x1 - width * float(side_padding),
        y1 - height * float(top_padding),
        x2 + width * float(side_padding),
        y2 + height * float(bottom_padding),
    )
    clamped = clamp_box_to_frame(padded_box, frame)
    if clamped is None:
        return None
    x1, y1, x2, y2 = clamped
    return frame[y1:y2, x1:x2]


def image_sharpness(crop):
    if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def compute_color_reid_feature(crop):
    if crop is None or crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 10:
        return None

    resized = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    upper = hsv[8:70, :]
    lower = hsv[58:124, :]
    hist_upper = cv2.calcHist([upper], [0, 1], None, [24, 16], [0, 180, 0, 256]).flatten()
    hist_lower = cv2.calcHist([lower], [0, 1], None, [24, 16], [0, 180, 0, 256]).flatten()
    feature = np.concatenate([hist_upper * 0.65, hist_lower * 0.35]).astype(np.float32)
    norm = float(np.linalg.norm(feature))
    if norm <= 1e-6:
        return None
    return feature / norm

def cosine_similarity(feature_a, feature_b):
    if feature_a is None or feature_b is None:
        return 0.0
    return float(np.dot(feature_a, feature_b))

def remap_state_dict_for_timm(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("base."):
            new_key = key.replace("base.", "", 1)
        elif key.startswith("b1."):
            new_key = key.replace("b1.", "blocks.1.", 1)
        elif key.startswith("b2."):
            new_key = key.replace("b2.", "blocks.2.", 1)
        else:
            new_key = key
        remapped[new_key] = value
    return remapped

def unwrap_torch_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint

def remap_state_dict_for_fastreid(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    remapped = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "", 1)
        if new_key.startswith("base."):
            new_key = new_key.replace("base.", "backbone.", 1)
        remapped[new_key] = value
    return remapped

class TransReIDFeatureExtractor:
    def __init__(self, checkpoint_path, device="cuda", fastreid_root="fast-reid"):
        self.model = None
        self.backend = None
        self.checkpoint_path = Path(checkpoint_path)
        self._checkpoint_sha256 = None
        self._config_sha256 = None
        self.fastreid_root = Path(fastreid_root) if fastreid_root else None
        # Cameras now run their frame pipelines concurrently in worker
        # threads (see main_tracker.py). This single ReID model instance is
        # shared across all of them, so we serialize the actual forward
        # passes to avoid any cross-thread CUDA/state issues. Everything
        # else (crop resizing, numpy post-processing) still happens outside
        # the lock and can overlap freely.
        self._lock = threading.Lock()
        if torch is None:
            print("Warning: torch is not available, TransReID feature extractor disabled.")
            return

        self.device = torch.device(device)

        print(f"[Hardware Check] TransReID is running on: {self.device.type.upper()}")
        if self._load_fastreid_model():
            return

        if timm is None:
            print("Warning: FastReID/timm is not available, TransReID feature extractor disabled.")
            return

        self._load_timm_model()

    def is_available(self):
        return self.model is not None

    def feature_space_id(self, dimension):
        """Fingerprint the exact model and preprocessing feature space."""
        if self._checkpoint_sha256 is None:
            self._checkpoint_sha256 = _sha256_file(self.checkpoint_path)
        if self.backend == "fastreid":
            config_path = self.fastreid_root / "configs" / "Market1501" / "bagtricks_vit.yml"
            if self._config_sha256 is None:
                self._config_sha256 = _sha256_file(config_path)
            model_spec = "bagtricks_vit_stride12x12"
            preprocess = "resize128x256-inter_area-bgr2rgb-chw-f32-range0_255"
        else:
            model_spec = "vit_base_patch16_224-img240x224"
            preprocess = "resize224x240-inter_area-bgr2rgb-chw-f32-div255-default_cfg_norm"
        metadata = {
            "kind": "transreid",
            "backend": self.backend,
            "checkpoint_sha256": self._checkpoint_sha256,
            "config_sha256": self._config_sha256,
            "model_spec": model_spec,
            "preprocess": preprocess,
            "preprocess_revision": 1,
            "dimension": int(dimension),
        }
        canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "fs1:" + hashlib.sha256(canonical).hexdigest()

    def _load_fastreid_model(self):
        if self.fastreid_root is None or not self.fastreid_root.exists():
            print(f"FastReID folder not found: {self.fastreid_root}.")
            return False

        root_text = str(self.fastreid_root.resolve())
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        config_path = self.fastreid_root / "configs" / "Market1501" / "bagtricks_vit.yml"
        if not config_path.exists():
            print(f"FastReID config not found: {config_path}.")
            return False

        try:
            from fastreid.config import get_cfg
            from fastreid.modeling.meta_arch import build_model

            cfg = get_cfg()
            cfg.merge_from_file(str(config_path))
            cfg.MODEL.BACKBONE.STRIDE_SIZE = [12, 12]
            cfg.MODEL.BACKBONE.PRETRAIN = False
            cfg.MODEL.BACKBONE.PRETRAIN_PATH = ""
            cfg.MODEL.DEVICE = str(self.device)
            cfg.MODEL.WEIGHTS = str(self.checkpoint_path)

            model = build_model(cfg)
            model.eval()
            model.to(self.device)

            if not self.checkpoint_path.exists():
                print(f"TransReID checkpoint not found: {self.checkpoint_path}")
                return False

            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            state_dict = remap_state_dict_for_fastreid(unwrap_torch_checkpoint(checkpoint))
            incompatible = model.load_state_dict(state_dict, strict=False)
            missing = getattr(incompatible, "missing_keys", incompatible[0] if incompatible else [])
            unexpected = getattr(incompatible, "unexpected_keys", incompatible[1] if incompatible else [])
            print(f"TransReID FastReID backend loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

            self.model = model
            self.backend = "fastreid"
            return True
        except Exception as exc:
            print(f"Unable to load FastReID TransReID backend: {exc}")
            self.model = None
            return False

    def _load_timm_model(self):
        try:
            self.model = timm.create_model(
                "vit_base_patch16_224",
                pretrained=False,
                num_classes=0,
                img_size=(240, 224),
            )
            self.model.to(self.device)
            self.model.eval()
            if self.checkpoint_path.exists():
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                state_dict = unwrap_torch_checkpoint(checkpoint)
                if isinstance(state_dict, dict):
                    state_dict = remap_state_dict_for_timm(state_dict)
                    incompatible = self.model.load_state_dict(state_dict, strict=False)
                    if incompatible.missing_keys:
                        print(f"TransReID missing keys: {incompatible.missing_keys[:10]}")
                    if incompatible.unexpected_keys:
                        print(f"TransReID unexpected keys: {incompatible.unexpected_keys[:10]}")
            else:
                print(f"TransReID checkpoint not found: {self.checkpoint_path}")
            self.backend = "timm"
        except Exception as exc:
            print(f"Unable to load TransReID checkpoint: {exc}")
            self.model = None
            self.backend = None

    def extract(self, crop):
        if self.model is None or crop is None:
            return None

        try:
            if self.backend == "fastreid":
                resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_AREA)
                rgb = resized[:, :, ::-1]
                tensor = torch.as_tensor(rgb.transpose(2, 0, 1).astype("float32"))
                tensor = tensor.unsqueeze(0).to(self.device)
                with self._lock, torch.no_grad():
                    features = self.model(tensor)
                feature = features.detach().cpu().numpy().ravel().astype(np.float32)
                norm = float(np.linalg.norm(feature))
                if norm <= 1e-6:
                    return None
                return feature / norm

            resized = cv2.resize(crop, (224, 240), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
            tensor = tensor.unsqueeze(0).to(self.device)
            if hasattr(self.model, 'default_cfg') and self.model.default_cfg is not None:
                mean = torch.tensor(self.model.default_cfg.get('mean', (0.5, 0.5, 0.5)), device=self.device).view(3, 1, 1)
                std = torch.tensor(self.model.default_cfg.get('std', (0.5, 0.5, 0.5)), device=self.device).view(3, 1, 1)
                tensor = (tensor - mean) / std
            with self._lock, torch.no_grad():
                features = self.model(tensor)
            feature = features.detach().cpu().numpy().ravel().astype(np.float32)
            norm = float(np.linalg.norm(feature))
            if norm <= 1e-6:
                return None
            return feature / norm
        except Exception:
            return None

    def extract_many(self, crops):
        crops = [crop for crop in crops if crop is not None and crop.size > 0]
        if self.model is None or not crops:
            return []

        if self.backend != "fastreid":
            return [feature for feature in (self.extract(crop) for crop in crops) if feature is not None]

        try:
            tensors = []
            for crop in crops:
                resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_AREA)
                rgb = resized[:, :, ::-1]
                tensors.append(torch.as_tensor(rgb.transpose(2, 0, 1).astype("float32")))

            batched_tensor = torch.stack(tensors).to(self.device)
            with self._lock, torch.no_grad():
                features = self.model(batched_tensor)

            normalized_features = []
            for feature in features.detach().cpu().numpy().astype(np.float32):
                norm = float(np.linalg.norm(feature))
                if norm > 1e-6:
                    normalized_features.append(feature / norm)
            return normalized_features
        except Exception:
            return [feature for feature in (self.extract(crop) for crop in crops) if feature is not None]

    def extract_many_aligned(self, crops):
        """Like extract_many, but the returned list is always the same
        length as `crops`, with None standing in for any crop that failed to
        produce a feature. extract_many() silently drops failures, which
        breaks positional alignment with track_ids -- unsafe for per-frame
        batching where a feature must be matched back to a specific person.
        """
        if self.model is None or not crops:
            return [None] * len(crops)

        if self.backend != "fastreid":
            return [self.extract(crop) if crop is not None and crop.size > 0 else None for crop in crops]

        valid_indices = [i for i, crop in enumerate(crops) if crop is not None and crop.size > 0]
        if not valid_indices:
            return [None] * len(crops)

        try:
            tensors = []
            for i in valid_indices:
                resized = cv2.resize(crops[i], (128, 256), interpolation=cv2.INTER_AREA)
                rgb = resized[:, :, ::-1]
                tensors.append(torch.as_tensor(rgb.transpose(2, 0, 1).astype("float32")))

            batched_tensor = torch.stack(tensors).to(self.device)
            with self._lock, torch.no_grad():
                features = self.model(batched_tensor)

            results = [None] * len(crops)
            for local_index, original_index in enumerate(valid_indices):
                feature = features[local_index].detach().cpu().numpy().astype(np.float32)
                norm = float(np.linalg.norm(feature))
                if norm > 1e-6:
                    results[original_index] = feature / norm
            return results
        except Exception:
            return [self.extract(crop) if crop is not None and crop.size > 0 else None for crop in crops]
class EvacuationRoleClassifier:
    """CPU-only MobileNetV2 gate used by the v7 intake path."""

    CLASS_NAMES = ("cag", "evacuee", "scdf")

    def __init__(self, checkpoint_path=DEFAULT_REID_ROLE_CHECKPOINT):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model = None
        self.transform = None
        if torch is None or nn is None or self.checkpoint_path is None or not self.checkpoint_path.exists():
            return

        try:
            from torchvision import transforms
            from torchvision.models import mobilenet_v2

            model = mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(self.CLASS_NAMES))
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            state_dict = unwrap_torch_checkpoint(checkpoint)
            model.load_state_dict(state_dict)
            model.eval()
            self.model = model
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            print(f"Role classifier loaded from {self.checkpoint_path} on CPU")
        except Exception as exc:
            print(f"Unable to load evacuation role classifier: {exc}")
            self.model = None
            self.transform = None

    def predict(self, crop):
        if self.model is None or self.transform is None or crop is None or crop.size == 0:
            return "evacuee", 0.0

        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0)
            with torch.no_grad():
                probabilities = torch.softmax(self.model(tensor)[0], dim=0)
            confidence, class_index = torch.max(probabilities, dim=0)
            return self.CLASS_NAMES[int(class_index.item())], float(confidence.item())
        except Exception as exc:
            print(f"Role classification failed: {exc}")
            return "evacuee", 0.0


class AppearanceIdentityMemory:
    """Shared, asynchronous five-slot TransReID identity coordinator.

    Local tracker IDs are namespaced by camera. A new local track contributes
    five quality-controlled crops, which are processed as one batch by a
    background worker. Mapped tracks are dictionary lookups except for a
    bounded single-crop inference used to fill a genuinely missing semantic
    orientation slot.
    """

    SCHEMA_VERSION = 3

    def __init__(
        self,
        similarity_threshold=DEFAULT_REID_SIMILARITY_THRESHOLD,
        ttl_frames=DEFAULT_REID_MEMORY_TTL_FRAMES,
        ema_alpha=DEFAULT_REID_EMA_ALPHA,
        reid_extractor=None,
        verbose=False,
        distance_threshold=None,
        morph_threshold=0.08,
        max_gallery_size=5,
        db_path=None,
        persistence_store=None,
        intake_frames=5,
        gallery_update_interval_frames=DEFAULT_REID_SEMANTIC_COOLDOWN_FRAMES,
        evidence_dir=None,
        intake_delay_seconds=DEFAULT_REID_INTAKE_DELAY_SECONDS,
        intake_timeout_seconds=DEFAULT_REID_INTAKE_TIMEOUT_SECONDS,
        blur_threshold=DEFAULT_REID_BLUR_THRESHOLD,
        semantic_confidence_threshold=DEFAULT_REID_SEMANTIC_CONFIDENCE,
        semantic_cooldown_frames=None,
        semantic_retry_frames=DEFAULT_REID_SEMANTIC_RETRY_FRAMES,
        intake_retry_frames=DEFAULT_REID_INTAKE_RETRY_FRAMES,
        max_retry_frames=DEFAULT_REID_MAX_RETRY_FRAMES,
        queue_size=DEFAULT_REID_QUEUE_SIZE,
        role_checkpoint=DEFAULT_REID_ROLE_CHECKPOINT,
        role_confidence_threshold=DEFAULT_REID_ROLE_CONFIDENCE,
        enable_role_classification=True,
        enable_demographics=True,
        demographics_device=None,
        cross_camera_fusion_distance_cm=None,
        cross_camera_max_skew_seconds=0.35,
        shadow_iou_threshold=DEFAULT_REID_SHADOW_IOU_THRESHOLD,
        shadow_containment_threshold=DEFAULT_REID_SHADOW_CONTAINMENT_THRESHOLD,
        shadow_center_distance_ratio=DEFAULT_REID_SHADOW_CENTER_DISTANCE_RATIO,
        shadow_probation_frames=DEFAULT_REID_SHADOW_PROBATION_FRAMES,
        shadow_separation_frames=DEFAULT_REID_SHADOW_SEPARATION_FRAMES,
        start_worker=True,
    ):
        del morph_threshold, max_gallery_size  # incompatible with fixed named slots
        self.similarity_threshold = float(similarity_threshold)
        self.distance_threshold = (
            max(0.0, 1.0 - self.similarity_threshold)
            if distance_threshold is None
            else float(distance_threshold)
        )
        if distance_threshold is None and similarity_threshold == DEFAULT_REID_SIMILARITY_THRESHOLD:
            self.distance_threshold = DEFAULT_REID_DISTANCE_THRESHOLD
        self.ttl_frames = max(1, int(ttl_frames))  # local bindings only; masters never expire
        self.ema_alpha = float(ema_alpha)
        self.reid_extractor = reid_extractor
        self.verbose = bool(verbose)
        self.db_path = Path(db_path) if db_path else None
        self.persistence_store = persistence_store
        self.evidence_dir = Path(evidence_dir) if evidence_dir else None
        self.intake_frames = max(1, int(intake_frames))
        self.intake_delay_seconds = max(0.0, float(intake_delay_seconds))
        self.intake_timeout_seconds = max(self.intake_delay_seconds, float(intake_timeout_seconds))
        self.blur_threshold = max(0.0, float(blur_threshold))
        self.semantic_confidence_threshold = float(semantic_confidence_threshold)
        self.semantic_cooldown_frames = max(
            1,
            int(
                gallery_update_interval_frames
                if semantic_cooldown_frames is None
                else semantic_cooldown_frames
            ),
        )
        self.semantic_retry_frames = max(1, int(semantic_retry_frames))
        self.intake_retry_frames = max(1, int(intake_retry_frames))
        self.max_retry_frames = max(self.intake_retry_frames, int(max_retry_frames))
        self.role_checkpoint = Path(role_checkpoint) if role_checkpoint else None
        self.role_confidence_threshold = min(1.0, max(0.0, float(role_confidence_threshold)))
        self.enable_role_classification = bool(enable_role_classification)
        self.enable_demographics = bool(enable_demographics)
        self.demographics_device = demographics_device
        self.cross_camera_fusion_distance_cm = (
            None
            if cross_camera_fusion_distance_cm is None
            else max(0.0, float(cross_camera_fusion_distance_cm))
        )
        self.cross_camera_max_skew_seconds = max(0.0, float(cross_camera_max_skew_seconds))
        self.shadow_iou_threshold = min(1.0, max(0.0, float(shadow_iou_threshold)))
        self.shadow_containment_threshold = min(
            1.0,
            max(0.0, float(shadow_containment_threshold)),
        )
        self.shadow_center_distance_ratio = max(0.0, float(shadow_center_distance_ratio))
        self.shadow_probation_frames = max(0, int(shadow_probation_frames))
        self.shadow_separation_frames = max(1, int(shadow_separation_frames))

        self.identities = {}
        self.next_identity_id = 1
        self.track_to_identity = {}
        self.track_last_seen = {}
        self.track_results = {}
        self.track_binding_metadata = {}
        self.track_generations = {}
        self.pending_intake = {}
        self.visible_track_keys_by_camera = {}
        self.track_boxes = {}
        self.shadow_tracks = {}
        self.pending_semantic_slots = set()
        self.next_semantic_attempt_frame = {}
        self.semantic_probe_quality = {}
        self.recent_master_observations = {}

        self._lock = threading.RLock()
        self._persistence_lock = threading.Lock()
        self._task_queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._demographics_queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop_token = object()
        self._worker = None
        self._demographics_worker = None
        self._role_classifier = None
        self._demographics_engine = None
        self._closed = False

        self.load_database()
        if start_worker:
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="reid-analyst",
                daemon=True,
            )
            self._worker.start()

    @staticmethod
    def _empty_gallery():
        return {slot_name: None for slot_name in REID_GALLERY_SLOTS}

    @staticmethod
    def _track_key(track_id, camera_id=None):
        local_id = int(track_id)
        return local_id if camera_id is None else (str(camera_id), local_id)

    @staticmethod
    def _camera_from_key(track_key):
        return track_key[0] if isinstance(track_key, tuple) else None

    @staticmethod
    def _normalized_box(box):
        if box is None:
            return None
        try:
            values = np.asarray(box, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if values.size < 4 or not np.all(np.isfinite(values[:4])):
            return None
        x1, y1, x2, y2 = map(float, values[:4])
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _shadow_overlap_score(self, candidate_box, canonical_box):
        """Return duplicate-likeness when two boxes cover the same person.

        Geometry is deliberately only a nomination gate. A surviving
        replacement must still pass the normal five-crop appearance check
        before it can inherit the canonical track's master ID.
        """

        candidate = self._normalized_box(candidate_box)
        canonical = self._normalized_box(canonical_box)
        if candidate is None or canonical is None:
            return None

        ax1, ay1, ax2, ay2 = candidate
        bx1, by1, bx2, by2 = canonical
        intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
        intersection = intersection_width * intersection_height
        if intersection <= 0.0:
            return None

        candidate_area = (ax2 - ax1) * (ay2 - ay1)
        canonical_area = (bx2 - bx1) * (by2 - by1)
        union = candidate_area + canonical_area - intersection
        iou = intersection / union if union > 0.0 else 0.0
        containment = intersection / min(candidate_area, canonical_area)

        candidate_center = np.asarray(((ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5))
        canonical_center = np.asarray(((bx1 + bx2) * 0.5, (by1 + by2) * 0.5))
        canonical_diagonal = float(np.hypot(bx2 - bx1, by2 - by1))
        center_ratio = (
            float(np.linalg.norm(candidate_center - canonical_center)) / canonical_diagonal
            if canonical_diagonal > 1e-6
            else float("inf")
        )
        if center_ratio > self.shadow_center_distance_ratio:
            return None
        if iou < self.shadow_iou_threshold and containment < self.shadow_containment_threshold:
            return None
        return max(iou, containment), iou, containment, center_ratio

    @staticmethod
    def _normalize_feature(feature):
        if feature is None:
            return None
        array = np.asarray(feature, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(array.astype(np.float64)))
        if norm <= 1e-6:
            return None
        return (array.astype(np.float64) / norm).astype(np.float32)

    @staticmethod
    def _crop_digest(crop):
        if crop is None or crop.size == 0:
            return None
        digest = hashlib.sha256()
        digest.update(str(crop.shape).encode("ascii"))
        digest.update(str(crop.dtype).encode("ascii"))
        digest.update(crop.tobytes())
        return digest.hexdigest()

    @staticmethod
    def _quality_score(sample):
        return float(sample.get("sharpness", 0.0)) * max(1.0, float(sample.get("area", 0.0)) ** 0.5)

    def _new_record(self, role="evacuee", role_confidence=0.0):
        demographics_status = "Pending" if self.enable_demographics else "Disabled"
        return {
            "role": role,
            "role_confidence": float(role_confidence),
            "age": demographics_status if role == "evacuee" else "N/A",
            "gender": demographics_status if role == "evacuee" else "N/A",
            "gallery": self._empty_gallery(),
            "hits": 0,
            "last_seen_monotonic": time.monotonic(),
        }

    def load_database(self):
        if self.persistence_store is not None:
            try:
                payload = self.persistence_store.load_payload()
            except Exception as exc:
                print(f"Unable to load ReID identities from backend: {exc}. Starting with an empty gallery.")
                return
            evidence_enabled = False
            source_label = "FastAPI/SQLite backend"
        else:
            if self.db_path is None or not self.db_path.exists():
                return
            try:
                with self.db_path.open("rb") as handle:
                    payload = pickle.load(handle)
            except Exception as exc:
                raise RuntimeError(f"Unable to load ReID database {self.db_path}: {exc}") from exc
            evidence_enabled = bool(payload.get("evidence_enabled", False)) if isinstance(payload, dict) else False
            source_label = str(self.db_path)

        if not isinstance(payload, dict) or payload.get("schema_version") != self.SCHEMA_VERSION:
            raise RuntimeError(
                f"ReID database {self.db_path} is not the fresh five-slot schema. "
                "Delete or move the old database before starting this version."
            )
        loaded_identities = payload.get("identities")
        if not isinstance(loaded_identities, dict):
            raise RuntimeError(f"ReID database {self.db_path} has no identities dictionary.")

        identities = {}
        for raw_identity_id, raw_record in loaded_identities.items():
            if not isinstance(raw_record, dict):
                continue
            gallery = raw_record.get("gallery")
            if not isinstance(gallery, dict) or set(gallery) != set(REID_GALLERY_SLOTS):
                raise RuntimeError(
                    f"ReID database {self.db_path} contains an invalid five-slot gallery."
                )
            if gallery.get("baseline") is None:
                raise RuntimeError(f"Identity {raw_identity_id} has no baseline gallery slot.")
            normalized_gallery = self._empty_gallery()
            baseline_feature_space = None
            for slot_name in REID_GALLERY_SLOTS:
                slot = gallery.get(slot_name)
                if slot is None:
                    continue
                if not isinstance(slot, dict):
                    raise RuntimeError(f"Invalid {slot_name} slot for identity {raw_identity_id}.")
                feature = self._normalize_feature(slot.get("feature"))
                if feature is None:
                    raise RuntimeError(f"Invalid feature in {slot_name} slot for identity {raw_identity_id}.")
                feature_space_id = slot.get("feature_space_id")
                if not isinstance(feature_space_id, str) or not feature_space_id:
                    raise RuntimeError(
                        f"Missing feature-space ID in {slot_name} slot for identity {raw_identity_id}."
                    )
                if int(slot.get("feature_dimension", -1)) != int(feature.size):
                    raise RuntimeError(
                        f"Feature dimension mismatch in {slot_name} slot for identity {raw_identity_id}."
                    )
                if baseline_feature_space is None:
                    baseline_feature_space = feature_space_id
                elif feature_space_id != baseline_feature_space:
                    raise RuntimeError(f"Mixed feature spaces for identity {raw_identity_id}.")
                if evidence_enabled:
                    image_path = slot.get("image_path")
                    saved_crop = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if image_path else None
                    if saved_crop is None or self._crop_digest(saved_crop) != slot.get("digest"):
                        raise RuntimeError(
                            f"Missing or corrupt evidence for {slot_name} slot of identity {raw_identity_id}."
                        )
                normalized_slot = dict(slot)
                normalized_slot["feature"] = feature
                normalized_gallery[slot_name] = normalized_slot

            identity_id = int(raw_identity_id)
            record = dict(raw_record)
            record["gallery"] = normalized_gallery
            identities[identity_id] = record

        with self._lock:
            self.identities = identities
            self.next_identity_id = max(self.identities.keys(), default=0) + 1
        print(f"Loaded {len(identities)} five-slot ReID identities from {source_label}")

    def save_database(self, identity_id=None):
        if self.persistence_store is not None:
            if identity_id is None:
                return
            with self._lock:
                record = self.identities.get(int(identity_id))
                snapshot = copy.deepcopy(record) if record is not None else None
            if snapshot is None:
                return
            try:
                self.persistence_store.save_identity(int(identity_id), snapshot)
            except Exception as exc:
                print(f"Unable to save ReID identity {identity_id} to backend: {exc}")
            return
        if self.db_path is None:
            return
        with self._lock:
            payload = {
                "schema_version": self.SCHEMA_VERSION,
                "evidence_enabled": self.evidence_dir is not None,
                "identities": self.identities,
            }
            serialized = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        with self._persistence_lock:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.db_path.with_name(f"{self.db_path.name}.tmp")
            try:
                with temporary_path.open("wb") as handle:
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_path, self.db_path)
            except Exception as exc:
                print(f"Unable to save ReID database {self.db_path}: {exc}")
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _save_raw_evidence(self, identity_id, slot_name, crop, frame_index, camera_id):
        if self.evidence_dir is None or crop is None or crop.size == 0:
            return None
        try:
            master_dir = self.evidence_dir / f"Master_{int(identity_id):04d}"
            master_dir.mkdir(parents=True, exist_ok=True)
            camera_label = "camera" if camera_id is None else str(camera_id).replace("/", "_").replace("\\", "_")
            output_path = master_dir / f"Slot_{slot_name}_{camera_label}_frame_{int(frame_index)}.png"
            temporary_path = master_dir / (
                f".{output_path.stem}.{threading.get_ident()}.tmp.png"
            )
            if not cv2.imwrite(str(temporary_path), crop):
                raise OSError("cv2.imwrite returned False")
            saved_crop = cv2.imread(str(temporary_path), cv2.IMREAD_COLOR)
            if saved_crop is None or self._crop_digest(saved_crop) != self._crop_digest(crop):
                raise OSError("lossless evidence verification failed")
            os.replace(temporary_path, output_path)
            return str(output_path)
        except Exception as exc:
            print(f"Unable to save {slot_name} ReID evidence for ID {identity_id}: {exc}")
            try:
                temporary_path.unlink(missing_ok=True)
            except (OSError, UnboundLocalError):
                pass
            return None

    def _make_slot(self, identity_id, slot_name, feature, sample, feature_source, feature_space_id):
        crop = sample["crop"]
        image_path = self._save_raw_evidence(
            identity_id,
            slot_name,
            crop,
            sample["frame_index"],
            sample.get("camera_id"),
        )
        if self.evidence_dir is not None and image_path is None:
            raise RuntimeError(
                f"Refusing to store {slot_name} feature for ID {identity_id} without its evidence image."
            )
        normalized_feature = self._normalize_feature(feature)
        if normalized_feature is None:
            raise RuntimeError(f"Refusing to store an invalid {slot_name} feature for ID {identity_id}.")
        return {
            "feature": normalized_feature,
            "feature_source": feature_source,
            "feature_space_id": feature_space_id,
            "feature_dimension": int(normalized_feature.size),
            "image_path": image_path,
            "digest": self._crop_digest(crop),
            "captured_frame": int(sample["frame_index"]),
            "captured_at": float(sample.get("observed_at", time.monotonic())),
            "camera_id": sample.get("camera_id"),
            "sharpness": float(sample.get("sharpness", 0.0)),
            "detection_confidence": sample.get("detection_confidence"),
        }

    def _feature_space_id(self, feature_source, feature):
        dimension = int(np.asarray(feature).size)
        if feature_source == "color_histogram":
            contract = f"color-histogram-v1-resize64x128-hsv-bins24x16-regions-weighted-dim{dimension}"
            return "fs1:" + hashlib.sha256(contract.encode("utf-8")).hexdigest()
        provider = getattr(self.reid_extractor, "feature_space_id", None)
        if callable(provider):
            return str(provider(dimension))
        explicit = getattr(self.reid_extractor, "feature_space_id", None)
        if isinstance(explicit, str) and explicit:
            return explicit
        extractor_type = type(self.reid_extractor)
        contract = (
            f"transreid-adapter-v1:{extractor_type.__module__}."
            f"{extractor_type.__qualname__}:dim{dimension}"
        )
        return "fs1:" + hashlib.sha256(contract.encode("utf-8")).hexdigest()

    def _extract_aligned_features(self, crops):
        if not crops:
            return [], "none", None

        features = None
        source = "transreid"
        if self.reid_extractor is not None and hasattr(self.reid_extractor, "extract_many_aligned"):
            features = self.reid_extractor.extract_many_aligned(crops)
        elif self.reid_extractor is not None and hasattr(self.reid_extractor, "extract_many"):
            extracted = self.reid_extractor.extract_many(crops)
            if len(extracted) == len(crops):
                features = extracted

        extractor_available = self.reid_extractor is not None
        availability_check = getattr(self.reid_extractor, "is_available", None)
        if callable(availability_check):
            extractor_available = bool(availability_check())
        if features is None or not any(self._normalize_feature(feature) is not None for feature in features):
            if extractor_available:
                raise RuntimeError("TransReID extraction failed; refusing to create a fallback identity.")
            source = "color_histogram"
            features = [compute_color_reid_feature(crop) for crop in crops]
        normalized = [self._normalize_feature(feature) for feature in features]
        valid = [feature for feature in normalized if feature is not None]
        dimensions = {int(feature.size) for feature in valid}
        if len(dimensions) > 1:
            raise RuntimeError("Extractor returned inconsistent feature dimensions.")
        feature_space_id = self._feature_space_id(source, valid[0]) if valid else None
        return normalized, source, feature_space_id

    def _physical_match_allowed_locked(self, identity_id, camera_id, map_point, observed_at):
        if camera_id is None or self.cross_camera_fusion_distance_cm is None:
            return True
        if observed_at is None or not np.isfinite(float(observed_at)):
            return False
        normalized_point = None
        if map_point is not None:
            point_array = np.asarray(map_point, dtype=float).reshape(-1)
            if point_array.size != 2 or not np.all(np.isfinite(point_array)):
                return False
            normalized_point = point_array
        observations = self.recent_master_observations.get(identity_id, {})
        for other_camera, observation in observations.items():
            if other_camera == str(camera_id):
                continue
            time_skew = abs(float(observed_at) - float(observation["observed_at"]))
            if time_skew > self.cross_camera_max_skew_seconds:
                continue
            other_point = observation.get("map_point")
            if normalized_point is None or other_point is None:
                continue
            if float(np.linalg.norm(normalized_point - np.asarray(other_point, dtype=float))) > self.cross_camera_fusion_distance_cm:
                return False
        return True

    def _record_master_observation_locked(self, identity_id, track_key, map_point, observed_at):
        camera_id = self._camera_from_key(track_key)
        if camera_id is None:
            return
        normalized_point = None
        if map_point is not None:
            point_array = np.asarray(map_point, dtype=float).reshape(-1)
            if point_array.size == 2 and np.all(np.isfinite(point_array)):
                normalized_point = (float(point_array[0]), float(point_array[1]))
        self.recent_master_observations.setdefault(identity_id, {})[str(camera_id)] = {
            "track_key": track_key,
            "map_point": normalized_point,
            "observed_at": float(observed_at),
        }

    def _matching_identity_locked(
        self,
        query_feature,
        query_feature_space_id=None,
        excluded_identity_ids=None,
        camera_id=None,
        map_point=None,
        observed_at=None,
    ):
        excluded = set(excluded_identity_ids or ())
        best_identity = None
        best_slot = None
        best_distance = float("inf")

        for identity_id, record in self.identities.items():
            if identity_id in excluded:
                continue
            if observed_at is not None and not self._physical_match_allowed_locked(
                identity_id,
                camera_id,
                map_point,
                observed_at,
            ):
                continue
            gallery = record.get("gallery", {})
            for slot_name in REID_GALLERY_SLOTS:
                slot = gallery.get(slot_name)
                if not slot:
                    continue
                if (
                    query_feature_space_id is not None
                    and slot.get("feature_space_id") != query_feature_space_id
                ):
                    continue
                saved_feature = self._normalize_feature(slot.get("feature"))
                if saved_feature is None or saved_feature.shape != query_feature.shape:
                    continue
                # Re-normalize in float64 for the comparison itself. This
                # keeps an exact 0.35 boundary from slipping below the strict
                # threshold solely because float32 normalization rounded up.
                query64 = np.asarray(query_feature, dtype=np.float64)
                saved64 = np.asarray(saved_feature, dtype=np.float64)
                query64 /= np.linalg.norm(query64)
                saved64 /= np.linalg.norm(saved64)
                distance = 1.0 - float(np.dot(query64, saved64))
                if distance < best_distance:
                    best_distance = distance
                    best_identity = identity_id
                    best_slot = slot_name

        if best_identity is None or not best_distance < self.distance_threshold:
            return None, None, None
        return best_identity, best_slot, best_distance

    def find_matching_identity(
        self,
        feature,
        frame_index=None,
        excluded_identity_ids=None,
        feature_space_id=None,
    ):
        del frame_index  # persistent master galleries do not expire by camera frame count
        normalized = self._normalize_feature(feature)
        if normalized is None:
            return None, 0.0, None
        with self._lock:
            identity_id, _slot_name, distance = self._matching_identity_locked(
                normalized,
                query_feature_space_id=feature_space_id,
                excluded_identity_ids=excluded_identity_ids,
            )
        if identity_id is None:
            return None, 0.0, None
        return identity_id, 1.0 - distance, distance

    def _same_camera_active_ids_locked(self, camera_id, excluded_track_key=None):
        if camera_id is None:
            visible_keys = self.visible_track_keys_by_camera.get(None, set())
        else:
            visible_keys = self.visible_track_keys_by_camera.get(str(camera_id), set())
        return {
            self.track_to_identity[key]
            for key in visible_keys
            if key != excluded_track_key and key in self.track_to_identity
        }

    def _target_identity_match_locked(self, identity_id, query_feature, feature_space_id):
        """Compare with exactly one master, independent of normal exclusions."""

        if identity_id not in self.identities:
            return None, None, None
        excluded = set(self.identities)
        excluded.discard(identity_id)
        return self._matching_identity_locked(
            query_feature,
            query_feature_space_id=feature_space_id,
            excluded_identity_ids=excluded,
        )

    def _visible_same_camera_identity_owners_locked(self, identity_id, camera_id, excluded_key=None):
        camera_key = None if camera_id is None else str(camera_id)
        return {
            key
            for key in self.visible_track_keys_by_camera.get(camera_key, set())
            if key != excluded_key and self.track_to_identity.get(key) == identity_id
        }

    def _clear_nonvisible_same_camera_owners_locked(self, identity_id, camera_id, preserved_key=None):
        """Retire stale local aliases without ever stealing from a live box."""

        camera_key = None if camera_id is None else str(camera_id)
        visible_keys = self.visible_track_keys_by_camera.get(camera_key, set())
        stale_owner_keys = [
            owner_key
            for owner_key, owner_identity_id in list(self.track_to_identity.items())
            if (
                owner_key != preserved_key
                and owner_identity_id == identity_id
                and self._camera_from_key(owner_key) == camera_key
                and owner_key not in visible_keys
            )
        ]
        for owner_key in stale_owner_keys:
            self._clear_local_binding_locked(owner_key)
        return stale_owner_keys

    def _promote_verified_shadow_locked(self, key, shadow):
        """Atomically move a verified master onto its surviving local track."""

        identity_id = shadow.get("identity_id")
        if not shadow.get("verified") or identity_id not in self.identities:
            return None
        camera_id = self._camera_from_key(key)
        if self._visible_same_camera_identity_owners_locked(
            identity_id,
            camera_id,
            excluded_key=key,
        ):
            return None

        self._clear_nonvisible_same_camera_owners_locked(
            identity_id,
            camera_id,
            preserved_key=key,
        )
        verification = dict(shadow.get("verification", {}))
        self.track_to_identity[key] = identity_id
        self.track_binding_metadata[key] = {
            "query_feature_space_id": verification.get("query_feature_space_id"),
            "matched_feature_space_id": verification.get("matched_feature_space_id"),
            "matched_slot": verification.get("matched_slot"),
            "distance": verification.get("distance"),
            "appearance_confirmed": bool(verification.get("appearance_confirmed", False)),
            "feature_source": verification.get("feature_source"),
            "handoff_from_track_key": shadow.get("canonical_key"),
        }
        self.track_results[key] = {
            "similarity": float(verification.get("similarity", 0.0)),
            "reidentified": True,
            "matched_slot": verification.get("matched_slot"),
        }
        self.pending_intake.pop(key, None)
        self.shadow_tracks.pop(key, None)
        return identity_id

    def _next_track_generation_locked(self, key):
        generation = int(self.track_generations.get(key, 0)) + 1
        self.track_generations[key] = generation
        return generation

    def _clear_local_binding_locked(self, key, clear_last_seen=False):
        self.track_to_identity.pop(key, None)
        self.track_results.pop(key, None)
        self.track_binding_metadata.pop(key, None)
        self.pending_intake.pop(key, None)
        self.shadow_tracks.pop(key, None)
        self._next_track_generation_locked(key)
        if clear_last_seen:
            self.track_last_seen.pop(key, None)

    def _release_shadow_locked(self, key):
        """Release a candidate into normal intake and invalidate stale work."""

        self.shadow_tracks.pop(key, None)
        if key in self.pending_intake:
            self.pending_intake.pop(key, None)
            self.track_results.pop(key, None)
            self.track_binding_metadata.pop(key, None)
            self._next_track_generation_locked(key)

    def _intake_task_is_current_locked(self, task, require_visible=True):
        key = task.get("track_key")
        state = self.pending_intake.get(key)
        if (
            state is None
            or not state.get("submitted")
            or int(state.get("generation", -1)) != int(task.get("generation", -2))
        ):
            return False
        if require_visible:
            camera_key = self._camera_from_key(key)
            if camera_key in self.visible_track_keys_by_camera:
                return key in self.visible_track_keys_by_camera[camera_key]
        return True

    def observe_tracks(
        self,
        track_ids,
        boxes,
        frame_index=None,
        camera_id=None,
        observed_at=None,
    ):
        """Register one camera frame and nominate newly spawned shadow tracks.

        A candidate is only nominated when it is a newly visible, unbound
        local track whose box strongly overlaps an older, currently visible
        track in the same camera. The older track may already own a master ID
        or still be completing its initial intake. Brief candidates are
        suppressed without ReID; persistent candidates receive one bounded
        five-crop appearance check. If the canonical track disappears, a
        matching candidate inherits its master through an atomic handoff.
        """

        raw_track_ids = list(track_ids if track_ids is not None else ())
        ordered_keys = [self._track_key(track_id, camera_id) for track_id in raw_track_ids]
        keys = set(ordered_keys)
        raw_boxes = list(boxes) if boxes is not None else []
        current_boxes = {}
        for index, key in enumerate(ordered_keys):
            if index >= len(raw_boxes):
                break
            normalized = self._normalized_box(raw_boxes[index])
            if normalized is not None:
                current_boxes[key] = normalized

        camera_key = None if camera_id is None else str(camera_id)
        with self._lock:
            previous_keys = set(self.visible_track_keys_by_camera.get(camera_key, set()))

            for missing_key in previous_keys - keys:
                if missing_key in self.pending_intake:
                    self.pending_intake.pop(missing_key, None)
                    self.track_results.pop(missing_key, None)
                    self.track_binding_metadata.pop(missing_key, None)
                    self._next_track_generation_locked(missing_key)
                # A vanished provisional track must not be resurrected by a
                # background task that was already extracting its features.
                self.shadow_tracks.pop(missing_key, None)

            # A brief detector wobble must not turn a duplicate into a new
            # identity. Require consecutive independent-motion observations
            # before releasing a nominated shadow.
            for key, shadow in list(self.shadow_tracks.items()):
                if self._camera_from_key(key) != camera_key or key not in keys:
                    continue
                canonical_key = shadow.get("canonical_key")
                identity_id = shadow.get("identity_id")
                mapped_identity_id = self.track_to_identity.get(canonical_key)
                if mapped_identity_id in self.identities:
                    if identity_id is not None and identity_id != mapped_identity_id:
                        self._release_shadow_locked(key)
                        continue
                    identity_id = mapped_identity_id
                    shadow["identity_id"] = identity_id
                    shadow["provisional"] = False
                elif identity_id is None:
                    # A provisional canonical is useful only while its own
                    # intake is still alive. Its disappearance or cancellation
                    # releases the newer candidate immediately.
                    if canonical_key not in keys or canonical_key not in self.pending_intake:
                        self._release_shadow_locked(key)
                        continue
                elif not shadow.get("verified"):
                    # An unverified target whose local owner was revoked is no
                    # longer a safe handoff candidate.
                    self._release_shadow_locked(key)
                    continue

                if canonical_key in keys and key in current_boxes and canonical_key in current_boxes:
                    overlap_score = self._shadow_overlap_score(
                        current_boxes[key],
                        current_boxes[canonical_key],
                    )
                    if overlap_score is None:
                        shadow["separation_frames"] = int(shadow.get("separation_frames", 0)) + 1
                        if shadow["separation_frames"] >= self.shadow_separation_frames:
                            self._release_shadow_locked(key)
                            continue
                    else:
                        shadow["separation_frames"] = 0
                        shadow["overlap_frames"] = int(shadow.get("overlap_frames", 0)) + 1
                        shadow["overlap_score"] = overlap_score
                shadow["last_frame"] = None if frame_index is None else int(frame_index)
                shadow["last_seen"] = (
                    time.monotonic() if observed_at is None else float(observed_at)
                )

            canonical_keys = []
            for canonical_key in keys:
                if canonical_key not in current_boxes:
                    continue
                mapped_identity_id = self.track_to_identity.get(canonical_key)
                if mapped_identity_id in self.identities:
                    canonical_keys.append((canonical_key, mapped_identity_id, False))
                elif canonical_key in previous_keys and canonical_key in self.pending_intake:
                    # The older track is still building its first five-crop
                    # master. Hold the newly spawned overlap behind it instead
                    # of allowing two identical intakes to race into IDs 1/2.
                    canonical_keys.append((canonical_key, None, True))
            for key in keys - previous_keys:
                if (
                    key in self.track_to_identity
                    or key in self.pending_intake
                    or key in self.shadow_tracks
                    or key not in current_boxes
                ):
                    continue
                best_canonical = None
                best_identity_id = None
                best_provisional = False
                best_score = None
                for canonical_key, canonical_identity_id, provisional in canonical_keys:
                    if canonical_key == key:
                        continue
                    canonical_box = current_boxes.get(canonical_key, self.track_boxes.get(canonical_key))
                    score = self._shadow_overlap_score(current_boxes[key], canonical_box)
                    if score is not None and (best_score is None or score > best_score):
                        best_canonical = canonical_key
                        best_identity_id = canonical_identity_id
                        best_provisional = provisional
                        best_score = score
                if best_canonical is None:
                    continue
                self.shadow_tracks[key] = {
                    "canonical_key": best_canonical,
                    "identity_id": best_identity_id,
                    "provisional": best_provisional,
                    "verified": False,
                    "first_frame": None if frame_index is None else int(frame_index),
                    "last_frame": None if frame_index is None else int(frame_index),
                    "first_seen": time.monotonic() if observed_at is None else float(observed_at),
                    "last_seen": time.monotonic() if observed_at is None else float(observed_at),
                    "overlap_score": best_score,
                    "overlap_frames": 1,
                    "separation_frames": 0,
                }

            self.track_boxes.update(current_boxes)
            self.visible_track_keys_by_camera[camera_key] = keys
            if frame_index is not None:
                for key in keys:
                    previous_seen = self.track_last_seen.get(key)
                    if previous_seen is None:
                        continue
                    frame_gap = int(frame_index) - int(previous_seen[0])
                    if frame_gap < 0 or frame_gap > self.ttl_frames:
                        self._clear_local_binding_locked(key)
            return {
                self.track_to_identity[key]
                for key in keys
                if key in self.track_to_identity
            }

    def mapped_identity_ids(self, track_ids, camera_id=None, frame_index=None):
        """Backward-compatible visibility update without box-aware shadows."""

        return self.observe_tracks(
            track_ids,
            None,
            frame_index=frame_index,
            camera_id=camera_id,
        )

    def is_track_suppressed(self, track_id, camera_id=None):
        key = self._track_key(track_id, camera_id)
        with self._lock:
            shadow = self.shadow_tracks.get(key)
            if shadow is None or key in self.track_to_identity:
                return False
            camera_key = self._camera_from_key(key)
            return shadow.get("canonical_key") in self.visible_track_keys_by_camera.get(
                camera_key,
                set(),
            )

    def lookup(self, track_id, camera_id=None):
        key = self._track_key(track_id, camera_id)
        with self._lock:
            return self.track_to_identity.get(key)

    def assignment_metadata(self, track_id, camera_id=None):
        key = self._track_key(track_id, camera_id)
        with self._lock:
            return dict(self.track_binding_metadata.get(key, {}))

    def pending_count(self, track_id, camera_id=None):
        key = self._track_key(track_id, camera_id)
        with self._lock:
            state = self.pending_intake.get(key)
            return len(state.get("samples", ())) if state else 0

    def required_intake_count(self):
        return self.intake_frames

    def gallery_status(self, identity_id):
        with self._lock:
            record = self.identities.get(identity_id)
            if record is None:
                return 0, len(REID_GALLERY_SLOTS)
            gallery = record.get("gallery", {})
            return sum(gallery.get(slot) is not None for slot in REID_GALLERY_SLOTS), len(REID_GALLERY_SLOTS)

    def identity_metadata(self, identity_id):
        with self._lock:
            record = self.identities.get(identity_id)
            if record is None:
                return {}
            baseline = record.get("gallery", {}).get("baseline") or {}
            return {
                "role": record.get("role", "evacuee"),
                "age": record.get("age", "Unknown"),
                "gender": record.get("gender", "Unknown"),
                "gallery_filled": sum(
                    record.get("gallery", {}).get(slot) is not None
                    for slot in REID_GALLERY_SLOTS
                ),
                "gallery_total": len(REID_GALLERY_SLOTS),
                "feature_source": baseline.get("feature_source"),
            }

    def semantic_probe_due(self, track_id, crop, frame_index, detection_confidence, camera_id=None):
        key = self._track_key(track_id, camera_id)
        with self._lock:
            identity_id = self.track_to_identity.get(key)
            if identity_id is None:
                return False
            record = self.identities.get(identity_id)
            if record is None:
                return False
            if all(record.get("gallery", {}).get(slot) is not None for slot in REID_SEMANTIC_SLOTS):
                return False
            semantic_clock_key = (identity_id, self._camera_from_key(key))
            if int(frame_index) < int(self.next_semantic_attempt_frame.get(semantic_clock_key, 0)):
                return False
        if detection_confidence is None or float(detection_confidence) <= self.semantic_confidence_threshold:
            with self._lock:
                self.next_semantic_attempt_frame[semantic_clock_key] = (
                    int(frame_index) + self.semantic_retry_frames
                )
            return False
        sharpness = image_sharpness(crop)
        with self._lock:
            if self.track_to_identity.get(key) != identity_id:
                return False
            if sharpness <= self.blur_threshold:
                self.next_semantic_attempt_frame[semantic_clock_key] = (
                    int(frame_index) + self.semantic_retry_frames
                )
                return False
            self.semantic_probe_quality[(key, int(frame_index))] = sharpness
        return True

    def _queue_task_locked(self, task):
        if self._worker is None:
            self._process_task(task)
            return True
        try:
            self._task_queue.put_nowait(task)
            return True
        except queue.Full:
            if self.verbose:
                print("ReID analyst queue is full; task will be retried.")
            return False

    def _schedule_semantic_locked(
        self,
        key,
        identity_id,
        crop,
        frame_index,
        orientation,
        detection_confidence,
        observed_at,
    ):
        record = self.identities.get(identity_id)
        if record is None:
            return
        gallery = record.get("gallery", {})
        if all(gallery.get(slot) is not None for slot in REID_SEMANTIC_SLOTS):
            return
        semantic_clock_key = (identity_id, self._camera_from_key(key))
        if int(frame_index) < int(self.next_semantic_attempt_frame.get(semantic_clock_key, 0)):
            return

        sharpness = self.semantic_probe_quality.pop((key, int(frame_index)), None)
        if sharpness is None:
            sharpness = image_sharpness(crop)
        if (
            detection_confidence is None
            or float(detection_confidence) <= self.semantic_confidence_threshold
            or sharpness <= self.blur_threshold
            or orientation not in REID_SEMANTIC_SLOTS
        ):
            self.next_semantic_attempt_frame[semantic_clock_key] = int(frame_index) + self.semantic_retry_frames
            return
        if gallery.get(orientation) is not None or (identity_id, orientation) in self.pending_semantic_slots:
            self.next_semantic_attempt_frame[semantic_clock_key] = int(frame_index) + self.semantic_retry_frames
            return

        task = {
            "type": "semantic",
            "track_key": key,
            "identity_id": identity_id,
            "slot_name": orientation,
            "sample": {
                "crop": crop.copy(),
                "frame_index": int(frame_index),
                "camera_id": self._camera_from_key(key),
                "observed_at": float(observed_at),
                "sharpness": sharpness,
                "area": int(crop.shape[0] * crop.shape[1]),
                "detection_confidence": float(detection_confidence),
                "orientation": orientation,
            },
        }
        if self._queue_task_locked(task):
            self.pending_semantic_slots.add((identity_id, orientation))
            self.next_semantic_attempt_frame[semantic_clock_key] = int(frame_index) + self.semantic_cooldown_frames
        else:
            self.next_semantic_attempt_frame[semantic_clock_key] = int(frame_index) + self.semantic_retry_frames

    def assign(
        self,
        track_id,
        crop,
        frame_index,
        precomputed_feature=None,
        excluded_identity_ids=None,
        camera_id=None,
        detection_confidence=None,
        orientation=None,
        observed_at=None,
        map_point=None,
    ):
        del precomputed_feature
        key = self._track_key(track_id, camera_id)
        now = time.monotonic() if observed_at is None else float(observed_at)
        if crop is None or crop.size == 0:
            with self._lock:
                return self.track_to_identity.get(key), 0.0, False

        with self._lock:
            previous_seen = self.track_last_seen.get(key)
            if previous_seen is not None:
                frame_gap = int(frame_index) - int(previous_seen[0])
                if frame_gap < 0 or frame_gap > self.ttl_frames:
                    self._clear_local_binding_locked(key)
            self.track_last_seen[key] = (int(frame_index), now)

            handoff_identity_id = None
            handoff_from_key = None
            shadow = self.shadow_tracks.get(key)
            if shadow is not None:
                handoff_from_key = shadow.get("canonical_key")
                camera_key = self._camera_from_key(key)
                visible_keys = self.visible_track_keys_by_camera.get(camera_key, set())
                canonical_visible = handoff_from_key in visible_keys
                mapped_target = self.track_to_identity.get(handoff_from_key)
                handoff_identity_id = shadow.get("identity_id")

                if mapped_target in self.identities:
                    if handoff_identity_id is not None and handoff_identity_id != mapped_target:
                        self._release_shadow_locked(key)
                        shadow = None
                        handoff_identity_id = None
                        handoff_from_key = None
                    else:
                        handoff_identity_id = mapped_target
                        shadow["identity_id"] = mapped_target
                        shadow["provisional"] = False
                elif handoff_identity_id is None:
                    if canonical_visible and handoff_from_key in self.pending_intake:
                        # The older overlapping track has not produced its
                        # master yet. It owns the only intake until that race
                        # resolves, regardless of how long both boxes persist.
                        return None, 0.0, False
                    self._release_shadow_locked(key)
                    shadow = None
                    handoff_from_key = None
                elif not shadow.get("verified"):
                    self._release_shadow_locked(key)
                    shadow = None
                    handoff_identity_id = None
                    handoff_from_key = None

                if shadow is not None and shadow.get("verified"):
                    if canonical_visible:
                        # One successful five-crop comparison is enough. Keep
                        # the duplicate hidden without repeating GPU work.
                        return None, 0.0, False
                    promoted_identity_id = self._promote_verified_shadow_locked(key, shadow)
                    if promoted_identity_id is None:
                        # Another live same-camera owner appeared between the
                        # frame observation and this assignment. Preserve the
                        # single-owner invariant and wait for the next frame.
                        return None, 0.0, False
                    handoff_identity_id = promoted_identity_id
                    shadow = None
                elif (
                    shadow is not None
                    and canonical_visible
                    and int(shadow.get("overlap_frames", 0)) <= self.shadow_probation_frames
                ):
                    # The first few overlap frames are treated as detector
                    # noise. A persistent candidate earns one appearance test
                    # only after this cheap probation has elapsed.
                    return None, 0.0, False

            identity_id = self.track_to_identity.get(key)
            if identity_id is not None and identity_id in self.identities:
                if not self._physical_match_allowed_locked(
                    identity_id,
                    self._camera_from_key(key),
                    map_point,
                    now,
                ):
                    revoked_identity_id = identity_id
                    camera_observations = self.recent_master_observations.get(identity_id, {})
                    camera_observations.pop(str(self._camera_from_key(key)), None)
                    self._clear_local_binding_locked(key)
                    identity_id = None
                    if self.verbose:
                        print(
                            f"ReID: revoked {key} -> Master {revoked_identity_id} "
                            "after an impossible map jump"
                        )
                else:
                    record = self.identities[identity_id]
                    record["hits"] = int(record.get("hits", 0)) + 1
                    record["last_seen_monotonic"] = now
                    self._record_master_observation_locked(identity_id, key, map_point, now)
                    result = self.track_results.pop(key, None)
                    self._schedule_semantic_locked(
                        key,
                        identity_id,
                        crop,
                        frame_index,
                        orientation,
                        detection_confidence,
                        now,
                    )
                    if result is not None:
                        return identity_id, result["similarity"], result["reidentified"]
                    return identity_id, 1.0, False
            if identity_id is not None:
                self._clear_local_binding_locked(key)

            state = self.pending_intake.get(key)
            if state is None:
                state = {
                    "first_seen": now,
                    "last_frame": None,
                    "samples": [],
                    "submitted": False,
                    "generation": self._next_track_generation_locked(key),
                    "next_retry_frame": int(frame_index),
                    "failure_count": 0,
                    "handoff_identity_id": handoff_identity_id,
                    "handoff_from_key": handoff_from_key,
                }
                self.pending_intake[key] = state
            if state["submitted"]:
                return None, 0.0, False
            if now - float(state["first_seen"]) < self.intake_delay_seconds:
                return None, 0.0, False
            if state["last_frame"] == int(frame_index):
                return None, 0.0, False

            sharpness = image_sharpness(crop)
            timed_out = now - float(state["first_seen"]) >= self.intake_timeout_seconds
            if sharpness <= self.blur_threshold and not timed_out:
                return None, 0.0, False

            state["last_frame"] = int(frame_index)
            state["samples"].append(
                {
                    "crop": crop.copy(),
                    "frame_index": int(frame_index),
                    "camera_id": camera_id,
                    "observed_at": now,
                    "sharpness": sharpness,
                    "area": int(crop.shape[0] * crop.shape[1]),
                    "detection_confidence": None if detection_confidence is None else float(detection_confidence),
                    "orientation": orientation if orientation in REID_SEMANTIC_SLOTS else None,
                    "map_point": None if map_point is None else tuple(map(float, map_point)),
                }
            )
            if len(state["samples"]) > self.intake_frames:
                state["samples"] = state["samples"][-self.intake_frames :]
            if len(state["samples"]) < self.intake_frames:
                return None, 0.0, False
            if int(frame_index) < int(state.get("next_retry_frame", 0)):
                return None, 0.0, False

            camera_key = self._camera_from_key(key)
            visible_peer_keys = set(self.visible_track_keys_by_camera.get(camera_key, set()))
            visible_peer_keys.discard(key)

            task = {
                "type": "intake",
                "track_key": key,
                "camera_id": camera_id,
                "frame_index": int(frame_index),
                "samples": [
                    {**sample, "crop": sample["crop"].copy()}
                    for sample in state["samples"][: self.intake_frames]
                ],
                "excluded_identity_ids": set(excluded_identity_ids or ()),
                "same_camera_peer_keys": visible_peer_keys,
                "generation": state["generation"],
                "handoff_identity_id": state.get("handoff_identity_id"),
                "handoff_from_key": state.get("handoff_from_key"),
            }
            if self._queue_task_locked(task):
                state["submitted"] = True
            else:
                state["next_retry_frame"] = int(frame_index) + self.intake_retry_frames
            return None, 0.0, False

    def _get_role_classifier(self):
        if not self.enable_role_classification:
            return None
        if self._role_classifier is None:
            self._role_classifier = EvacuationRoleClassifier(self.role_checkpoint)
        return self._role_classifier

    def _process_intake_task(self, task):
        with self._lock:
            if not self._intake_task_is_current_locked(task):
                return
        samples = task["samples"]
        crops = [sample["crop"] for sample in samples]
        features, feature_source, feature_space_id = self._extract_aligned_features(crops)
        valid_indices = [index for index, feature in enumerate(features) if feature is not None]
        if not valid_indices:
            raise RuntimeError("No ReID feature could be extracted from the intake burst.")

        query_feature = self._normalize_feature(
            np.mean(np.asarray([features[index] for index in valid_indices], dtype=np.float32), axis=0)
        )
        if query_feature is None:
            raise RuntimeError("The intake fingerprint had zero norm.")
        hero_index = max(valid_indices, key=lambda index: self._quality_score(samples[index]))
        role_classifier = self._get_role_classifier()
        role, role_confidence = (
            role_classifier.predict(crops[hero_index])
            if role_classifier is not None
            else ("evacuee", 0.0)
        )
        if role in ("cag", "scdf") and role_confidence < self.role_confidence_threshold:
            role = "evacuee"

        demographics_task = None
        key = task["track_key"]
        camera_id = task.get("camera_id")
        handoff_identity_id = task.get("handoff_identity_id")
        handoff_from_key = task.get("handoff_from_key")
        handoff_committed = False
        latest_spatial_sample = next(
            (sample for sample in reversed(samples) if sample.get("map_point") is not None),
            samples[-1],
        )
        with self._lock:
            if not self._intake_task_is_current_locked(task):
                return
            if handoff_identity_id is not None:
                shadow = self.shadow_tracks.get(key)
                if (
                    shadow is None
                    or shadow.get("identity_id") != handoff_identity_id
                    or shadow.get("canonical_key") != handoff_from_key
                ):
                    return
            dynamic_exclusions = self._same_camera_active_ids_locked(camera_id, excluded_track_key=key)
            submitted_peer_ids = {
                self.track_to_identity[peer_key]
                for peer_key in task.get("same_camera_peer_keys", ())
                if peer_key in self.track_to_identity
            }
            excluded = (
                set(task.get("excluded_identity_ids", ()))
                | dynamic_exclusions
                | submitted_peer_ids
            )
            identity_id = None
            matched_slot = None
            distance = None

            if handoff_identity_id is not None:
                identity_id, matched_slot, distance = self._target_identity_match_locked(
                    handoff_identity_id,
                    query_feature,
                    feature_space_id,
                )
                if identity_id is not None:
                    visible_owners = self._visible_same_camera_identity_owners_locked(
                        handoff_identity_id,
                        camera_id,
                        excluded_key=key,
                    )
                    if visible_owners:
                        # Appearance confirms that both simultaneous boxes are
                        # the same person. Remember that result, keep exactly
                        # one visible owner, and do not run TransReID again.
                        shadow = self.shadow_tracks.get(key)
                        if shadow is not None:
                            shadow["canonical_key"] = min(visible_owners, key=repr)
                            shadow["identity_id"] = handoff_identity_id
                            shadow["provisional"] = False
                            shadow["verified"] = True
                            baseline = self.identities[handoff_identity_id]["gallery"]["baseline"]
                            shadow["verification"] = {
                                "query_feature_space_id": feature_space_id,
                                "matched_feature_space_id": baseline.get("feature_space_id"),
                                "matched_slot": matched_slot,
                                "distance": distance,
                                "similarity": 1.0 - float(distance),
                                "appearance_confirmed": bool(
                                    feature_source == "transreid"
                                    and feature_space_id == baseline.get("feature_space_id")
                                ),
                                "feature_source": feature_source,
                            }
                        self.pending_intake.pop(key, None)
                        self._next_track_generation_locked(key)
                        return

                    self._clear_nonvisible_same_camera_owners_locked(
                        handoff_identity_id,
                        camera_id,
                        preserved_key=key,
                    )
                    handoff_committed = True
                else:
                    # Appearance has vetoed the geometric handoff. Reuse this
                    # already-computed batch in the normal match/create path,
                    # while explicitly preventing a second attempt at the
                    # rejected canonical master.
                    self.shadow_tracks.pop(key, None)
                    excluded.add(handoff_identity_id)

            while identity_id is None:
                candidate_identity_id, candidate_slot, candidate_distance = self._matching_identity_locked(
                    query_feature,
                    query_feature_space_id=feature_space_id,
                    excluded_identity_ids=excluded,
                    camera_id=camera_id,
                    map_point=latest_spatial_sample.get("map_point"),
                    observed_at=latest_spatial_sample.get("observed_at"),
                )
                if candidate_identity_id is None:
                    matched_slot = None
                    distance = None
                    break
                visible_owners = self._visible_same_camera_identity_owners_locked(
                    candidate_identity_id,
                    camera_id,
                    excluded_key=key,
                )
                if visible_owners:
                    # Visibility is the final single-owner guard. Do not steal
                    # a master even if a stale submission snapshot omitted it;
                    # try the next eligible gallery instead.
                    excluded.add(candidate_identity_id)
                    continue
                identity_id = candidate_identity_id
                matched_slot = candidate_slot
                distance = candidate_distance
                self._clear_nonvisible_same_camera_owners_locked(
                    identity_id,
                    camera_id,
                    preserved_key=key,
                )
            reidentified = identity_id is not None
            if identity_id is None:
                identity_id = self.next_identity_id
                baseline_sample = samples[hero_index]
                baseline_slot = self._make_slot(
                    identity_id,
                    "baseline",
                    features[hero_index],
                    baseline_sample,
                    feature_source,
                    feature_space_id,
                )
                self.next_identity_id += 1
                self.identities[identity_id] = self._new_record(role, role_confidence)
                self.identities[identity_id]["gallery"]["baseline"] = baseline_slot
                if role == "evacuee" and self.enable_demographics:
                    demographics_task = {
                        "identity_id": identity_id,
                        "crops": [crop.copy() for crop in crops],
                    }
            record = self.identities[identity_id]
            baseline_space = record["gallery"]["baseline"].get("feature_space_id")
            if baseline_space != feature_space_id:
                raise RuntimeError("Refusing to mix incompatible ReID feature spaces in one master gallery.")

            # Reuse already-computed intake features for distinct, reliable
            # semantic views instead of scheduling avoidable future GPU calls.
            best_semantic_samples = {}
            for index in valid_indices:
                if index == hero_index:
                    continue
                slot_name = samples[index].get("orientation")
                if slot_name not in REID_SEMANTIC_SLOTS:
                    continue
                previous = best_semantic_samples.get(slot_name)
                if previous is None or self._quality_score(samples[index]) > self._quality_score(samples[previous]):
                    best_semantic_samples[slot_name] = index
            for slot_name, index in best_semantic_samples.items():
                if record["gallery"].get(slot_name) is None:
                    slot = self._make_slot(
                        identity_id,
                        slot_name,
                        features[index],
                        samples[index],
                        feature_source,
                        feature_space_id,
                    )
                    record["gallery"][slot_name] = slot

            record["hits"] = int(record.get("hits", 0)) + 1
            record["last_seen_monotonic"] = time.monotonic()
            self.track_to_identity[key] = identity_id
            self.track_binding_metadata[key] = {
                "query_feature_space_id": feature_space_id,
                "matched_feature_space_id": baseline_space,
                "matched_slot": matched_slot,
                "distance": distance,
                "appearance_confirmed": bool(
                    feature_source == "transreid" and feature_space_id == baseline_space
                ),
                "feature_source": feature_source,
                "handoff_from_track_key": handoff_from_key if handoff_committed else None,
            }
            self._record_master_observation_locked(
                identity_id,
                key,
                latest_spatial_sample.get("map_point"),
                latest_spatial_sample.get("observed_at", time.monotonic()),
            )
            self.track_results[key] = {
                "similarity": 0.0 if distance is None else 1.0 - float(distance),
                "reidentified": reidentified,
                "matched_slot": matched_slot,
            }
            self.pending_intake.pop(key, None)
            self.shadow_tracks.pop(key, None)

        self.save_database(identity_id)

        if demographics_task is not None:
            self._ensure_demographics_worker()
            try:
                self._demographics_queue.put_nowait(demographics_task)
            except queue.Full:
                print(f"Demographics queue full; ID {identity_id} marked Unknown.")
                with self._lock:
                    record = self.identities.get(identity_id)
                    if record is not None:
                        record["age"] = "Unknown"
                        record["gender"] = "Unknown"
                self.save_database(identity_id)

        if self.verbose:
            if reidentified:
                print(
                    f"ReID: {key} -> Master {identity_id} via {matched_slot} "
                    f"(distance={distance:.3f})"
                )
            else:
                print(f"ReID: created Master {identity_id} from one {len(crops)}-crop batch")

    def _process_semantic_task(self, task):
        sample = task["sample"]
        features, feature_source, feature_space_id = self._extract_aligned_features([sample["crop"]])
        feature = features[0] if features else None
        if feature is None:
            raise RuntimeError("No feature could be extracted for the semantic slot.")

        identity_id = task["identity_id"]
        slot_name = task["slot_name"]
        with self._lock:
            record = self.identities.get(identity_id)
            if record is None or record.get("gallery", {}).get(slot_name) is not None:
                return
            # Discard stale work if ByteTrack remapped this local key while
            # the GPU task was waiting in the queue.
            if self.track_to_identity.get(task["track_key"]) != identity_id:
                return
            baseline = record.get("gallery", {}).get("baseline") or {}
            if baseline.get("feature_space_id") != feature_space_id:
                raise RuntimeError("Semantic crop used an incompatible ReID feature space.")
            slot = self._make_slot(
                identity_id,
                slot_name,
                feature,
                sample,
                feature_source,
                feature_space_id,
            )
            record["gallery"][slot_name] = slot
        self.save_database(identity_id)
        if self.verbose:
            print(f"ReID: filled {slot_name} for Master {identity_id}")

    def _process_task(self, task):
        if task["type"] == "intake":
            self._process_intake_task(task)
        elif task["type"] == "semantic":
            self._process_semantic_task(task)
        else:
            raise ValueError(f"Unknown ReID analyst task: {task['type']}")

    def _worker_loop(self):
        while True:
            task = self._task_queue.get()
            try:
                if task is self._stop_token:
                    return
                self._process_task(task)
            except Exception as exc:
                print(f"ReID analyst task failed: {exc}")
                if isinstance(task, dict) and task.get("type") == "intake":
                    with self._lock:
                        state = self.pending_intake.get(task.get("track_key"))
                        if (
                            state is not None
                            and int(state.get("generation", -1)) == int(task.get("generation", -2))
                        ):
                            failure_count = int(state.get("failure_count", 0)) + 1
                            retry_frames = min(
                                self.max_retry_frames,
                                self.intake_retry_frames * (2 ** min(failure_count - 1, 8)),
                            )
                            state["submitted"] = False
                            state["samples"] = []
                            state["last_frame"] = None
                            task_samples = task.get("samples") or ()
                            state["first_seen"] = float(
                                task_samples[-1].get("observed_at", time.monotonic())
                                if task_samples
                                else time.monotonic()
                            )
                            state["failure_count"] = failure_count
                            state["next_retry_frame"] = (
                                int(task.get("frame_index", 0)) + int(retry_frames)
                            )
                            state["generation"] = self._next_track_generation_locked(
                                task.get("track_key")
                            )
                elif isinstance(task, dict) and task.get("type") == "semantic":
                    with self._lock:
                        sample = task.get("sample", {})
                        semantic_clock_key = (
                            task.get("identity_id"),
                            self._camera_from_key(task.get("track_key")),
                        )
                        self.next_semantic_attempt_frame[semantic_clock_key] = (
                            int(sample.get("frame_index", 0)) + self.semantic_retry_frames
                        )
            finally:
                if isinstance(task, dict) and task.get("type") == "semantic":
                    with self._lock:
                        self.pending_semantic_slots.discard((task.get("identity_id"), task.get("slot_name")))
                self._task_queue.task_done()

    def _ensure_demographics_worker(self):
        if self._demographics_worker is not None:
            return
        with self._lock:
            if self._demographics_worker is None:
                self._demographics_worker = threading.Thread(
                    target=self._demographics_worker_loop,
                    name="demographics-analyst",
                    daemon=True,
                )
                self._demographics_worker.start()

    def _demographics_worker_loop(self):
        while True:
            task = self._demographics_queue.get()
            try:
                if task is self._stop_token:
                    return
                if self._demographics_engine is None:
                    from demographics import DemographicsEngine

                    self._demographics_engine = DemographicsEngine(device=self.demographics_device)
                age, gender = self._demographics_engine.analyze_batch(task["crops"])
                with self._lock:
                    record = self.identities.get(task["identity_id"])
                    if record is not None and record.get("role") == "evacuee":
                        record["age"] = age
                        record["gender"] = gender
                self.save_database(task["identity_id"])
            except Exception as exc:
                print(f"Demographics analysis failed: {exc}")
                if isinstance(task, dict):
                    with self._lock:
                        record = self.identities.get(task.get("identity_id"))
                        if record is not None:
                            record["age"] = "Unknown"
                            record["gender"] = "Unknown"
                    self.save_database(task.get("identity_id"))
            finally:
                self._demographics_queue.task_done()

    def wait_for_idle(self, timeout=5.0):
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            if self._task_queue.unfinished_tasks == 0 and self._demographics_queue.unfinished_tasks == 0:
                return True
            time.sleep(0.005)
        return self._task_queue.unfinished_tasks == 0 and self._demographics_queue.unfinished_tasks == 0

    def close(self, drain=True, timeout=10.0):
        if self._closed:
            return
        self._closed = True
        if drain:
            self.wait_for_idle(timeout=timeout)
        if self._worker is not None and self._worker.is_alive():
            self._task_queue.put(self._stop_token)
            self._worker.join(timeout=timeout)
        if self._demographics_worker is not None and self._demographics_worker.is_alive():
            self._demographics_queue.put(self._stop_token)
            self._demographics_worker.join(timeout=timeout)
        if self.persistence_store is not None:
            with self._lock:
                identity_ids = list(self.identities)
            for identity_id in identity_ids:
                self.save_database(identity_id)
