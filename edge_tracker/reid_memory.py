import hashlib
import copy
import json
import os
import pickle
import queue
import subprocess
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
    DEFAULT_PROVISIONAL_CHALLENGE_DISTANCE,
    DEFAULT_PROVISIONAL_LOCATION_CONFIRM_FRAMES,
    REID_GALLERY_SLOTS,
    REID_SEMANTIC_SLOTS,
)
from identity_debug import identity_event


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
        if self._load_transreid_jpm_model():
            return
        print(
            "Warning: Exact TransReID JPM/SIE loading failed; appearance ReID is disabled "
            "instead of using an incompatible partial-weight fallback."
        )

    def is_available(self):
        return self.model is not None

    def feature_space_id(self, dimension):
        """Fingerprint the exact model and preprocessing feature space."""
        if self._checkpoint_sha256 is None:
            self._checkpoint_sha256 = _sha256_file(self.checkpoint_path)
        if self.backend == "transreid_jpm":
            config_path = Path(__file__).resolve().parent / "transreid_jpm.py"
            if self._config_sha256 is None:
                self._config_sha256 = _sha256_file(config_path)
            model_spec = "msmt17-vit-base-transreid-jpm-sie15-stride12"
            preprocess = "resize128x256-inter_linear-bgr2rgb-chw-f32-div255-mean0.5-std0.5-sie0"
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

    def _load_transreid_jpm_model(self):
        if self.fastreid_root is None or not self.fastreid_root.exists():
            print(f"TransReID ViT dependency folder not found: {self.fastreid_root}.")
            return False

        root_text = str(self.fastreid_root.resolve())
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        try:
            from fastreid.modeling.backbones.vision_transformer import VisionTransformer
            from transreid_jpm import (
                TRANSREID_FEATURE_DIM,
                build_transreid_jpm_from_checkpoint,
            )

            if not self.checkpoint_path.exists():
                print(f"TransReID checkpoint not found: {self.checkpoint_path}")
                return False

            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            model, spec = build_transreid_jpm_from_checkpoint(checkpoint, VisionTransformer)
            model.eval()
            model.to(self.device)
            print(
                "TransReID JPM/SIE backend loaded. Missing: 0, Unexpected: 0; "
                f"classes: {spec['num_classes']}, SIE cameras: {spec['camera_count']}, "
                f"feature dimension: {TRANSREID_FEATURE_DIM}"
            )

            self.model = model
            self.backend = "transreid_jpm"
            return True
        except Exception as exc:
            print(f"Unable to load exact TransReID JPM/SIE checkpoint: {exc}")
            self.model = None
            self.backend = None
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
            if self.backend == "transreid_jpm":
                resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
                tensor = tensor.sub(0.5).div(0.5)
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

        if self.backend != "transreid_jpm":
            return [feature for feature in (self.extract(crop) for crop in crops) if feature is not None]

        try:
            tensors = []
            for crop in crops:
                resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
                tensors.append(tensor.sub(0.5).div(0.5))

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

        if self.backend != "transreid_jpm":
            return [self.extract(crop) if crop is not None and crop.size > 0 else None for crop in crops]

        valid_indices = [i for i, crop in enumerate(crops) if crop is not None and crop.size > 0]
        if not valid_indices:
            return [None] * len(crops)

        try:
            tensors = []
            for i in valid_indices:
                resized = cv2.resize(crops[i], (128, 256), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
                tensors.append(tensor.sub(0.5).div(0.5))

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
        evidence_camera_ids=None,
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
        role_classifier=None,
        demographics_engine=None,
        cross_camera_fusion_distance_cm=None,
        cross_camera_max_skew_seconds=0.35,
        shadow_iou_threshold=DEFAULT_REID_SHADOW_IOU_THRESHOLD,
        shadow_containment_threshold=DEFAULT_REID_SHADOW_CONTAINMENT_THRESHOLD,
        shadow_center_distance_ratio=DEFAULT_REID_SHADOW_CENTER_DISTANCE_RATIO,
        shadow_probation_frames=DEFAULT_REID_SHADOW_PROBATION_FRAMES,
        shadow_separation_frames=DEFAULT_REID_SHADOW_SEPARATION_FRAMES,
        provisional_challenge_distance=DEFAULT_PROVISIONAL_CHALLENGE_DISTANCE,
        provisional_location_confirm_frames=DEFAULT_PROVISIONAL_LOCATION_CONFIRM_FRAMES,
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
        self.evidence_camera_ids = (
            None
            if evidence_camera_ids is None
            else {str(camera_id) for camera_id in evidence_camera_ids}
        )
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
        self.provisional_challenge_distance = max(
            self.distance_threshold,
            float(provisional_challenge_distance),
        )
        # Existing-ID matches near the configured threshold are much more
        # vulnerable when evacuees wear similar clothing.  Distances above
        # this strong boundary require the same master to win a second,
        # independently collected intake batch before the binding is final.
        self.strong_match_distance = min(self.distance_threshold, 0.20)
        self.provisional_location_confirm_frames = max(
            1,
            int(provisional_location_confirm_frames),
        )

        self.identities = {}
        self.next_identity_id = 1
        # Location-only cross-camera groups use negative internal tokens so
        # they cannot consume or appear as permanent master numbers.  A
        # positive ID is allocated only after their global ReID search has
        # finished without matching an established identity.
        self.next_temporary_group_id = -1
        self.track_to_identity = {}
        self.track_last_seen = {}
        self.track_results = {}
        self.track_binding_metadata = {}
        self.track_generations = {}
        self.pending_intake = {}
        # Tracks in a promising cross-camera location pair may finish their
        # GPU intake, but an unmatched result must not allocate a permanent
        # master until the short coordinator hold is resolved.
        self.new_master_holds = {}
        self.visible_track_keys_by_camera = {}
        self.track_boxes = {}
        self.shadow_tracks = {}
        self.pending_semantic_slots = set()
        self.next_semantic_attempt_frame = {}
        self.semantic_probe_quality = {}
        self.recent_master_observations = {}
        self.physical_violation_counts = {}
        self._evidence_capture_paths = {}
        # Evidence for tracks attached by location to an existing master is
        # quarantined here. It is committed to the permanent gallery/folder
        # only after appearance or the stable-location fallback confirms it.
        self.pending_member_evidence = {}

        self._lock = threading.RLock()
        self._persistence_lock = threading.Lock()
        self._persistence_condition = threading.Condition()
        self._pending_persistence = {}
        self._persistence_active = False
        self._persistence_stopping = False
        self._task_queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._demographics_queue = queue.Queue(maxsize=max(1, int(queue_size)))
        # The in-process thread only transfers crops to a lightweight helper
        # process. PNG hashing, encoding, and disk I/O happen outside this
        # process so they cannot hold the GIL or the identity lock.
        self._evidence_queue = queue.Queue()
        self._stop_token = object()
        self._worker = None
        self._demographics_worker = None
        self._evidence_worker = None
        self._evidence_process = None
        self._persistence_worker = None
        # Programmatic sessions may inject worker-preloaded models. CLI users
        # retain the original lazy-loading behaviour when these are omitted.
        self._role_classifier = role_classifier
        self._demographics_engine = demographics_engine
        self._closed = False

        self.load_database()
        if self.persistence_store is not None:
            self._persistence_worker = threading.Thread(
                target=self._persistence_worker_loop,
                name="reid-persistence",
                daemon=True,
            )
            self._persistence_worker.start()
        if start_worker:
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="reid-analyst",
                daemon=True,
            )
            self._worker.start()
        if self.evidence_dir is not None:
            worker_script = Path(__file__).resolve().with_name("reid_evidence_writer.py")
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._evidence_process = subprocess.Popen(
                [sys.executable, "-u", str(worker_script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                creationflags=creationflags,
                bufsize=0,
            )
            self._evidence_worker = threading.Thread(
                target=self._evidence_worker_loop,
                name="reid-evidence-sender",
                daemon=True,
            )
            self._evidence_worker.start()

    @staticmethod
    def _empty_gallery():
        return {slot_name: None for slot_name in REID_GALLERY_SLOTS}

    @staticmethod
    def _track_key(track_id, camera_id=None):
        local_id = int(track_id)
        return local_id if camera_id is None else (str(camera_id), local_id)

    @staticmethod
    def _public_identity_id(identity_id):
        return identity_id if identity_id is not None and identity_id > 0 else None

    @staticmethod
    def _temporary_group_token(identity_id):
        if identity_id is None or identity_id >= 0:
            return None
        return f"tmp_{abs(int(identity_id))}"

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

    @classmethod
    def _sample_debug_summary(cls, sample):
        crop = sample.get("crop")
        return {
            "frame_index": sample.get("frame_index"),
            "camera_id": sample.get("camera_id"),
            "observed_at": sample.get("observed_at"),
            "crop_shape": None if crop is None else tuple(int(value) for value in crop.shape),
            "area": sample.get("area"),
            "sharpness": sample.get("sharpness"),
            "detection_confidence": sample.get("detection_confidence"),
            "detection_box": sample.get("detection_box"),
            "orientation": sample.get("orientation"),
            "map_point": sample.get("map_point"),
            "body_complete": sample.get("body_complete"),
            "body_details": sample.get("body_details"),
            "baseline_quality_score": cls._quality_score(sample),
        }

    def _new_record(self, role="evacuee", role_confidence=0.0, identity_state="confirmed"):
        demographics_status = "Pending" if self.enable_demographics else "Disabled"
        return {
            "identity_state": str(identity_state),
            "location_managed": identity_state in ("provisional", "challenged"),
            "confirmation_reason": None,
            "role": role,
            "role_confidence": float(role_confidence),
            "role_classified": identity_state == "confirmed",
            "age": demographics_status if role == "evacuee" else "N/A",
            "gender": demographics_status if role == "evacuee" else "N/A",
            "gallery": self._empty_gallery(),
            "camera_views": {},
            "camera_baselines": {},
            "member_track_keys": set(),
            "pending_member_keys": set(),
            "challenged_member_keys": set(),
            # Members whose appearance check already argued against this master.
            # They may still be confirmed by a later positive ReID result, but
            # never by the stable-location fallback alone.
            "appearance_rejected_member_keys": set(),
            "pending_member_location_streaks": {},
            "global_reid_checked_track_keys": set(),
            "location_match_frames": 0,
            "reid_comparisons": {},
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
        known_identity_ids = set()
        for raw_identity_id, raw_record in loaded_identities.items():
            try:
                identity_id = int(raw_identity_id)
            except (TypeError, ValueError):
                if self.persistence_store is not None:
                    print(f"Skipping backend ReID identity with invalid ID: {raw_identity_id!r}")
                    continue
                raise RuntimeError(f"Invalid identity ID {raw_identity_id!r} in {self.db_path}.")
            known_identity_ids.add(identity_id)
            if not isinstance(raw_record, dict):
                continue
            gallery = raw_record.get("gallery")
            if not isinstance(gallery, dict) or set(gallery) != set(REID_GALLERY_SLOTS):
                raise RuntimeError(
                    f"ReID database {self.db_path} contains an invalid five-slot gallery."
                )
            if gallery.get("baseline") is None:
                if self.persistence_store is not None:
                    print(
                        f"Skipping incomplete backend ReID identity {identity_id}: "
                        "no baseline gallery slot."
                    )
                    continue
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
                evidence_expected = bool(slot.get("evidence_expected", evidence_enabled))
                if evidence_enabled and evidence_expected:
                    image_path = slot.get("image_path")
                    saved_crop = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if image_path else None
                    if saved_crop is None or self._crop_digest(saved_crop) != slot.get("digest"):
                        raise RuntimeError(
                            f"Missing or corrupt evidence for {slot_name} slot of identity {raw_identity_id}."
                        )
                normalized_slot = dict(slot)
                normalized_slot["feature"] = feature
                normalized_slot["evidence_expected"] = evidence_expected
                normalized_gallery[slot_name] = normalized_slot

            record = dict(raw_record)
            record["gallery"] = normalized_gallery
            record["identity_state"] = "confirmed"
            record["role_classified"] = True
            record.setdefault("confirmation_reason", "loaded_gallery")
            record.setdefault("camera_views", {})
            record.setdefault("camera_baselines", {})
            record["member_track_keys"] = set()
            record["pending_member_keys"] = set()
            record["challenged_member_keys"] = set()
            record["appearance_rejected_member_keys"] = set()
            record["pending_member_location_streaks"] = {}
            record["global_reid_checked_track_keys"] = set()
            record.setdefault("location_managed", False)
            record.setdefault("location_match_frames", 0)
            record.setdefault("reid_comparisons", {})
            identities[identity_id] = record

        with self._lock:
            self.identities = identities
            self.next_identity_id = max(
                set(self.identities).union(known_identity_ids),
                default=0,
            ) + 1
        print(f"Loaded {len(identities)} five-slot ReID identities from {source_label}")

    def _persistence_worker_loop(self):
        while True:
            with self._persistence_condition:
                while not self._pending_persistence and not self._persistence_stopping:
                    self._persistence_condition.wait()
                if self._persistence_stopping and not self._pending_persistence:
                    return
                identity_id = next(iter(self._pending_persistence))
                snapshot = self._pending_persistence.pop(identity_id)
                self._persistence_active = True

            try:
                self.persistence_store.save_identity(identity_id, snapshot)
            except Exception as exc:
                print(f"Unable to save ReID identity {identity_id} to backend: {exc}")
            finally:
                with self._persistence_condition:
                    self._persistence_active = False
                    self._persistence_condition.notify_all()

    def _persistence_is_idle(self):
        if self.persistence_store is None:
            return True
        with self._persistence_condition:
            return not self._pending_persistence and not self._persistence_active

    def _wait_for_persistence_idle(self, timeout):
        if self.persistence_store is None:
            return True
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._persistence_condition:
            while self._pending_persistence or self._persistence_active:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._persistence_condition.wait(timeout=remaining)
            return True

    def save_database(self, identity_id=None):
        if self.persistence_store is not None:
            if identity_id is None:
                return
            identity_id = int(identity_id)
            with self._lock:
                record = self.identities.get(identity_id)
                snapshot = copy.deepcopy(record) if record is not None else None
            if snapshot is None:
                return
            if snapshot.get("gallery", {}).get("baseline") is None:
                return
            with self._persistence_condition:
                if self._persistence_stopping:
                    return
                # Keep only the newest unsent snapshot for each identity. This
                # prevents bursts of angle/evidence updates from creating an
                # HTTP backlog while preserving the final state.
                self._pending_persistence[identity_id] = snapshot
                self._persistence_condition.notify()
            return
        if self.db_path is None:
            return
        with self._persistence_lock:
            with self._lock:
                payload = {
                    "schema_version": self.SCHEMA_VERSION,
                    "evidence_enabled": self.evidence_dir is not None,
                    # Provisional records intentionally stay in memory.  They
                    # do not yet have a valid baseline and must not enter the
                    # strict persisted five-slot schema.
                    "identities": {
                        saved_id: record
                        for saved_id, record in self.identities.items()
                        if record.get("gallery", {}).get("baseline") is not None
                    },
                }
                serialized = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
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

    def _evidence_output_path(self, identity_id, slot_name, frame_index, camera_id):
        identity_id = int(identity_id)
        folder_name = (
            f"Temporary_{abs(identity_id):04d}"
            if identity_id < 0
            else f"Master_{identity_id:04d}"
        )
        master_dir = self.evidence_dir / folder_name
        camera_label = "camera" if camera_id is None else str(camera_id).replace("/", "_").replace("\\", "_")
        return master_dir / f"Slot_{slot_name}_{camera_label}_frame_{int(frame_index)}.png"

    def _make_slot(self, identity_id, slot_name, feature, sample, feature_source, feature_space_id):
        crop = sample["crop"]
        normalized_feature = self._normalize_feature(feature)
        if normalized_feature is None:
            raise RuntimeError(f"Refusing to store an invalid {slot_name} feature for ID {identity_id}.")
        camera_id = sample.get("camera_id")
        evidence_expected = bool(
            self.evidence_dir is not None
            and (
                self.evidence_camera_ids is None
                or str(camera_id) in self.evidence_camera_ids
            )
        )
        digest = None
        evidence_task = None
        image_path = None
        if evidence_expected:
            capture_key = (int(identity_id), str(camera_id), int(sample["frame_index"]))
            image_path = self._evidence_capture_paths.get(capture_key)
            if image_path is None:
                output_path = self._evidence_output_path(
                    identity_id,
                    slot_name,
                    sample["frame_index"],
                    camera_id,
                )
                image_path = str(output_path)
                self._evidence_capture_paths[capture_key] = image_path
                evidence_task = {
                    "identity_id": int(identity_id),
                    "slot_name": str(slot_name),
                    # The intake/semantic task already owns this crop. The
                    # evidence path does not make another full-resolution copy.
                    "crop": crop,
                    "output_path": image_path,
                }
        slot = {
            "feature": normalized_feature,
            "feature_source": feature_source,
            "feature_space_id": feature_space_id,
            "feature_dimension": int(normalized_feature.size),
            "image_path": image_path,
            "digest": digest,
            "evidence_expected": evidence_expected,
            "captured_frame": int(sample["frame_index"]),
            "captured_at": float(sample.get("observed_at", time.monotonic())),
            "camera_id": camera_id,
            "sharpness": float(sample.get("sharpness", 0.0)),
            "detection_confidence": sample.get("detection_confidence"),
        }
        return slot, evidence_task

    def _queue_evidence_save(self, evidence_task):
        if evidence_task is not None:
            self._evidence_queue.put_nowait(evidence_task)

    def _send_evidence_task(self, evidence_task):
        process = self._evidence_process
        if (
            process is None
            or process.poll() is not None
            or process.stdin is None
            or process.stdout is None
        ):
            return {"ok": False, "error": "evidence writer process is not running"}
        pickle.dump(evidence_task, process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
        process.stdin.flush()
        result = pickle.load(process.stdout)
        if not isinstance(result, dict):
            return {"ok": False, "error": "invalid response from evidence writer process"}
        return result

    @staticmethod
    def _slots_with_path(record, image_path):
        for slot in record.get("gallery", {}).values():
            if isinstance(slot, dict) and slot.get("image_path") == image_path:
                yield slot
        for slot in record.get("camera_baselines", {}).values():
            if isinstance(slot, dict) and slot.get("image_path") == image_path:
                yield slot
        for camera_gallery in record.get("camera_views", {}).values():
            for slot in camera_gallery.values():
                if isinstance(slot, dict) and slot.get("image_path") == image_path:
                    yield slot

    def _complete_evidence_save_locked(self, evidence_task, digest):
        record = self.identities.get(evidence_task["identity_id"])
        if record is None:
            return
        for slot in self._slots_with_path(record, evidence_task["output_path"]):
            slot["digest"] = digest

    def _rollback_failed_evidence_locked(self, evidence_task):
        identity_id = evidence_task["identity_id"]
        failed_path = evidence_task["output_path"]
        record = self.identities.get(identity_id)
        if record is None:
            return

        gallery = record.get("gallery", {})
        baseline = gallery.get("baseline")
        if isinstance(baseline, dict) and baseline.get("image_path") == failed_path:
            self.identities.pop(identity_id, None)
            for track_key, bound_identity_id in list(self.track_to_identity.items()):
                if bound_identity_id == identity_id:
                    self.track_to_identity.pop(track_key, None)
                    self.track_binding_metadata.pop(track_key, None)
                    self.track_results.pop(track_key, None)
            return

        for slot_name, slot in list(gallery.items()):
            if isinstance(slot, dict) and slot.get("image_path") == failed_path:
                gallery[slot_name] = None
        for camera_id, slot in list(record.get("camera_baselines", {}).items()):
            if isinstance(slot, dict) and slot.get("image_path") == failed_path:
                record["camera_baselines"].pop(camera_id, None)
        for camera_gallery in record.get("camera_views", {}).values():
            for slot_name, slot in list(camera_gallery.items()):
                if isinstance(slot, dict) and slot.get("image_path") == failed_path:
                    camera_gallery[slot_name] = None

    def _evidence_worker_loop(self):
        while True:
            evidence_task = self._evidence_queue.get()
            try:
                if evidence_task is self._stop_token:
                    process = self._evidence_process
                    if process is not None and process.poll() is None and process.stdin is not None:
                        try:
                            pickle.dump(None, process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
                            process.stdin.flush()
                        except (BrokenPipeError, OSError):
                            pass
                    return
                try:
                    result = self._send_evidence_task(evidence_task)
                except (BrokenPipeError, EOFError, OSError, pickle.PickleError) as exc:
                    result = {"ok": False, "error": str(exc)}
                if result.get("ok"):
                    with self._lock:
                        self._complete_evidence_save_locked(
                            evidence_task,
                            result.get("digest"),
                        )
                    self.save_database(evidence_task["identity_id"])
                else:
                    print(
                        f"Unable to save {evidence_task['slot_name']} ReID evidence for "
                        f"ID {evidence_task['identity_id']}: {result.get('error', 'unknown error')}"
                    )
                    with self._lock:
                        self._rollback_failed_evidence_locked(evidence_task)
                    identity_event(
                        "reid_evidence_save_failed",
                        master_id=evidence_task.get("identity_id"),
                        slot_name=evidence_task.get("slot_name"),
                        output_path=evidence_task.get("output_path"),
                    )
            finally:
                self._evidence_queue.task_done()

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
            # TEMP_IDENTITY_DEBUG
            identity_event(
                "physical_match_rejected",
                throttle_key=(identity_id, camera_id, "invalid_observation_time"),
                throttle_seconds=1.0,
                master_id=identity_id,
                camera_id=camera_id,
                reason="invalid_observation_time",
                observed_at=observed_at,
            )
            return False
        normalized_point = None
        if map_point is not None:
            point_array = np.asarray(map_point, dtype=float).reshape(-1)
            if point_array.size != 2 or not np.all(np.isfinite(point_array)):
                # TEMP_IDENTITY_DEBUG
                identity_event(
                    "physical_match_rejected",
                    throttle_key=(identity_id, camera_id, "invalid_map_point"),
                    throttle_seconds=1.0,
                    master_id=identity_id,
                    camera_id=camera_id,
                    reason="invalid_map_point",
                    map_point=map_point,
                )
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
            physical_distance = float(np.linalg.norm(normalized_point - np.asarray(other_point, dtype=float)))
            if physical_distance > self.cross_camera_fusion_distance_cm:
                violation_key = (identity_id, str(camera_id), str(other_camera))
                location_managed = bool(
                    self.identities.get(identity_id, {}).get("location_managed")
                )
                violation_count = int(self.physical_violation_counts.get(violation_key, 0)) + 1
                self.physical_violation_counts[violation_key] = violation_count
                # TEMP_IDENTITY_DEBUG
                identity_event(
                    (
                        "physical_match_warning"
                        if location_managed and violation_count < 3
                        else "physical_match_rejected"
                    ),
                    throttle_key=(identity_id, camera_id, other_camera, "distance"),
                    throttle_seconds=1.0,
                    master_id=identity_id,
                    camera_id=camera_id,
                    other_camera_id=other_camera,
                    reason="distance",
                    map_point=normalized_point,
                    other_map_point=other_point,
                    distance_cm=physical_distance,
                    distance_limit_cm=self.cross_camera_fusion_distance_cm,
                    time_skew_seconds=time_skew,
                    time_skew_limit_seconds=self.cross_camera_max_skew_seconds,
                    consecutive_violations=violation_count,
                    violations_required=3 if location_managed else 1,
                )
                if not location_managed or violation_count >= 3:
                    return False
                continue
            self.physical_violation_counts.pop(
                (identity_id, str(camera_id), str(other_camera)),
                None,
            )
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
        debug_context=None,
        return_rejected=False,
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

        accepted = best_identity is not None and best_distance < self.distance_threshold
        if debug_context is not None:
            # TEMP_IDENTITY_DEBUG: log the best rejected candidate before the
            # normal return value intentionally discards its distance.
            identity_event(
                "reid_match_decision",
                **dict(debug_context),
                camera_id=camera_id,
                map_point=map_point,
                observed_at=observed_at,
                excluded_master_ids=sorted(excluded),
                query_feature_space_id=query_feature_space_id,
                best_master_id=best_identity,
                best_slot=best_slot,
                best_distance=None if best_identity is None else best_distance,
                distance_threshold=self.distance_threshold,
                accepted=accepted,
                rejection_reason=(
                    None
                    if accepted
                    else "no_compatible_gallery"
                    if best_identity is None
                    else "distance_threshold"
                ),
            )
        if not accepted:
            if return_rejected:
                return (
                    best_identity,
                    best_slot,
                    None if best_identity is None else best_distance,
                )
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

    def _target_identity_match_locked(
        self,
        identity_id,
        query_feature,
        feature_space_id,
        debug_context=None,
        return_rejected=False,
    ):
        """Compare with exactly one master, independent of normal exclusions."""

        if identity_id not in self.identities:
            return None, None, None
        excluded = set(self.identities)
        excluded.discard(identity_id)
        return self._matching_identity_locked(
            query_feature,
            query_feature_space_id=feature_space_id,
            excluded_identity_ids=excluded,
            debug_context=debug_context,
            return_rejected=return_rejected,
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

    def _reject_second_visible_owner_locked(
        self,
        identity_id,
        track_keys,
        existing_ids=None,
        event_name="binding_declined",
    ):
        """Return True when binding these keys would give one master two live owners.

        The ReID intake and shadow-handoff paths already refuse to take a
        master that a visible same-camera track owns.  The location-driven
        paths were added later to recover people whose opposite-facing camera
        angles defeat appearance matching, and they bypassed that rule.  A
        second live owner is not merely cosmetic: both tracks then read and
        write the same identity-keyed foot and motion memory, so two people
        render as a single map point and every cross-camera distance check for
        that master starts failing.

        Callers must invoke this before mutating any state, so a refusal is a
        clean no-op.
        """

        if identity_id not in self.identities:
            return False
        keys = list(track_keys)
        batch = set(keys)
        if existing_ids is None:
            existing_ids = [self.track_to_identity.get(key) for key in keys]
        for key, existing_id in zip(keys, existing_ids):
            if existing_id == identity_id:
                # Already this master's owner; re-affirming is not a new claim.
                continue
            conflicting = {
                owner_key
                for owner_key in self._visible_same_camera_identity_owners_locked(
                    identity_id,
                    self._camera_from_key(key),
                    excluded_key=key,
                )
                if owner_key not in batch
            }
            if not conflicting:
                continue
            identity_event(
                event_name,
                master_id=self._public_identity_id(identity_id),
                temporary_group_id=self._temporary_group_token(identity_id),
                track_key=key,
                camera_id=self._camera_from_key(key),
                conflicting_track_keys=sorted(conflicting, key=repr),
                reason="visible_same_camera_owner",
            )
            return True
        return False

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
        # TEMP_IDENTITY_DEBUG
        identity_event(
            "shadow_handoff_committed",
            track_key=key,
            previous_track_key=shadow.get("canonical_key"),
            master_id=identity_id,
            matched_slot=verification.get("matched_slot"),
            distance=verification.get("distance"),
            appearance_confirmed=bool(verification.get("appearance_confirmed", False)),
        )
        self.pending_intake.pop(key, None)
        self.shadow_tracks.pop(key, None)
        return identity_id

    def _next_track_generation_locked(self, key):
        generation = int(self.track_generations.get(key, 0)) + 1
        self.track_generations[key] = generation
        return generation

    def _clear_local_binding_locked(self, key, clear_last_seen=False):
        self._discard_pending_member_evidence_locked(key, "track_binding_cleared")
        identity_id = self.track_to_identity.pop(key, None)
        record = self.identities.get(identity_id)
        if record is not None:
            record.setdefault("member_track_keys", set()).discard(key)
            record.setdefault("pending_member_keys", set()).discard(key)
            record.setdefault("challenged_member_keys", set()).discard(key)
            record.setdefault("appearance_rejected_member_keys", set()).discard(key)
            record.setdefault("pending_member_location_streaks", {}).pop(key, None)
        self.track_results.pop(key, None)
        self.track_binding_metadata.pop(key, None)
        self.pending_intake.pop(key, None)
        self.new_master_holds.pop(key, None)
        self.shadow_tracks.pop(key, None)
        self._next_track_generation_locked(key)
        if clear_last_seen:
            self.track_last_seen.pop(key, None)

    def _release_shadow_locked(self, key, reason="unspecified"):
        """Release a candidate into normal intake and invalidate stale work."""

        shadow = self.shadow_tracks.get(key)
        if shadow is not None:
            # TEMP_IDENTITY_DEBUG
            identity_event(
                "shadow_released",
                track_key=key,
                canonical_track_key=shadow.get("canonical_key"),
                target_master_id=shadow.get("identity_id"),
                reason=reason,
                overlap_frames=shadow.get("overlap_frames"),
                separation_frames=shadow.get("separation_frames"),
            )
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
                        self._release_shadow_locked(key, reason="canonical_master_changed")
                        continue
                    identity_id = mapped_identity_id
                    shadow["identity_id"] = identity_id
                    shadow["provisional"] = False
                elif identity_id is None:
                    # A provisional canonical is useful only while its own
                    # intake is still alive. Its disappearance or cancellation
                    # releases the newer candidate immediately.
                    if canonical_key not in keys or canonical_key not in self.pending_intake:
                        self._release_shadow_locked(
                            key,
                            reason="provisional_canonical_disappeared_or_intake_cancelled",
                        )
                        continue
                elif not shadow.get("verified"):
                    # An unverified target whose local owner was revoked is no
                    # longer a safe handoff candidate.
                    self._release_shadow_locked(
                        key,
                        reason="target_binding_revoked_before_verification",
                    )
                    continue

                if canonical_key in keys and key in current_boxes and canonical_key in current_boxes:
                    overlap_score = self._shadow_overlap_score(
                        current_boxes[key],
                        current_boxes[canonical_key],
                    )
                    if overlap_score is None:
                        shadow["separation_frames"] = int(shadow.get("separation_frames", 0)) + 1
                        if shadow["separation_frames"] >= self.shadow_separation_frames:
                            self._release_shadow_locked(key, reason="separated_from_canonical")
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
                # TEMP_IDENTITY_DEBUG
                identity_event(
                    "shadow_nominated",
                    track_key=key,
                    canonical_track_key=best_canonical,
                    target_master_id=best_identity_id,
                    provisional=best_provisional,
                    overlap_score=best_score,
                    frame_index=frame_index,
                    camera_id=camera_id,
                )

            self.track_boxes.update(current_boxes)
            self.visible_track_keys_by_camera[camera_key] = keys
            if frame_index is not None:
                for key in keys:
                    previous_seen = self.track_last_seen.get(key)
                    if previous_seen is None:
                        continue
                    frame_gap = int(frame_index) - int(previous_seen[0])
                    if frame_gap < 0 or frame_gap > self.ttl_frames:
                        previous_identity_id = self.track_to_identity.get(key)
                        # TEMP_IDENTITY_DEBUG
                        identity_event(
                            "track_binding_reset",
                            track_key=key,
                            master_id=previous_identity_id,
                            camera_id=camera_id,
                            frame_index=frame_index,
                            previous_frame_index=previous_seen[0],
                            frame_gap=frame_gap,
                            ttl_frames=self.ttl_frames,
                            reason="frame_rewind" if frame_gap < 0 else "frame_gap_exceeded_ttl",
                        )
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
            identity_id = self.track_to_identity.get(key)
            return self._public_identity_id(identity_id)

    def temporary_group(self, track_id, camera_id=None):
        """Return the public token for an unnumbered location group."""

        key = self._track_key(track_id, camera_id)
        with self._lock:
            identity_id = self.track_to_identity.get(key)
            if identity_id is None or identity_id >= 0:
                return None
            return self._temporary_group_token(identity_id)

    def hold_new_master_creation(self, left_track_key, right_track_key, hold_token):
        """Prevent unmatched intake results from allocating a new master."""

        track_keys = (left_track_key, right_track_key)
        with self._lock:
            held_keys = []
            for key in track_keys:
                if self.track_to_identity.get(key) is not None:
                    continue
                tokens = self.new_master_holds.setdefault(key, set())
                if hold_token not in tokens:
                    tokens.add(hold_token)
                held_keys.append(key)
            if held_keys:
                identity_event(
                    "new_master_hold_applied",
                    console=False,
                    hold_token=hold_token,
                    held_track_keys=held_keys,
                    pair_track_keys=track_keys,
                )
            return tuple(held_keys)

    def release_new_master_hold(self, left_track_key, right_track_key, hold_token, reason):
        """Release one pair hold and re-arm any completed deferred intake."""

        track_keys = (left_track_key, right_track_key)
        resumed_keys = []
        with self._lock:
            for key in track_keys:
                tokens = self.new_master_holds.get(key)
                if tokens is not None:
                    tokens.discard(hold_token)
                    if not tokens:
                        self.new_master_holds.pop(key, None)
                if key in self.new_master_holds:
                    continue
                state = self.pending_intake.get(key)
                if state is None or not state.pop("deferred_by_new_master_hold", False):
                    continue
                state["submitted"] = False
                state["next_retry_frame"] = int(state.get("last_frame") or 0) + 1
                resumed_keys.append(key)
            identity_event(
                "new_master_hold_released",
                console=False,
                hold_token=hold_token,
                pair_track_keys=track_keys,
                resumed_track_keys=resumed_keys,
                reason=reason,
            )
        return tuple(resumed_keys)

    def lookup_track_key(self, track_key):
        """Return the shared ID for an already-canonical camera/track key."""

        with self._lock:
            return self.track_to_identity.get(track_key)

    def identity_state(self, identity_id):
        with self._lock:
            record = self.identities.get(identity_id)
            return record.get("identity_state") if record is not None else None

    @staticmethod
    def _track_identity_state_locked(record, track_key):
        if track_key in record.get("challenged_member_keys", ()):
            return "challenged"
        if track_key in record.get("pending_member_keys", ()):
            return "provisional"
        return record.get("identity_state", "confirmed")

    def track_identity_state(self, track_key):
        with self._lock:
            identity_id = self.track_to_identity.get(track_key)
            record = self.identities.get(identity_id)
            if record is None:
                return None
            return self._track_identity_state_locked(record, track_key)

    def identity_is_location_managed(self, identity_id):
        with self._lock:
            record = self.identities.get(identity_id)
            return bool(record and record.get("location_managed"))

    def create_provisional_pair(self, left_track_key, right_track_key):
        """Create an internal, unnumbered group for a stable camera pair."""

        track_keys = (left_track_key, right_track_key)
        if any(not isinstance(key, tuple) or len(key) != 2 for key in track_keys):
            return None
        with self._lock:
            mapped = [self.track_to_identity.get(key) for key in track_keys]
            mapped_ids = {identity_id for identity_id in mapped if identity_id is not None}
            if len(mapped_ids) == 1 and mapped[0] == mapped[1]:
                return mapped[0]

            provisional_ids = {
                identity_id
                for identity_id in mapped_ids
                if self.identities.get(identity_id, {}).get("identity_state")
                in ("provisional", "challenged")
            }
            confirmed_ids = mapped_ids - provisional_ids
            if len(confirmed_ids) > 1 or len(provisional_ids) > 1 or (
                confirmed_ids and provisional_ids
            ):
                return None

            attaching_to_confirmed = bool(confirmed_ids)
            created_new_record = False
            if confirmed_ids:
                identity_id = next(iter(confirmed_ids))
            elif provisional_ids:
                identity_id = next(iter(provisional_ids))
            else:
                identity_id = None

            # Single-owner invariant.  Geometry alone must never hand a master
            # to a second live track in a camera where another visible track
            # already owns it; the two would then share every identity-keyed
            # position memory and collapse onto one map point.  Checked before
            # any mutation so a refusal leaves no partial state behind.
            if identity_id is not None and self._reject_second_visible_owner_locked(
                identity_id,
                track_keys,
                mapped,
                "provisional_pair_declined",
            ):
                return None

            if identity_id is None:
                identity_id = self.next_temporary_group_id
                self.next_temporary_group_id -= 1
                self.identities[identity_id] = self._new_record(identity_state="provisional")
                created_new_record = True

            record = self.identities[identity_id]
            record["location_managed"] = True
            if attaching_to_confirmed:
                established_baseline = record.get("gallery", {}).get("baseline")
                if established_baseline is not None:
                    record.setdefault("camera_baselines", {})[
                        "__established_master__"
                    ] = dict(established_baseline)
                # Reuse any already-labelled views from the established
                # camera.  The later camera still completes its own intake
                # before the geometric link becomes fully confirmed.
                for slot_name, slot in record.get("gallery", {}).items():
                    if not slot or not slot.get("camera_id"):
                        continue
                    camera_id = str(slot["camera_id"])
                    if slot_name == "baseline":
                        record.setdefault("camera_baselines", {}).setdefault(
                            camera_id,
                            dict(slot),
                        )
                    elif slot_name in REID_SEMANTIC_SLOTS:
                        camera_gallery = record.setdefault("camera_views", {}).setdefault(
                            camera_id,
                            {name: None for name in REID_SEMANTIC_SLOTS},
                        )
                        if camera_gallery.get(slot_name) is None:
                            camera_gallery[slot_name] = dict(slot)
            for key, existing_id in zip(track_keys, mapped):
                if existing_id not in (None, identity_id):
                    return None
                self.track_to_identity[key] = identity_id
                record["member_track_keys"].add(key)
                newly_attached = existing_id is None
                if attaching_to_confirmed and newly_attached:
                    record.setdefault("pending_member_keys", set()).add(key)
                    record.setdefault("pending_member_location_streaks", {})[key] = 0
                    pending_camera = str(key[0])
                    # Evidence for the newly attached local track must be
                    # fresh.  Never confirm it using an older person's crop
                    # left in this camera's historical slots.
                    record.setdefault("camera_baselines", {}).pop(pending_camera, None)
                    record.setdefault("camera_views", {})[pending_camera] = {
                        name: None for name in REID_SEMANTIC_SLOTS
                    }
                state = self.pending_intake.get(key)
                if state is not None:
                    state["provisional_identity_id"] = identity_id
                if created_new_record or newly_attached:
                    metadata = self.track_binding_metadata.setdefault(key, {})
                    metadata.update(
                        {
                            "appearance_confirmed": False,
                            "identity_state": (
                                "provisional"
                                if attaching_to_confirmed
                                else record["identity_state"]
                            ),
                            "confirmation_reason": None,
                            "provisional_intake_complete": False,
                            "temporary_group_id": (
                                f"tmp_{abs(int(identity_id))}"
                                if identity_id < 0
                                else None
                            ),
                        }
                    )

            identity_event(
                (
                    "provisional_member_attached"
                    if attaching_to_confirmed
                    else "provisional_identity_created"
                ),
                temporary_group_id=(
                    f"tmp_{abs(int(identity_id))}" if identity_id < 0 else None
                ),
                master_id=identity_id if identity_id > 0 else None,
                member_track_keys=track_keys,
                reason="repeated_cross_camera_location",
            )
            return identity_id

    def _provisional_global_reid_complete_locked(self, identity_id):
        """Return whether every currently bound provisional member was searched globally."""

        record = self.identities.get(identity_id)
        if record is None:
            return False
        member_keys = {
            key
            for key in record.get("member_track_keys", ())
            if self.track_to_identity.get(key) == identity_id
        }
        checked_keys = set(record.get("global_reid_checked_track_keys", ()))
        return bool(member_keys) and member_keys.issubset(checked_keys)

    @staticmethod
    def _slot_is_better(candidate, existing):
        if candidate is None:
            return False
        if existing is None:
            return True
        return (
            float(candidate.get("sharpness", 0.0)),
            float(candidate.get("detection_confidence") or 0.0),
        ) > (
            float(existing.get("sharpness", 0.0)),
            float(existing.get("detection_confidence") or 0.0),
        )

    def _stage_pending_member_evidence_locked(
        self,
        identity_id,
        track_key,
        camera_id,
        baseline_slot,
        baseline_task,
        semantic_slots,
    ):
        stage = self.pending_member_evidence.setdefault(
            track_key,
            {
                "identity_id": identity_id,
                "camera_id": str(camera_id),
                "baseline": None,
                "baseline_task": None,
                "views": {},
                "view_tasks": {},
            },
        )
        if stage.get("identity_id") != identity_id:
            stage.clear()
            stage.update(
                {
                    "identity_id": identity_id,
                    "camera_id": str(camera_id),
                    "baseline": None,
                    "baseline_task": None,
                    "views": {},
                    "view_tasks": {},
                }
            )
        if self._slot_is_better(baseline_slot, stage.get("baseline")):
            stage["baseline"] = baseline_slot
            stage["baseline_task"] = baseline_task
        for slot_name, (slot, evidence_task) in semantic_slots.items():
            if self._slot_is_better(slot, stage["views"].get(slot_name)):
                stage["views"][slot_name] = slot
                stage["view_tasks"][slot_name] = evidence_task

    def _commit_pending_member_evidence_locked(self, identity_id, track_key):
        stage = self.pending_member_evidence.pop(track_key, None)
        record = self.identities.get(identity_id)
        if stage is None or record is None or stage.get("identity_id") != identity_id:
            return False
        camera_id = str(stage.get("camera_id"))
        baseline = stage.get("baseline")
        camera_baselines = record.setdefault("camera_baselines", {})
        if self._slot_is_better(baseline, camera_baselines.get(camera_id)):
            camera_baselines[camera_id] = baseline
            self._queue_evidence_save(stage.get("baseline_task"))
        camera_gallery = record.setdefault("camera_views", {}).setdefault(
            camera_id,
            {slot_name: None for slot_name in REID_SEMANTIC_SLOTS},
        )
        for slot_name, slot in stage.get("views", {}).items():
            if self._slot_is_better(slot, camera_gallery.get(slot_name)):
                camera_gallery[slot_name] = slot
                self._queue_evidence_save(stage.get("view_tasks", {}).get(slot_name))
        identity_event(
            "pending_member_evidence_committed",
            master_id=identity_id,
            track_key=track_key,
            camera_id=camera_id,
            stored_orientations=sorted(stage.get("views", {})),
        )
        return True

    @staticmethod
    def _defer_provisional_evidence_locked(record, evidence_task):
        """Hold an unpromoted group's crop until the group is proven real."""

        if record is None or evidence_task is None:
            return
        record.setdefault("deferred_evidence_tasks", []).append(evidence_task)

    def _flush_deferred_evidence_locked(self, record, reason, final_identity_id=None):
        """Write a now-trusted group's held crops to the evidence folder.

        A group is numbered only at promotion, so its held tasks still address
        the ``Temporary_NNNN`` folder.  Re-address them to the master's folder
        and keep the slots' ``image_path`` in step, otherwise the saved digest
        would never find its slot.
        """

        if record is None:
            return 0
        tasks = record.pop("deferred_evidence_tasks", [])
        for evidence_task in tasks:
            if (
                final_identity_id is not None
                and int(evidence_task.get("identity_id", 0)) != int(final_identity_id)
            ):
                # Match slots on the exact stored string.  Round-tripping it
                # through Path would rewrite the separators and silently miss.
                previous_path = str(evidence_task.get("output_path"))
                folder = (
                    f"Temporary_{abs(int(final_identity_id)):04d}"
                    if int(final_identity_id) < 0
                    else f"Master_{int(final_identity_id):04d}"
                )
                parsed = Path(previous_path)
                new_path = str(parsed.parent.parent / folder / parsed.name)
                for slot in self._slots_with_path(record, previous_path):
                    slot["image_path"] = new_path
                for capture_key, path in list(self._evidence_capture_paths.items()):
                    if path == previous_path:
                        self._evidence_capture_paths[capture_key] = new_path
                evidence_task["identity_id"] = int(final_identity_id)
                evidence_task["output_path"] = new_path
            self._queue_evidence_save(evidence_task)
        if tasks:
            identity_event(
                "deferred_evidence_released",
                console=False,
                master_id=self._public_identity_id(final_identity_id),
                released_count=len(tasks),
                reason=reason,
            )
        return len(tasks)

    def _discard_pending_member_evidence_locked(self, track_key, reason):
        stage = self.pending_member_evidence.pop(track_key, None)
        if stage is not None:
            identity_event(
                "pending_member_evidence_discarded",
                master_id=stage.get("identity_id"),
                track_key=track_key,
                camera_id=stage.get("camera_id"),
                reason=reason,
            )

    def _borderline_match_needs_retry_locked(
        self,
        track_key,
        identity_id,
        matched_slot,
        distance,
        feature_space_id,
        frame_index,
        phase,
    ):
        metadata = self.track_binding_metadata.setdefault(track_key, {})
        if distance is None or distance <= self.strong_match_distance:
            metadata.pop("tentative_reid_match", None)
            return False
        if distance >= self.distance_threshold:
            metadata.pop("tentative_reid_match", None)
            return False

        previous = metadata.get("tentative_reid_match") or {}
        same_candidate = bool(
            previous.get("identity_id") == identity_id
            and previous.get("feature_space_id") == feature_space_id
        )
        confirmations = int(previous.get("confirmations", 0)) + 1 if same_candidate else 1
        if confirmations >= 2:
            metadata.pop("tentative_reid_match", None)
            identity_event(
                "borderline_reid_confirmed",
                track_key=track_key,
                master_id=identity_id,
                matched_slot=matched_slot,
                distance=distance,
                strong_distance_threshold=self.strong_match_distance,
                distance_threshold=self.distance_threshold,
                confirmations=confirmations,
                phase=phase,
            )
            return False

        metadata["tentative_reid_match"] = {
            "identity_id": identity_id,
            "feature_space_id": feature_space_id,
            "matched_slot": matched_slot,
            "distance": float(distance),
            "confirmations": confirmations,
        }
        state = self.pending_intake.get(track_key)
        if state is not None:
            retry_started_at = (
                state["samples"][-1].get("observed_at", time.monotonic())
                if state.get("samples")
                else time.monotonic()
            )
            state["submitted"] = False
            state["samples"] = []
            state["first_seen"] = float(retry_started_at)
            state["next_retry_frame"] = int(frame_index) + self.intake_retry_frames
            state["generation"] = self._next_track_generation_locked(track_key)
        identity_event(
            "borderline_reid_deferred",
            track_key=track_key,
            candidate_master_id=identity_id,
            matched_slot=matched_slot,
            distance=distance,
            strong_distance_threshold=self.strong_match_distance,
            distance_threshold=self.distance_threshold,
            confirmations=confirmations,
            confirmations_required=2,
            phase=phase,
        )
        return True

    def _merge_provisional_into_confirmed_locked(
        self,
        provisional_identity_id,
        target_identity_id,
        matched_track_key,
        matched_slot,
        distance,
        feature_source,
        feature_space_id,
    ):
        """Move a location-paired provisional group onto an existing master."""

        provisional = self.identities.get(provisional_identity_id)
        target = self.identities.get(target_identity_id)
        if (
            provisional is None
            or target is None
            or provisional_identity_id == target_identity_id
            or provisional.get("identity_state") not in ("provisional", "challenged")
            or target.get("identity_state", "confirmed") != "confirmed"
        ):
            return None

        member_keys = {
            key
            for key in provisional.get("member_track_keys", ())
            if self.track_to_identity.get(key) == provisional_identity_id
        }
        if matched_track_key not in member_keys:
            return None

        # The per-key loop below only retires *non-visible* same-camera owners,
        # so without this a merge could hand the target master to a second live
        # track while the first one is still on screen.  Refuse before touching
        # the target record; the caller then leaves the provisional group alone
        # and it can merge later, once the camera has one owner again.
        if self._reject_second_visible_owner_locked(
            target_identity_id,
            sorted(member_keys, key=repr),
            [provisional_identity_id] * len(member_keys),
            "provisional_merge_declined",
        ):
            # Remember which master this group really belongs to.  Declining is
            # not enough on its own: the group would otherwise reach its stable
            # location streak and be promoted to a brand-new master, so the
            # refusal would manufacture the very duplicate it exists to prevent.
            provisional["merge_blocked_by_master"] = target_identity_id
            return None

        target["location_managed"] = True
        target["last_member_confirmation_reason"] = "global_reid"
        target["last_seen_monotonic"] = max(
            float(target.get("last_seen_monotonic", 0.0)),
            float(provisional.get("last_seen_monotonic", 0.0)),
        )
        target["hits"] = int(target.get("hits", 0)) + int(provisional.get("hits", 0))

        target_baselines = target.setdefault("camera_baselines", {})
        for camera_id, slot in provisional.get("camera_baselines", {}).items():
            existing = target_baselines.get(camera_id)
            if self._slot_is_better(slot, existing):
                target_baselines[camera_id] = dict(slot)

        target_views = target.setdefault("camera_views", {})
        for camera_id, camera_gallery in provisional.get("camera_views", {}).items():
            destination = target_views.setdefault(
                camera_id,
                {slot_name: None for slot_name in REID_SEMANTIC_SLOTS},
            )
            for slot_name in REID_SEMANTIC_SLOTS:
                slot = camera_gallery.get(slot_name)
                if self._slot_is_better(slot, destination.get(slot_name)):
                    destination[slot_name] = dict(slot)

        baseline_space = (target.get("gallery", {}).get("baseline") or {}).get(
            "feature_space_id"
        )
        for slot_name in REID_SEMANTIC_SLOTS:
            candidates = [
                camera_gallery.get(slot_name)
                for camera_gallery in target_views.values()
                if camera_gallery.get(slot_name) is not None
                and camera_gallery.get(slot_name).get("feature_space_id") == baseline_space
            ]
            if not candidates:
                continue
            best = max(
                candidates,
                key=lambda slot: (
                    float(slot.get("sharpness", 0.0)),
                    float(slot.get("detection_confidence") or 0.0),
                ),
            )
            if self._slot_is_better(best, target.get("gallery", {}).get(slot_name)):
                target["gallery"][slot_name] = dict(best)

        target_observations = self.recent_master_observations.setdefault(
            target_identity_id,
            {},
        )
        for camera_id, observation in self.recent_master_observations.pop(
            provisional_identity_id,
            {},
        ).items():
            existing = target_observations.get(camera_id)
            if existing is None or float(observation.get("observed_at", 0.0)) >= float(
                existing.get("observed_at", 0.0)
            ):
                target_observations[camera_id] = dict(observation)

        for key in sorted(member_keys, key=repr):
            camera_id = self._camera_from_key(key)
            self._clear_nonvisible_same_camera_owners_locked(
                target_identity_id,
                camera_id,
                preserved_key=key,
            )
            self.track_to_identity[key] = target_identity_id
            target.setdefault("member_track_keys", set()).add(key)
            metadata = self.track_binding_metadata.setdefault(key, {})
            is_appearance_match = key == matched_track_key
            metadata.update(
                {
                    "query_feature_space_id": (
                        feature_space_id
                        if is_appearance_match
                        else metadata.get("query_feature_space_id")
                    ),
                    "matched_feature_space_id": baseline_space,
                    "matched_slot": matched_slot if is_appearance_match else None,
                    "distance": distance if is_appearance_match else None,
                    "appearance_confirmed": bool(is_appearance_match),
                    "feature_source": (
                        feature_source
                        if is_appearance_match
                        else metadata.get("feature_source")
                    ),
                    "identity_state": "confirmed",
                    "confirmation_reason": "global_reid",
                    "provisional_intake_complete": True,
                    "temporary_group_id": None,
                }
            )
            self.track_results[key] = {
                "similarity": (
                    1.0 - float(distance)
                    if is_appearance_match and distance is not None
                    else 0.0
                ),
                "reidentified": True,
                "matched_slot": matched_slot if is_appearance_match else None,
            }
            self.pending_intake.pop(key, None)
            self.shadow_tracks.pop(key, None)
            self._next_track_generation_locked(key)

        for pending_key in list(self.pending_semantic_slots):
            if isinstance(pending_key, tuple) and pending_key and pending_key[0] == provisional_identity_id:
                self.pending_semantic_slots.discard(pending_key)
        for attempt_key in list(self.next_semantic_attempt_frame):
            if isinstance(attempt_key, tuple) and attempt_key and attempt_key[0] == provisional_identity_id:
                self.next_semantic_attempt_frame.pop(attempt_key, None)
        for violation_key in list(self.physical_violation_counts):
            if isinstance(violation_key, tuple) and violation_key and violation_key[0] == provisional_identity_id:
                self.physical_violation_counts.pop(violation_key, None)

        # Appearance has just vouched for this group, so its held crops are
        # trustworthy.  The slots were copied onto the target above, so hand
        # the tasks over and flush against the target -- flushing against the
        # provisional would rewrite paths on slots nobody reads again.
        target.setdefault("deferred_evidence_tasks", []).extend(
            provisional.pop("deferred_evidence_tasks", [])
        )
        self._flush_deferred_evidence_locked(
            target,
            "merged_into_existing_master",
            final_identity_id=target_identity_id,
        )
        self.identities.pop(provisional_identity_id, None)
        identity_event(
            "provisional_global_reid_merged",
            provisional_master_id=(
                provisional_identity_id if provisional_identity_id > 0 else None
            ),
            temporary_group_id=(
                f"tmp_{abs(int(provisional_identity_id))}"
                if provisional_identity_id < 0
                else None
            ),
            master_id=target_identity_id,
            matched_track_key=matched_track_key,
            member_track_keys=sorted(member_keys, key=repr),
            matched_slot=matched_slot,
            distance=distance,
            distance_threshold=self.distance_threshold,
            feature_source=feature_source,
            feature_space_id=feature_space_id,
        )
        return target_identity_id

    def _promote_provisional_locked(self, identity_id, reason):
        record = self.identities.get(identity_id)
        if record is None or record.get("identity_state") == "confirmed":
            return None

        if not self._provisional_global_reid_complete_locked(identity_id):
            identity_event(
                "provisional_promotion_deferred",
                throttle_key=(identity_id, "global_reid_incomplete"),
                throttle_seconds=1.0,
                master_id=identity_id,
                reason="global_reid_incomplete",
                checked_track_keys=sorted(
                    record.get("global_reid_checked_track_keys", ()),
                    key=repr,
                ),
                member_track_keys=sorted(
                    record.get("member_track_keys", ()),
                    key=repr,
                ),
            )
            return None

        # A group that appearance says belongs to an existing master, but whose
        # merge is currently blocked by a live same-camera owner, must wait
        # rather than become a second master for the same person.  The check is
        # live rather than sticky, so it clears by itself as soon as the
        # conflicting local track disappears and the merge can be retried.
        blocked_target = record.get("merge_blocked_by_master")
        if blocked_target is not None and blocked_target in self.identities:
            member_keys = sorted(record.get("member_track_keys", ()), key=repr)
            still_blocking = self._reject_second_visible_owner_locked(
                blocked_target,
                member_keys,
                [identity_id] * len(member_keys),
                "provisional_promotion_blocked",
            )
            if still_blocking:
                identity_event(
                    "provisional_promotion_deferred",
                    throttle_key=(identity_id, "merge_target_owner_visible"),
                    throttle_seconds=1.0,
                    master_id=self._public_identity_id(identity_id),
                    temporary_group_id=self._temporary_group_token(identity_id),
                    reason="merge_target_owner_visible",
                    blocked_merge_target_master_id=blocked_target,
                    member_track_keys=member_keys,
                )
                return None
            record.pop("merge_blocked_by_master", None)

        candidates = list(record.get("camera_baselines", {}).values())
        for camera_gallery in record.get("camera_views", {}).values():
            candidates.extend(slot for slot in camera_gallery.values() if slot is not None)
        candidates = [slot for slot in candidates if slot and slot.get("feature") is not None]
        if not candidates:
            return None

        baseline = max(
            candidates,
            key=lambda slot: (
                float(slot.get("sharpness", 0.0)),
                float(slot.get("detection_confidence") or 0.0),
            ),
        )
        baseline_space = baseline.get("feature_space_id")
        record["gallery"]["baseline"] = dict(baseline)
        for slot_name in REID_SEMANTIC_SLOTS:
            semantic_candidates = [
                camera_gallery.get(slot_name)
                for camera_gallery in record.get("camera_views", {}).values()
                if camera_gallery.get(slot_name) is not None
                and camera_gallery.get(slot_name).get("feature_space_id") == baseline_space
            ]
            if semantic_candidates:
                record["gallery"][slot_name] = dict(
                    max(
                        semantic_candidates,
                        key=lambda slot: (
                            float(slot.get("sharpness", 0.0)),
                            float(slot.get("detection_confidence") or 0.0),
                        ),
                    )
                )

        temporary_group_id = identity_id if identity_id < 0 else None
        master_id = identity_id
        if temporary_group_id is not None:
            master_id = self.next_identity_id
            self.next_identity_id += 1
            self.identities[master_id] = record
            self.identities.pop(temporary_group_id, None)
            observations = self.recent_master_observations.pop(temporary_group_id, None)
            if observations is not None:
                self.recent_master_observations[master_id] = observations
            for key, mapped_identity_id in list(self.track_to_identity.items()):
                if mapped_identity_id == temporary_group_id:
                    self.track_to_identity[key] = master_id
            for state in self.pending_intake.values():
                if state.get("provisional_identity_id") == temporary_group_id:
                    state["provisional_identity_id"] = master_id

        record["identity_state"] = "confirmed"
        record["confirmation_reason"] = str(reason)
        # The group is now a real person, so its held crops may be written.
        self._flush_deferred_evidence_locked(
            record, f"promoted_{reason}", final_identity_id=master_id
        )
        for key in record.get("member_track_keys", ()):
            if self.track_to_identity.get(key) != master_id:
                continue
            metadata = self.track_binding_metadata.setdefault(key, {})
            metadata["identity_state"] = "confirmed"
            metadata["confirmation_reason"] = str(reason)
            metadata["appearance_confirmed"] = reason == "same_angle_reid"
            metadata["matched_feature_space_id"] = baseline_space
            metadata["temporary_group_id"] = None

        identity_event(
            "provisional_identity_promoted",
            temporary_group_id=(
                f"tmp_{abs(int(temporary_group_id))}"
                if temporary_group_id is not None
                else None
            ),
            master_id=master_id,
            reason=reason,
            location_match_frames=record.get("location_match_frames", 0),
        )
        return master_id

    def _confirm_pending_members_locked(
        self,
        identity_id,
        reason,
        appearance_confirmed,
        member_keys=None,
    ):
        record = self.identities.get(identity_id)
        if record is None:
            return False
        available_keys = set(record.get("pending_member_keys", ())) | set(
            record.get("challenged_member_keys", ())
        )
        if reason == "stable_location":
            # Standing in the right place for long enough is a fallback for
            # members appearance could not judge -- never an override for ones
            # it already judged and rejected.  Without this, a track that ReID
            # scored 0.458 against this master is still committed, and its crop
            # becomes part of the permanent gallery.
            rejected_keys = available_keys & set(
                record.get("appearance_rejected_member_keys", ())
            )
            if rejected_keys:
                identity_event(
                    "stable_location_confirmation_withheld",
                    throttle_key=(identity_id, "appearance_rejected"),
                    throttle_seconds=1.0,
                    master_id=self._public_identity_id(identity_id),
                    temporary_group_id=self._temporary_group_token(identity_id),
                    withheld_track_keys=sorted(rejected_keys, key=repr),
                    reason="appearance_already_rejected",
                )
            available_keys -= rejected_keys
        available_keys = {
            key
            for key in available_keys
            if self._camera_from_key(key) not in self.visible_track_keys_by_camera
            or key
            in self.visible_track_keys_by_camera.get(self._camera_from_key(key), set())
        }
        pending_keys = (
            available_keys
            if member_keys is None
            else available_keys & set(member_keys)
        )
        if not pending_keys:
            return False
        record.setdefault("pending_member_keys", set()).difference_update(pending_keys)
        record.setdefault("challenged_member_keys", set()).difference_update(pending_keys)
        for key in pending_keys:
            record.setdefault("pending_member_location_streaks", {}).pop(key, None)
            self._commit_pending_member_evidence_locked(identity_id, key)
        record["last_member_confirmation_reason"] = str(reason)
        baseline_space = (record.get("gallery", {}).get("baseline") or {}).get(
            "feature_space_id"
        )
        for slot_name in REID_SEMANTIC_SLOTS:
            candidates = []
            existing_slot = record.get("gallery", {}).get(slot_name)
            if existing_slot is not None:
                candidates.append(existing_slot)
            candidates.extend(
                camera_gallery.get(slot_name)
                for camera_gallery in record.get("camera_views", {}).values()
                if camera_gallery.get(slot_name) is not None
                and camera_gallery.get(slot_name).get("feature_space_id") == baseline_space
            )
            if candidates:
                record["gallery"][slot_name] = dict(
                    max(
                        candidates,
                        key=lambda slot: (
                            float(slot.get("sharpness", 0.0)),
                            float(slot.get("detection_confidence") or 0.0),
                        ),
                    )
                )
        for key in pending_keys:
            if self.track_to_identity.get(key) != identity_id:
                continue
            metadata = self.track_binding_metadata.setdefault(key, {})
            metadata["identity_state"] = "confirmed"
            metadata["confirmation_reason"] = str(reason)
            metadata["appearance_confirmed"] = bool(appearance_confirmed)
            metadata["matched_feature_space_id"] = baseline_space
            metadata["provisional_intake_complete"] = True
        identity_event(
            "provisional_members_confirmed",
            master_id=identity_id,
            member_track_keys=sorted(pending_keys, key=repr),
            reason=reason,
            appearance_confirmed=bool(appearance_confirmed),
        )
        return True

    def _evaluate_provisional_evidence_locked(self, identity_id):
        record = self.identities.get(identity_id)
        if record is None:
            return False
        confirmed_with_pending_members = (
            record.get("identity_state") == "confirmed"
            and bool(
                set(record.get("pending_member_keys", ()))
                | set(record.get("challenged_member_keys", ()))
            )
        )
        if record.get("identity_state") == "confirmed" and not confirmed_with_pending_members:
            return False
        camera_views = {
            str(camera_id): dict(camera_gallery)
            for camera_id, camera_gallery in record.get("camera_views", {}).items()
        }
        if confirmed_with_pending_members:
            for track_key, stage in self.pending_member_evidence.items():
                if (
                    stage.get("identity_id") != identity_id
                    or track_key
                    not in (
                        set(record.get("pending_member_keys", ()))
                        | set(record.get("challenged_member_keys", ()))
                    )
                ):
                    continue
                camera_id = str(stage.get("camera_id"))
                staged_gallery = camera_views.setdefault(
                    camera_id,
                    {slot_name: None for slot_name in REID_SEMANTIC_SLOTS},
                )
                for slot_name, slot in stage.get("views", {}).items():
                    if self._slot_is_better(slot, staged_gallery.get(slot_name)):
                        staged_gallery[slot_name] = slot
        camera_ids = sorted(camera_views)
        distances = []
        for left_index, left_camera in enumerate(camera_ids):
            for right_camera in camera_ids[left_index + 1 :]:
                for slot_name in REID_SEMANTIC_SLOTS:
                    left_slot = camera_views[left_camera].get(slot_name)
                    right_slot = camera_views[right_camera].get(slot_name)
                    if not left_slot or not right_slot:
                        continue
                    if left_slot.get("feature_space_id") != right_slot.get("feature_space_id"):
                        continue
                    left_feature = self._normalize_feature(left_slot.get("feature"))
                    right_feature = self._normalize_feature(right_slot.get("feature"))
                    if left_feature is None or right_feature is None or left_feature.shape != right_feature.shape:
                        continue
                    distance = 1.0 - float(
                        np.dot(
                            np.asarray(left_feature, dtype=np.float64),
                            np.asarray(right_feature, dtype=np.float64),
                        )
                    )
                    comparison_key = f"{left_camera}:{right_camera}:{slot_name}"
                    record["reid_comparisons"][comparison_key] = distance
                    distances.append((distance, left_camera, right_camera, slot_name))

        if not distances:
            return False
        best_distance, left_camera, right_camera, slot_name = min(distances)
        if best_distance < self.distance_threshold:
            identity_event(
                "provisional_reid_confirmed",
                master_id=identity_id,
                left_camera=left_camera,
                right_camera=right_camera,
                orientation=slot_name,
                distance=best_distance,
                distance_threshold=self.distance_threshold,
            )
            if confirmed_with_pending_members:
                evidence_member_keys = {
                    key
                    for key in (
                        set(record.get("pending_member_keys", ()))
                        | set(record.get("challenged_member_keys", ()))
                    )
                    if str(self._camera_from_key(key)) in (left_camera, right_camera)
                }
                return self._confirm_pending_members_locked(
                    identity_id,
                    "same_angle_reid",
                    appearance_confirmed=True,
                    member_keys=evidence_member_keys,
                )
            return self._promote_provisional_locked(identity_id, "same_angle_reid")

        if best_distance >= self.provisional_challenge_distance:
            if confirmed_with_pending_members:
                challenged_keys = {
                    key
                    for key in record.get("pending_member_keys", ())
                    if str(self._camera_from_key(key)) in (left_camera, right_camera)
                }
                record.setdefault("pending_member_keys", set()).difference_update(challenged_keys)
                record.setdefault("challenged_member_keys", set()).update(challenged_keys)
                for key in challenged_keys:
                    self._discard_pending_member_evidence_locked(
                        key,
                        "same_angle_reid_challenged",
                    )
                affected_keys = challenged_keys
            else:
                record["identity_state"] = "challenged"
                affected_keys = set(record.get("member_track_keys", ()))
            for key in affected_keys:
                metadata = self.track_binding_metadata.setdefault(key, {})
                metadata["identity_state"] = "challenged"
            identity_event(
                "provisional_reid_challenged",
                master_id=identity_id,
                left_camera=left_camera,
                right_camera=right_camera,
                orientation=slot_name,
                distance=best_distance,
                challenge_distance=self.provisional_challenge_distance,
            )
        else:
            identity_event(
                "provisional_reid_inconclusive",
                throttle_key=(identity_id, left_camera, right_camera, slot_name),
                throttle_seconds=1.0,
                master_id=identity_id,
                left_camera=left_camera,
                right_camera=right_camera,
                orientation=slot_name,
                distance=best_distance,
                distance_threshold=self.distance_threshold,
                challenge_distance=self.provisional_challenge_distance,
            )
        return False

    def note_location_match(self, identity_id, pair_streak, observations):
        """Record continued geometric agreement and apply the safe fallback."""

        promoted_identity_id = None
        with self._lock:
            record = self.identities.get(identity_id)
            if record is None:
                return None
            record["location_managed"] = True
            # The coordinator's streak is already consecutive.  Store the
            # current value rather than the historical maximum so a camera
            # gap cannot cause an immediate stale promotion later.
            record["location_match_frames"] = int(pair_streak)
            matched_cameras = []
            currently_matched_pending_keys = set()
            for observation in observations:
                camera_id = observation.get("camera_id")
                local_track_id = observation.get("local_track_id")
                if camera_id is None or local_track_id is None:
                    continue
                key = self._track_key(local_track_id, camera_id)
                if self.track_to_identity.get(key) != identity_id:
                    continue
                matched_cameras.append(str(camera_id))
                record["member_track_keys"].add(key)
                if key in record.get("pending_member_keys", ()):
                    currently_matched_pending_keys.add(key)
                    record.setdefault("pending_member_location_streaks", {})[key] = int(
                        pair_streak
                    )
                self._record_master_observation_locked(
                    identity_id,
                    key,
                    observation.get("point"),
                    observation.get("captured_at"),
                )
            for left_camera in matched_cameras:
                for right_camera in matched_cameras:
                    if left_camera != right_camera:
                        self.physical_violation_counts.pop(
                            (identity_id, left_camera, right_camera),
                            None,
                        )
            if (
                record.get("identity_state") == "provisional"
                and record["location_match_frames"] >= self.provisional_location_confirm_frames
                and len(record.get("camera_baselines", {})) >= 2
            ):
                promoted_identity_id = self._promote_provisional_locked(
                    identity_id,
                    "stable_location",
                )
            elif (
                record.get("identity_state") == "confirmed"
                and bool(currently_matched_pending_keys)
            ):
                ready_member_keys = {
                    key
                    for key in currently_matched_pending_keys
                    if int(record.get("pending_member_location_streaks", {}).get(key, 0))
                    >= self.provisional_location_confirm_frames
                    and (
                        self.pending_member_evidence.get(key, {}).get("baseline")
                        is not None
                        or record.get("camera_baselines", {}).get(
                            str(self._camera_from_key(key))
                        )
                        is not None
                    )
                }
                if ready_member_keys:
                    promoted_identity_id = (
                        identity_id
                        if self._confirm_pending_members_locked(
                        identity_id,
                        "stable_location",
                        appearance_confirmed=False,
                        member_keys=ready_member_keys,
                        )
                        else None
                    )
            state = record.get("identity_state")
        if promoted_identity_id is not None:
            self._start_pending_demographics(promoted_identity_id)
            self.save_database(promoted_identity_id)
        return state

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
            identity_state = record.get("identity_state", "confirmed")
            if identity_state in ("provisional", "challenged"):
                gallery_filled = sum(
                    slot is not None
                    for camera_gallery in record.get("camera_views", {}).values()
                    for slot in camera_gallery.values()
                )
                gallery_total = 2 * len(REID_SEMANTIC_SLOTS)
            else:
                gallery_filled = sum(
                    record.get("gallery", {}).get(slot) is not None
                    for slot in REID_GALLERY_SLOTS
                )
                gallery_total = len(REID_GALLERY_SLOTS)
            return {
                "identity_state": identity_state,
                "confirmation_reason": record.get("confirmation_reason"),
                "role": record.get("role", "evacuee"),
                "age": record.get("age", "Unknown"),
                "gender": record.get("gender", "Unknown"),
                "gallery_filled": gallery_filled,
                "gallery_total": gallery_total,
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
            identity_state = record.get("identity_state", "confirmed")
            camera_key = self._camera_from_key(key)
            camera_specific = (
                identity_state in ("provisional", "challenged")
                or bool(record.get("location_managed"))
            )
            if camera_specific:
                camera_gallery = record.get("camera_views", {}).get(camera_key, {})
                gallery_complete = all(camera_gallery.get(slot) is not None for slot in REID_SEMANTIC_SLOTS)
            else:
                gallery_complete = all(
                    record.get("gallery", {}).get(slot) is not None
                    for slot in REID_SEMANTIC_SLOTS
                )
            if gallery_complete:
                return False
            semantic_clock_key = (identity_id, camera_key)
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
        body_complete=None,
    ):
        record = self.identities.get(identity_id)
        if record is None:
            return
        identity_state = record.get("identity_state", "confirmed")
        camera_id = self._camera_from_key(key)
        camera_specific = (
            identity_state in ("provisional", "challenged")
            or bool(record.get("location_managed"))
        )
        gallery = (
            record.setdefault("camera_views", {}).setdefault(
                camera_id,
                {slot: None for slot in REID_SEMANTIC_SLOTS},
            )
            if camera_specific
            else record.get("gallery", {})
        )
        if all(gallery.get(slot) is not None for slot in REID_SEMANTIC_SLOTS):
            return
        semantic_clock_key = (identity_id, camera_id)
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
            or (camera_specific and body_complete is not True)
        ):
            self.next_semantic_attempt_frame[semantic_clock_key] = int(frame_index) + self.semantic_retry_frames
            return
        pending_key = (
            (identity_id, camera_id, orientation)
            if camera_specific
            else (identity_id, orientation)
        )
        if gallery.get(orientation) is not None or pending_key in self.pending_semantic_slots:
            self.next_semantic_attempt_frame[semantic_clock_key] = int(frame_index) + self.semantic_retry_frames
            return

        task = {
            "type": "provisional_semantic" if camera_specific else "semantic",
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
            self.pending_semantic_slots.add(pending_key)
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
        intake_body_complete=None,
        intake_missing_regions=None,
        intake_body_details=None,
        intake_detection_box=None,
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
                    previous_identity_id = self.track_to_identity.get(key)
                    # TEMP_IDENTITY_DEBUG
                    identity_event(
                        "track_binding_reset",
                        track_key=key,
                        master_id=previous_identity_id,
                        camera_id=camera_id,
                        frame_index=frame_index,
                        previous_frame_index=previous_seen[0],
                        frame_gap=frame_gap,
                        ttl_frames=self.ttl_frames,
                        reason="frame_rewind" if frame_gap < 0 else "frame_gap_exceeded_ttl",
                    )
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
                        self._release_shadow_locked(key, reason="canonical_master_changed_during_assign")
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
                    self._release_shadow_locked(key, reason="provisional_canonical_no_longer_pending")
                    shadow = None
                    handoff_from_key = None
                elif not shadow.get("verified"):
                    self._release_shadow_locked(key, reason="unverified_target_binding_missing")
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
            provisional_identity_id = None
            if identity_id is not None and identity_id in self.identities:
                record = self.identities[identity_id]
                identity_state = record.get("identity_state", "confirmed")
                track_identity_state = self._track_identity_state_locked(record, key)
                if track_identity_state in ("provisional", "challenged"):
                    if not self._physical_match_allowed_locked(
                        identity_id,
                        self._camera_from_key(key),
                        map_point,
                        now,
                    ):
                        record.setdefault("member_track_keys", set()).discard(key)
                        record.setdefault("pending_member_keys", set()).discard(key)
                        record.setdefault("challenged_member_keys", set()).discard(key)
                        self.recent_master_observations.get(identity_id, {}).pop(
                            str(self._camera_from_key(key)),
                            None,
                        )
                        self._clear_local_binding_locked(key)
                        identity_event(
                            "provisional_binding_revoked",
                            track_key=key,
                            master_id=identity_id,
                            camera_id=self._camera_from_key(key),
                            frame_index=frame_index,
                            map_point=map_point,
                            observed_at=now,
                            reason="repeated_physical_mismatch",
                        )
                        identity_id = None
                        track_identity_state = None
                    else:
                        provisional_identity_id = identity_id
                if provisional_identity_id is not None:
                    record["hits"] = int(record.get("hits", 0)) + 1
                    record["last_seen_monotonic"] = now
                    record.setdefault("member_track_keys", set()).add(key)
                    self._record_master_observation_locked(identity_id, key, map_point, now)
                    metadata = self.track_binding_metadata.setdefault(key, {})
                    metadata["identity_state"] = track_identity_state
                    metadata["confirmation_reason"] = record.get("confirmation_reason")
                    metadata["appearance_confirmed"] = False
                    if metadata.get("provisional_intake_complete"):
                        self._schedule_semantic_locked(
                            key,
                            identity_id,
                            crop,
                            frame_index,
                            orientation,
                            detection_confidence,
                            now,
                            body_complete=intake_body_complete,
                        )
                        return self._public_identity_id(identity_id), 1.0, False
                    # Continue through the normal quality-controlled intake,
                    # but its worker will add evidence to this reserved ID
                    # instead of matching/creating an independent master.
                    identity_id = None
                elif identity_id is not None and not self._physical_match_allowed_locked(
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
                    # TEMP_IDENTITY_DEBUG
                    identity_event(
                        "track_binding_revoked",
                        track_key=key,
                        master_id=revoked_identity_id,
                        camera_id=self._camera_from_key(key),
                        map_point=map_point,
                        observed_at=now,
                        frame_index=frame_index,
                        reason="physical_match_rejected",
                    )
                    if self.verbose:
                        print(
                            f"ReID: revoked {key} -> Master {revoked_identity_id} "
                            "after an impossible map jump"
                        )
                elif identity_id is not None:
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
                        body_complete=intake_body_complete,
                    )
                    if result is not None:
                        return identity_id, result["similarity"], result["reidentified"]
                    return identity_id, 1.0, False
            if identity_id is not None:
                # TEMP_IDENTITY_DEBUG
                identity_event(
                    "track_binding_reset",
                    track_key=key,
                    master_id=identity_id,
                    camera_id=camera_id,
                    frame_index=frame_index,
                    reason="master_record_missing",
                )
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
                    "provisional_identity_id": provisional_identity_id,
                }
                self.pending_intake[key] = state
                # TEMP_IDENTITY_DEBUG
                identity_event(
                    "track_intake_started",
                    track_key=key,
                    camera_id=camera_id,
                    frame_index=frame_index,
                    observed_at=now,
                    map_point=map_point,
                    handoff_target_master_id=handoff_identity_id,
                    handoff_from_track_key=handoff_from_key,
                )
            if provisional_identity_id is not None:
                state["provisional_identity_id"] = provisional_identity_id
            if state["submitted"]:
                return self._public_identity_id(provisional_identity_id), 0.0, False
            if now - float(state["first_seen"]) < self.intake_delay_seconds:
                return self._public_identity_id(provisional_identity_id), 0.0, False
            if state["last_frame"] == int(frame_index):
                return self._public_identity_id(provisional_identity_id), 0.0, False

            # A timeout may relax the blur gate, but it must never turn a
            # partial-body image into the permanent ReID baseline.
            if intake_body_complete is False:
                state["last_frame"] = int(frame_index)
                missing_regions = tuple(intake_missing_regions or ())
                previous_rejection = state.get("last_body_rejection")
                should_log = (
                    previous_rejection is None
                    or previous_rejection[1] != missing_regions
                    or int(frame_index) - int(previous_rejection[0]) >= 30
                )
                if should_log:
                    # TEMP_IDENTITY_DEBUG
                    identity_event(
                        "intake_crop_rejected",
                        track_key=key,
                        camera_id=camera_id,
                        frame_index=frame_index,
                        observed_at=now,
                        reason="missing_body_parts",
                        missing_regions=missing_regions,
                    )
                    identity_event(
                        "intake_crop_rejected_detail",
                        console=False,
                        track_key=key,
                        camera_id=camera_id,
                        frame_index=frame_index,
                        generation=state.get("generation"),
                        reason="missing_body_parts",
                        missing_regions=missing_regions,
                        crop_shape=tuple(int(value) for value in crop.shape),
                        detection_box=intake_detection_box,
                        detection_confidence=detection_confidence,
                        orientation=orientation,
                        map_point=map_point,
                        body_details=intake_body_details,
                    )
                    state["last_body_rejection"] = (int(frame_index), missing_regions)
                return self._public_identity_id(provisional_identity_id), 0.0, False
            state.pop("last_body_rejection", None)

            sharpness = image_sharpness(crop)
            timed_out = now - float(state["first_seen"]) >= self.intake_timeout_seconds
            if sharpness <= self.blur_threshold and not timed_out:
                return self._public_identity_id(provisional_identity_id), 0.0, False

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
                    "detection_box": (
                        None
                        if intake_detection_box is None
                        else tuple(float(value) for value in intake_detection_box)
                    ),
                    "orientation": orientation if orientation in REID_SEMANTIC_SLOTS else None,
                    "map_point": None if map_point is None else tuple(map(float, map_point)),
                    "body_complete": intake_body_complete,
                    "body_details": copy.deepcopy(intake_body_details),
                }
            )
            if len(state["samples"]) > self.intake_frames:
                state["samples"] = state["samples"][-self.intake_frames :]
            if len(state["samples"]) < self.intake_frames:
                identity_event(
                    "intake_crop_accepted",
                    console=False,
                    track_key=key,
                    camera_id=camera_id,
                    frame_index=frame_index,
                    generation=state.get("generation"),
                    accepted_sample_count=len(state["samples"]),
                    required_sample_count=self.intake_frames,
                    provisional_identity_id=state.get("provisional_identity_id"),
                    sample=self._sample_debug_summary(state["samples"][-1]),
                )
                return self._public_identity_id(provisional_identity_id), 0.0, False
            identity_event(
                "intake_crop_accepted",
                console=False,
                track_key=key,
                camera_id=camera_id,
                frame_index=frame_index,
                generation=state.get("generation"),
                accepted_sample_count=len(state["samples"]),
                required_sample_count=self.intake_frames,
                provisional_identity_id=state.get("provisional_identity_id"),
                sample=self._sample_debug_summary(state["samples"][-1]),
            )
            if int(frame_index) < int(state.get("next_retry_frame", 0)):
                return self._public_identity_id(provisional_identity_id), 0.0, False

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
                "provisional_identity_id": state.get("provisional_identity_id"),
            }
            if self._queue_task_locked(task):
                state["submitted"] = True
                identity_event(
                    "intake_batch_submitted",
                    console=False,
                    track_key=key,
                    camera_id=camera_id,
                    frame_index=frame_index,
                    generation=state.get("generation"),
                    provisional_identity_id=state.get("provisional_identity_id"),
                    sample_count=len(task["samples"]),
                    samples=[
                        self._sample_debug_summary(sample)
                        for sample in task["samples"]
                    ],
                )
            else:
                state["next_retry_frame"] = int(frame_index) + self.intake_retry_frames
            return self._public_identity_id(provisional_identity_id), 0.0, False

    def _store_provisional_intake_locked(
        self,
        identity_id,
        task,
        samples,
        features,
        valid_indices,
        hero_index,
        feature_source,
        feature_space_id,
        role,
        role_confidence,
    ):
        key = task["track_key"]
        camera_id = str(task.get("camera_id"))
        record = self.identities.get(identity_id)
        track_identity_state = (
            self._track_identity_state_locked(record, key)
            if record is not None
            else None
        )
        if (
            record is None
            or (
                track_identity_state not in ("provisional", "challenged")
                and not (
                    track_identity_state == "confirmed"
                    and record.get("location_managed")
                )
            )
            or self.track_to_identity.get(key) != identity_id
        ):
            return None, None

        if not record.get("role_classified"):
            record["role"] = role
            record["role_confidence"] = float(role_confidence)
            if role != "evacuee":
                record["age"] = "N/A"
                record["gender"] = "N/A"
            record["role_classified"] = True

        new_provisional_record = record.get("identity_state") in (
            "provisional",
            "challenged",
        )
        target_confirmation = not new_provisional_record
        best_semantic_samples = {}
        for index in valid_indices:
            slot_name = samples[index].get("orientation")
            if slot_name not in REID_SEMANTIC_SLOTS:
                continue
            previous = best_semantic_samples.get(slot_name)
            if previous is None or self._quality_score(samples[index]) > self._quality_score(samples[previous]):
                best_semantic_samples[slot_name] = index

        if target_confirmation:
            baseline_slot, baseline_task = self._make_slot(
                identity_id,
                f"baseline_{camera_id}",
                features[hero_index],
                samples[hero_index],
                feature_source,
                feature_space_id,
            )
            staged_semantic_slots = {}
            for slot_name, index in best_semantic_samples.items():
                staged_semantic_slots[slot_name] = self._make_slot(
                    identity_id,
                    f"{camera_id}_{slot_name}",
                    features[index],
                    samples[index],
                    feature_source,
                    feature_space_id,
                )
            self._stage_pending_member_evidence_locked(
                identity_id,
                key,
                camera_id,
                baseline_slot,
                baseline_task,
                staged_semantic_slots,
            )
            identity_event(
                "baseline_selected",
                console=False,
                track_key=key,
                camera_id=camera_id,
                master_id=identity_id,
                temporary_group_id=(
                    f"tmp_{abs(int(identity_id))}" if identity_id < 0 else None
                ),
                slot_name=f"baseline_{camera_id}",
                baseline_state="pending_member_confirmation",
                feature_source=feature_source,
                feature_space_id=feature_space_id,
                selection_rule="maximum_sharpness_times_square_root_area",
                evidence_path=baseline_slot.get("image_path"),
                selected_sample=self._sample_debug_summary(samples[hero_index]),
            )
        else:
            camera_baselines = record.setdefault("camera_baselines", {})
            if camera_id not in camera_baselines:
                slot, evidence_task = self._make_slot(
                    identity_id,
                    f"baseline_{camera_id}",
                    features[hero_index],
                    samples[hero_index],
                    feature_source,
                    feature_space_id,
                )
                camera_baselines[camera_id] = slot
                # An unpromoted group is only a geometric guess: two people who
                # merely walked close together for a few frames share it.  The
                # feature must stay in memory so the group can verify itself,
                # but the PNG is withheld until promotion, so an unverified
                # guess can never put one person's photo in another's folder.
                self._defer_provisional_evidence_locked(record, evidence_task)
                identity_event(
                    "baseline_selected",
                    console=False,
                    track_key=key,
                    camera_id=camera_id,
                    master_id=identity_id if identity_id > 0 else None,
                    temporary_group_id=(
                        f"tmp_{abs(int(identity_id))}" if identity_id < 0 else None
                    ),
                    slot_name=f"baseline_{camera_id}",
                    baseline_state="provisional_camera_baseline",
                    feature_source=feature_source,
                    feature_space_id=feature_space_id,
                    selection_rule="maximum_sharpness_times_square_root_area",
                    evidence_path=slot.get("image_path"),
                    selected_sample=self._sample_debug_summary(samples[hero_index]),
                )

            camera_gallery = record.setdefault("camera_views", {}).setdefault(
                camera_id,
                {slot_name: None for slot_name in REID_SEMANTIC_SLOTS},
            )
            for slot_name, index in best_semantic_samples.items():
                if camera_gallery.get(slot_name) is not None:
                    continue
                slot, evidence_task = self._make_slot(
                    identity_id,
                    f"{camera_id}_{slot_name}",
                    features[index],
                    samples[index],
                    feature_source,
                    feature_space_id,
                )
                camera_gallery[slot_name] = slot
                self._defer_provisional_evidence_locked(record, evidence_task)

        record.setdefault("member_track_keys", set()).add(key)
        record["last_seen_monotonic"] = time.monotonic()
        if (
            role == "evacuee"
            and self.enable_demographics
            and record.get("age") == "Pending"
            and not record.get("pending_demographics_crops")
        ):
            record["pending_demographics_crops"] = [sample["crop"].copy() for sample in samples]
        latest_spatial_sample = next(
            (sample for sample in reversed(samples) if sample.get("map_point") is not None),
            samples[-1],
        )
        self._record_master_observation_locked(
            identity_id,
            key,
            latest_spatial_sample.get("map_point"),
            latest_spatial_sample.get("observed_at", time.monotonic()),
        )

        query_feature = self._normalize_feature(
            np.mean(
                np.asarray(
                    [features[index] for index in valid_indices],
                    dtype=np.float32,
                ),
                axis=0,
            )
        )
        if query_feature is None:
            raise RuntimeError("The provisional intake fingerprint had zero norm.")

        dynamic_exclusions = self._same_camera_active_ids_locked(
            camera_id,
            excluded_track_key=key,
        )
        match_phase = (
            "provisional_global_match"
            if new_provisional_record
            else "location_target_confirmation"
        )
        if new_provisional_record:
            excluded = {
                candidate_id
                for candidate_id, candidate_record in self.identities.items()
                if candidate_id == identity_id
                or candidate_record.get("identity_state", "confirmed") != "confirmed"
            }
        else:
            # This is a new member provisionally attached to an already
            # confirmed master. Its mandatory global check is specifically
            # against that master; never merge the established master record
            # into a different identity because one pending member matched it.
            excluded = set(self.identities)
            excluded.discard(identity_id)
        excluded |= dynamic_exclusions - ({identity_id} if target_confirmation else set())
        submitted_peer_ids = {
            self.track_to_identity[peer_key]
            for peer_key in task.get("same_camera_peer_keys", ())
            if peer_key in self.track_to_identity
        }
        identity_event(
            "reid_match_context",
            phase=match_phase,
            provisional_master_id=self._public_identity_id(identity_id),
            temporary_group_id=self._temporary_group_token(identity_id),
            track_key=key,
            camera_id=camera_id,
            frame_index=task.get("frame_index"),
            generation=task.get("generation"),
            caller_excluded_master_ids=sorted(task.get("excluded_identity_ids", ())),
            dynamic_same_camera_master_ids=sorted(dynamic_exclusions),
            submitted_peer_master_ids=sorted(submitted_peer_ids),
            map_point=latest_spatial_sample.get("map_point"),
            observed_at=latest_spatial_sample.get("observed_at"),
            feature_source=feature_source,
            feature_space_id=feature_space_id,
        )
        debug_context = {
            "phase": match_phase,
            "provisional_master_id": self._public_identity_id(identity_id),
            "temporary_group_id": self._temporary_group_token(identity_id),
            "track_key": key,
            "frame_index": task.get("frame_index"),
            "generation": task.get("generation"),
        }
        if target_confirmation:
            # Location may have attached (or even stable-confirmed) this track
            # while its intake task was waiting in the worker queue. Compare
            # with that latest target directly. A stale submission-time
            # exclusion must never send the task back to the create-ID path.
            matched_identity_id, matched_slot, distance = (
                self._target_identity_match_locked(
                    identity_id,
                    query_feature,
                    feature_space_id,
                    debug_context=debug_context,
                    return_rejected=True,
                )
            )
        else:
            matched_identity_id, matched_slot, distance = self._matching_identity_locked(
                query_feature,
                query_feature_space_id=feature_space_id,
                excluded_identity_ids=excluded,
                camera_id=camera_id,
                map_point=latest_spatial_sample.get("map_point"),
                observed_at=latest_spatial_sample.get("observed_at"),
                debug_context=debug_context,
            )
        target_accepted = bool(
            target_confirmation
            and matched_identity_id == identity_id
            and distance is not None
            and distance < self.distance_threshold
        )
        accepted_existing_match = bool(
            matched_identity_id is not None
            and distance is not None
            and distance < self.distance_threshold
            and (not target_confirmation or target_accepted)
        )
        if accepted_existing_match and self._borderline_match_needs_retry_locked(
            key,
            matched_identity_id,
            matched_slot,
            distance,
            feature_space_id,
            task.get("frame_index", 0),
            match_phase,
        ):
            return None, None
        record.setdefault("global_reid_checked_track_keys", set()).add(key)
        identity_event(
            "provisional_global_reid_checked",
            provisional_master_id=self._public_identity_id(identity_id),
            temporary_group_id=self._temporary_group_token(identity_id),
            track_key=key,
            camera_id=camera_id,
            matched_master_id=matched_identity_id,
            matched_slot=matched_slot,
            distance=distance,
            distance_threshold=self.distance_threshold,
            accepted=(
                target_accepted
                if target_confirmation
                else matched_identity_id is not None
            ),
            ignored_caller_excluded_master_ids=sorted(
                task.get("excluded_identity_ids", ())
            ),
            dynamic_same_camera_master_ids=sorted(dynamic_exclusions),
            feature_source=feature_source,
            feature_space_id=feature_space_id,
        )
        if matched_identity_id is not None and new_provisional_record:
            merged_identity_id = self._merge_provisional_into_confirmed_locked(
                identity_id,
                matched_identity_id,
                key,
                matched_slot,
                distance,
                feature_source,
                feature_space_id,
            )
            if merged_identity_id is not None:
                return None, merged_identity_id

        if target_confirmation:
            if target_accepted:
                self._confirm_pending_members_locked(
                    identity_id,
                    "global_reid",
                    appearance_confirmed=True,
                    member_keys={key},
                )
                record.setdefault("pending_member_keys", set()).discard(key)
                record.setdefault("challenged_member_keys", set()).discard(key)
                record.setdefault("pending_member_location_streaks", {}).pop(key, None)
                record.setdefault("appearance_rejected_member_keys", set()).discard(key)
                target_state = "confirmed"
                target_reason = "global_reid"
                identity_event(
                    "location_assignment_reid_confirmed",
                    master_id=identity_id,
                    track_key=key,
                    camera_id=camera_id,
                    matched_slot=matched_slot,
                    distance=distance,
                    distance_threshold=self.distance_threshold,
                )
            elif distance is not None and distance >= self.provisional_challenge_distance:
                self._discard_pending_member_evidence_locked(
                    key,
                    "global_reid_challenged",
                )
                record.setdefault("pending_member_keys", set()).discard(key)
                record.setdefault("challenged_member_keys", set()).add(key)
                record.setdefault("appearance_rejected_member_keys", set()).add(key)
                target_state = "challenged"
                target_reason = "global_reid_challenged"
                identity_event(
                    "location_assignment_reid_challenged",
                    master_id=identity_id,
                    track_key=key,
                    camera_id=camera_id,
                    matched_slot=matched_slot,
                    distance=distance,
                    challenge_distance=self.provisional_challenge_distance,
                )
            else:
                if distance is not None and distance >= self.distance_threshold:
                    # Between the match and challenge thresholds appearance is
                    # not confident enough to break the binding -- the cameras
                    # face each other, so a cross-view distance here is often
                    # genuine.  But it is confident enough to bar the
                    # stable-location shortcut from committing this crop.
                    record.setdefault("appearance_rejected_member_keys", set()).add(key)
                target_state = self._track_identity_state_locked(record, key)
                target_reason = record.get("confirmation_reason")
                identity_event(
                    "location_assignment_reid_inconclusive",
                    master_id=identity_id,
                    track_key=key,
                    camera_id=camera_id,
                    matched_slot=matched_slot,
                    distance=distance,
                    distance_threshold=self.distance_threshold,
                    challenge_distance=self.provisional_challenge_distance,
                )
        else:
            target_state = track_identity_state
            target_reason = record.get("confirmation_reason")

        self.track_binding_metadata[key] = {
            "query_feature_space_id": feature_space_id,
            "matched_feature_space_id": (
                feature_space_id if target_accepted else None
            ),
            "matched_slot": matched_slot if target_confirmation else None,
            "distance": distance if target_confirmation else None,
            "appearance_confirmed": target_accepted,
            "feature_source": feature_source,
            "identity_state": target_state,
            "confirmation_reason": target_reason,
            "provisional_intake_complete": True,
        }
        self.pending_intake.pop(key, None)
        self.shadow_tracks.pop(key, None)
        identity_event(
            "provisional_track_analyzed",
            track_key=key,
            camera_id=camera_id,
            master_id=self._public_identity_id(identity_id),
            temporary_group_id=self._temporary_group_token(identity_id),
            frame_index=task.get("frame_index"),
            stored_orientations=sorted(best_semantic_samples),
            feature_source=feature_source,
            feature_space_id=feature_space_id,
        )
        if target_confirmation:
            # Accepted, inconclusive, and challenged target checks are all
            # terminal for this queued task. None may fall through to the
            # normal global match/create path and overwrite the location ID.
            return None, identity_id
        promoted = self._evaluate_provisional_evidence_locked(identity_id)
        return (
            promoted,
            promoted,
        )

    def _start_pending_demographics(self, identity_id):
        with self._lock:
            record = self.identities.get(identity_id)
            if record is None or record.get("identity_state") != "confirmed":
                return
            crops = record.pop("pending_demographics_crops", None)
            if not crops or record.get("role") != "evacuee" or not self.enable_demographics:
                return
        self._ensure_demographics_worker()
        try:
            self._demographics_queue.put_nowait(
                {"identity_id": identity_id, "crops": crops}
            )
        except queue.Full:
            with self._lock:
                record = self.identities.get(identity_id)
                if record is not None:
                    record["age"] = "Unknown"
                    record["gender"] = "Unknown"

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
        identity_event(
            "intake_baseline_candidate_selected",
            console=False,
            track_key=task.get("track_key"),
            camera_id=task.get("camera_id"),
            frame_index=task.get("frame_index"),
            generation=task.get("generation"),
            provisional_identity_id=task.get("provisional_identity_id"),
            feature_source=feature_source,
            feature_space_id=feature_space_id,
            selection_rule="maximum_sharpness_times_square_root_area",
            selected_sample_index=hero_index,
            selected_sample=self._sample_debug_summary(samples[hero_index]),
            samples=[self._sample_debug_summary(sample) for sample in samples],
        )
        role_classifier = self._get_role_classifier()
        role, role_confidence = (
            role_classifier.predict(crops[hero_index])
            if role_classifier is not None
            else ("evacuee", 0.0)
        )
        if role in ("cag", "scdf") and role_confidence < self.role_confidence_threshold:
            role = "evacuee"

        provisional_handled = False
        promoted_identity_id = None
        persist_identity_id = None
        provisional_identity_id = task.get("provisional_identity_id")
        with self._lock:
            mapped_identity_id = self.track_to_identity.get(task["track_key"])
            if mapped_identity_id is not None:
                # The live binding is newer than the snapshot captured when
                # this worker task was queued (for example, a provisional ID
                # may already have merged into an established master).
                provisional_identity_id = mapped_identity_id
            provisional_record = self.identities.get(provisional_identity_id)
            provisional_track_state = (
                self._track_identity_state_locked(provisional_record, task["track_key"])
                if provisional_record is not None
                else None
            )
            if (
                provisional_record is not None
                and (
                    provisional_track_state in ("provisional", "challenged")
                    or (
                        provisional_track_state == "confirmed"
                        and provisional_record.get("location_managed")
                    )
                )
                and mapped_identity_id == provisional_identity_id
                and self._intake_task_is_current_locked(task)
            ):
                promoted_identity_id, persist_identity_id = self._store_provisional_intake_locked(
                    provisional_identity_id,
                    task,
                    samples,
                    features,
                    valid_indices,
                    hero_index,
                    feature_source,
                    feature_space_id,
                    role,
                    role_confidence,
                )
                provisional_handled = True
        if provisional_handled:
            if promoted_identity_id is not None:
                self._start_pending_demographics(promoted_identity_id)
            if persist_identity_id is not None:
                self.save_database(persist_identity_id)
            return

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
            # TEMP_IDENTITY_DEBUG
            identity_event(
                "reid_match_context",
                track_key=key,
                camera_id=camera_id,
                frame_index=task.get("frame_index"),
                generation=task.get("generation"),
                caller_excluded_master_ids=sorted(task.get("excluded_identity_ids", ())),
                dynamic_same_camera_master_ids=sorted(dynamic_exclusions),
                submitted_peer_master_ids=sorted(submitted_peer_ids),
                map_point=latest_spatial_sample.get("map_point"),
                observed_at=latest_spatial_sample.get("observed_at"),
                feature_source=feature_source,
                feature_space_id=feature_space_id,
            )
            identity_id = None
            matched_slot = None
            distance = None

            if handoff_identity_id is not None:
                identity_id, matched_slot, distance = self._target_identity_match_locked(
                    handoff_identity_id,
                    query_feature,
                    feature_space_id,
                    debug_context={
                        "phase": "shadow_target_match",
                        "track_key": key,
                        "target_master_id": handoff_identity_id,
                        "frame_index": task.get("frame_index"),
                        "generation": task.get("generation"),
                    },
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
                            # TEMP_IDENTITY_DEBUG
                            identity_event(
                                "shadow_verified",
                                track_key=key,
                                canonical_track_key=shadow.get("canonical_key"),
                                master_id=handoff_identity_id,
                                matched_slot=matched_slot,
                                distance=distance,
                                feature_source=feature_source,
                            )
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
                    # TEMP_IDENTITY_DEBUG
                    identity_event(
                        "shadow_appearance_veto",
                        track_key=key,
                        canonical_track_key=handoff_from_key,
                        target_master_id=handoff_identity_id,
                        reason="target_master_distance_rejected",
                    )
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
                    debug_context={
                        "phase": "normal_intake_match",
                        "track_key": key,
                        "frame_index": task.get("frame_index"),
                        "generation": task.get("generation"),
                    },
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
                    # TEMP_IDENTITY_DEBUG
                    identity_event(
                        "reid_candidate_excluded",
                        track_key=key,
                        candidate_master_id=candidate_identity_id,
                        reason="visible_same_camera_owner",
                        visible_owner_track_keys=visible_owners,
                        camera_id=camera_id,
                    )
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
            if (
                identity_id is not None
                and self._borderline_match_needs_retry_locked(
                    key,
                    identity_id,
                    matched_slot,
                    distance,
                    feature_space_id,
                    task.get("frame_index", 0),
                    "normal_intake_match",
                )
            ):
                return
            reidentified = identity_id is not None
            if identity_id is None:
                hold_tokens = tuple(sorted(map(repr, self.new_master_holds.get(key, ()))))
                if hold_tokens:
                    state = self.pending_intake.get(key)
                    if (
                        state is not None
                        and int(state.get("generation", -1))
                        == int(task.get("generation", -2))
                    ):
                        # The five-crop GPU analysis has completed. Preserve
                        # the intake and wait for the location coordinator to
                        # either create a provisional group or release the
                        # bounded hold. No permanent ID is allocated here.
                        state["deferred_by_new_master_hold"] = True
                        identity_event(
                            "new_master_creation_deferred",
                            track_key=key,
                            camera_id=camera_id,
                            frame_index=task.get("frame_index"),
                            generation=task.get("generation"),
                            hold_tokens=hold_tokens,
                            feature_source=feature_source,
                            feature_space_id=feature_space_id,
                            reason="promising_cross_camera_pair",
                        )
                    return
                identity_id = self.next_identity_id
                baseline_sample = samples[hero_index]
                baseline_slot, baseline_evidence_task = self._make_slot(
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
                self._queue_evidence_save(baseline_evidence_task)
                identity_event(
                    "baseline_selected",
                    console=False,
                    track_key=key,
                    camera_id=camera_id,
                    master_id=identity_id,
                    slot_name="baseline",
                    feature_source=feature_source,
                    feature_space_id=feature_space_id,
                    selection_rule="maximum_sharpness_times_square_root_area",
                    evidence_path=baseline_slot.get("image_path"),
                    selected_sample=self._sample_debug_summary(baseline_sample),
                )
                # TEMP_IDENTITY_DEBUG
                identity_event(
                    "master_created",
                    track_key=key,
                    camera_id=camera_id,
                    master_id=identity_id,
                    frame_index=task.get("frame_index"),
                    generation=task.get("generation"),
                    map_point=latest_spatial_sample.get("map_point"),
                    observed_at=latest_spatial_sample.get("observed_at"),
                    feature_source=feature_source,
                    feature_space_id=feature_space_id,
                    excluded_master_ids=sorted(excluded),
                    reason="no_eligible_master_match",
                )
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
                    slot, evidence_task = self._make_slot(
                        identity_id,
                        slot_name,
                        features[index],
                        samples[index],
                        feature_source,
                        feature_space_id,
                    )
                    record["gallery"][slot_name] = slot
                    self._queue_evidence_save(evidence_task)

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
            # TEMP_IDENTITY_DEBUG
            identity_event(
                "track_bound",
                track_key=key,
                camera_id=camera_id,
                master_id=identity_id,
                frame_index=task.get("frame_index"),
                generation=task.get("generation"),
                reidentified=reidentified,
                matched_slot=matched_slot,
                distance=distance,
                appearance_confirmed=bool(
                    feature_source == "transreid" and feature_space_id == baseline_space
                ),
                feature_source=feature_source,
                handoff_from_track_key=handoff_from_key if handoff_committed else None,
            )
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
            slot, evidence_task = self._make_slot(
                identity_id,
                slot_name,
                feature,
                sample,
                feature_source,
                feature_space_id,
            )
            record["gallery"][slot_name] = slot
            self._queue_evidence_save(evidence_task)
        self.save_database(identity_id)
        if self.verbose:
            print(f"ReID: filled {slot_name} for Master {identity_id}")

    def _process_provisional_semantic_task(self, task):
        sample = task["sample"]
        features, feature_source, feature_space_id = self._extract_aligned_features([sample["crop"]])
        feature = features[0] if features else None
        if feature is None:
            raise RuntimeError("No feature could be extracted for the provisional semantic slot.")

        identity_id = task["identity_id"]
        slot_name = task["slot_name"]
        camera_id = str(sample.get("camera_id"))
        promoted_identity_id = None
        stored_for_confirmed = False
        with self._lock:
            record = self.identities.get(identity_id)
            if record is None or record.get("identity_state") not in (
                "provisional",
                "challenged",
                "confirmed",
            ):
                return
            if self.track_to_identity.get(task["track_key"]) != identity_id:
                return
            track_camera = self._camera_from_key(task["track_key"])
            if (
                track_camera in self.visible_track_keys_by_camera
                and task["track_key"]
                not in self.visible_track_keys_by_camera.get(track_camera, set())
            ):
                return
            track_state = self._track_identity_state_locked(record, task["track_key"])
            pending_target_member = bool(
                record.get("identity_state") == "confirmed"
                and track_state in ("provisional", "challenged")
            )
            if pending_target_member:
                staged_views = self.pending_member_evidence.get(
                    task["track_key"], {}
                ).get("views", {})
                if staged_views.get(slot_name) is not None:
                    return
            else:
                camera_gallery = record.setdefault("camera_views", {}).setdefault(
                    camera_id,
                    {name: None for name in REID_SEMANTIC_SLOTS},
                )
                if camera_gallery.get(slot_name) is not None:
                    return
            slot, evidence_task = self._make_slot(
                identity_id,
                f"{camera_id}_{slot_name}",
                feature,
                sample,
                feature_source,
                feature_space_id,
            )
            if pending_target_member:
                self._stage_pending_member_evidence_locked(
                    identity_id,
                    task["track_key"],
                    camera_id,
                    None,
                    None,
                    {slot_name: (slot, evidence_task)},
                )
            else:
                camera_gallery[slot_name] = slot
                self._queue_evidence_save(evidence_task)
            identity_event(
                "provisional_angle_stored",
                master_id=identity_id if identity_id > 0 else None,
                temporary_group_id=(
                    f"tmp_{abs(int(identity_id))}" if identity_id < 0 else None
                ),
                camera_id=camera_id,
                orientation=slot_name,
                frame_index=sample.get("frame_index"),
            )
            if record.get("identity_state") == "confirmed" and not pending_target_member:
                if record.get("gallery", {}).get(slot_name) is None:
                    baseline_space = (record.get("gallery", {}).get("baseline") or {}).get(
                        "feature_space_id"
                    )
                    if baseline_space == feature_space_id:
                        record["gallery"][slot_name] = dict(camera_gallery[slot_name])
                stored_for_confirmed = True
            else:
                promoted_identity_id = self._evaluate_provisional_evidence_locked(
                    identity_id
                )
        if promoted_identity_id is not None:
            self._start_pending_demographics(promoted_identity_id)
        persistence_identity_id = promoted_identity_id or (
            identity_id if stored_for_confirmed else None
        )
        if persistence_identity_id is not None:
            self.save_database(persistence_identity_id)

    def _process_task(self, task):
        if task["type"] == "intake":
            self._process_intake_task(task)
        elif task["type"] == "semantic":
            self._process_semantic_task(task)
        elif task["type"] == "provisional_semantic":
            self._process_provisional_semantic_task(task)
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
                            # TEMP_IDENTITY_DEBUG
                            identity_event(
                                "intake_task_failed",
                                track_key=task.get("track_key"),
                                camera_id=task.get("camera_id"),
                                frame_index=task.get("frame_index"),
                                generation=task.get("generation"),
                                error=str(exc),
                                failure_count=failure_count,
                                retry_frames=retry_frames,
                                next_retry_frame=state.get("next_retry_frame"),
                            )
                elif isinstance(task, dict) and task.get("type") in ("semantic", "provisional_semantic"):
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
                if isinstance(task, dict) and task.get("type") in ("semantic", "provisional_semantic"):
                    with self._lock:
                        camera_id = self._camera_from_key(task.get("track_key"))
                        pending_key = (
                            (task.get("identity_id"), camera_id, task.get("slot_name"))
                            if task.get("type") == "provisional_semantic"
                            else (task.get("identity_id"), task.get("slot_name"))
                        )
                        self.pending_semantic_slots.discard(pending_key)
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
            if (
                self._task_queue.unfinished_tasks == 0
                and self._demographics_queue.unfinished_tasks == 0
                and self._evidence_queue.unfinished_tasks == 0
                and self._persistence_is_idle()
            ):
                return True
            time.sleep(0.005)
        return (
            self._task_queue.unfinished_tasks == 0
            and self._demographics_queue.unfinished_tasks == 0
            and self._evidence_queue.unfinished_tasks == 0
            and self._persistence_is_idle()
        )

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
        if self._evidence_worker is not None and self._evidence_worker.is_alive():
            self._evidence_queue.put(self._stop_token)
            self._evidence_worker.join(timeout=timeout)
        if self._evidence_process is not None:
            try:
                self._evidence_process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._evidence_process.terminate()
                self._evidence_process.wait(timeout=2.0)
            finally:
                if self._evidence_process.stdin is not None:
                    self._evidence_process.stdin.close()
                if self._evidence_process.stdout is not None:
                    self._evidence_process.stdout.close()
        if self.persistence_store is not None:
            with self._lock:
                identity_ids = list(self.identities)
            for identity_id in identity_ids:
                self.save_database(identity_id)
            if drain:
                self._wait_for_persistence_idle(timeout=timeout)
            with self._persistence_condition:
                if not drain:
                    self._pending_persistence.clear()
                self._persistence_stopping = True
                self._persistence_condition.notify_all()
            if self._persistence_worker is not None and self._persistence_worker.is_alive():
                self._persistence_worker.join(timeout=timeout)
