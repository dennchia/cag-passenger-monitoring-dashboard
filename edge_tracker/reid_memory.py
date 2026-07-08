import pickle
import sys
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
    DEFAULT_REID_EMA_ALPHA,
    DEFAULT_REID_MEMORY_TTL_FRAMES,
    DEFAULT_REID_SIMILARITY_THRESHOLD,
)


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

def crop_person(frame, box):
    clamped = clamp_box_to_frame(box, frame)
    if clamped is None:
        return None
    x1, y1, x2, y2 = clamped
    return frame[y1:y2, x1:x2]

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
        self.fastreid_root = Path(fastreid_root) if fastreid_root else None
        if torch is None:
            print("Warning: torch is not available, TransReID feature extractor disabled.")
            return

        self.device = torch.device(device)
        if self._load_fastreid_model():
            return

        if timm is None:
            print("Warning: FastReID/timm is not available, TransReID feature extractor disabled.")
            return

        self._load_timm_model()

    def is_available(self):
        return self.model is not None

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
                with torch.no_grad():
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
            with torch.no_grad():
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
            with torch.no_grad():
                features = self.model(batched_tensor)

            normalized_features = []
            for feature in features.detach().cpu().numpy().astype(np.float32):
                norm = float(np.linalg.norm(feature))
                if norm > 1e-6:
                    normalized_features.append(feature / norm)
            return normalized_features
        except Exception:
            return [feature for feature in (self.extract(crop) for crop in crops) if feature is not None]

class AppearanceIdentityMemory:
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
        intake_frames=1,
        gallery_update_interval_frames=30,
        evidence_dir=None,
    ):
        """Gallery-based identity memory inspired by the provided v6_transReID_tracking.py.

        - `distance_threshold` is a cosine-distance threshold (lower is closer). If not provided, it's derived from `similarity_threshold`.
        - `morph_threshold` controls in-place gallery morphing.
        - `max_gallery_size` limits stored angle slots per identity.
        - `intake_frames` rapid crops are averaged into one denoised baseline fingerprint for a new local track.
        - `evidence_dir` stores labeled crop snapshots for each accepted gallery slot.
        """
        self.similarity_threshold = similarity_threshold
        self.ttl_frames = ttl_frames
        self.ema_alpha = ema_alpha
        self.next_identity_id = 1
        # identities: id -> { 'gallery': [feature,...], 'last_seen': frame_index, 'hits': int, 'age': optional, 'gender': optional }
        self.identities = {}
        self.track_to_identity = {}
        self.reid_extractor = reid_extractor
        self.verbose = bool(verbose)
        self.db_path = Path(db_path) if db_path else None
        # distance thresholds (cosine distance: 1 - dot)
        if distance_threshold is None:
            # derive distance threshold from similarity_threshold if user passed that
            self.distance_threshold = max(0.01, 1.0 - float(self.similarity_threshold))
        else:
            self.distance_threshold = float(distance_threshold)
        self.morph_threshold = float(morph_threshold)
        self.max_gallery_size = int(max_gallery_size)
        self.intake_frames = max(1, int(intake_frames))
        self.gallery_update_interval_frames = max(1, int(gallery_update_interval_frames))
        self.evidence_dir = Path(evidence_dir) if evidence_dir else None
        self.pending_intake = {}
        self.load_database()

    def load_database(self):
        if self.db_path is None or not self.db_path.exists():
            return

        try:
            with self.db_path.open("rb") as handle:
                loaded = pickle.load(handle)
        except Exception as exc:
            print(f"Unable to load ReID database {self.db_path}: {exc}")
            return

        if not isinstance(loaded, dict):
            print(f"Ignoring ReID database {self.db_path}: expected dictionary.")
            return

        identities = {}
        for identity_id, record in loaded.items():
            try:
                normalized_id = int(identity_id)
            except (TypeError, ValueError):
                continue

            if isinstance(record, list):
                gallery = record
                record = {}
            elif isinstance(record, dict):
                gallery = record.get("gallery", [])
            else:
                continue

            normalized_gallery = []
            for feature in gallery:
                feature_array = np.asarray(feature, dtype=np.float32).ravel()
                norm = float(np.linalg.norm(feature_array))
                if norm > 1e-6:
                    normalized_gallery.append(feature_array / norm)

            if not normalized_gallery:
                continue

            identities[normalized_id] = {
                "gallery": normalized_gallery[: self.max_gallery_size],
                "last_seen": int(record.get("last_seen", 0)) if isinstance(record, dict) else 0,
                "last_gallery_update": int(record.get("last_gallery_update", 0)) if isinstance(record, dict) else 0,
                "hits": int(record.get("hits", 0)) if isinstance(record, dict) else 0,
            }
            if isinstance(record, dict):
                for key in ("age", "gender"):
                    if key in record:
                        identities[normalized_id][key] = record[key]
                if isinstance(record.get("evidence"), dict):
                    identities[normalized_id]["evidence"] = dict(record["evidence"])

        self.identities = identities
        self.next_identity_id = max(self.identities.keys(), default=0) + 1
        print(f"Loaded {len(self.identities)} ReID identities from {self.db_path}")

    def save_database(self):
        if self.db_path is None:
            return

        try:
            with self.db_path.open("wb") as handle:
                pickle.dump(self.identities, handle)
        except Exception as exc:
            print(f"Unable to save ReID database {self.db_path}: {exc}")

    def save_evidence_snapshot(self, identity_id, slot_index, crop, frame_index, label):
        if self.evidence_dir is None or crop is None or crop.size == 0:
            return None

        try:
            person_dir = self.evidence_dir / f"ID_{int(identity_id):04d}"
            person_dir.mkdir(parents=True, exist_ok=True)

            snapshot = crop.copy()
            text = f"ID {identity_id} Slot {slot_index} {label}"
            cv2.rectangle(snapshot, (0, 0), (min(snapshot.shape[1], 420), 32), (0, 0, 0), -1)
            cv2.putText(
                snapshot,
                text,
                (8, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            filename = f"ID_{int(identity_id):04d}_Slot_{int(slot_index)}_{label}_frame_{int(frame_index)}.jpg"
            output_path = person_dir / filename
            cv2.imwrite(str(output_path), snapshot)
            if self.verbose:
                print(f"AppearanceIdentityMemory: saved evidence {output_path}")
            return str(output_path)
        except Exception as exc:
            print(f"Unable to save ReID evidence snapshot: {exc}")
            return None

    def pending_count(self, track_id):
        state = self.pending_intake.get(track_id)
        return len(state["crops"]) if state is not None else 0

    def required_intake_count(self):
        return self.intake_frames

    def extract_feature(self, crop):
        feature = None
        if self.reid_extractor is not None:
            feature = self.reid_extractor.extract(crop)

        if feature is None:
            feature = compute_color_reid_feature(crop)

        if feature is None:
            return None

        norm = float(np.linalg.norm(feature))
        if norm <= 1e-6:
            return None
        return feature / norm

    def extract_denoised_fingerprint(self, crops):
        if not crops:
            return None

        features = []
        if self.reid_extractor is not None and hasattr(self.reid_extractor, "extract_many"):
            features = self.reid_extractor.extract_many(crops)

        if not features:
            features = [self.extract_feature(crop) for crop in crops]
            features = [feature for feature in features if feature is not None]

        if not features:
            return None

        fingerprint = np.mean(np.array(features, dtype=np.float32), axis=0)
        norm = float(np.linalg.norm(fingerprint))
        if norm <= 1e-6:
            return None
        return fingerprint / norm

    def find_matching_identity(self, feature, frame_index):
        absolute_min_distance = float('inf')
        matched_identity = None
        matched_local_min = None

        for identity_id, rec in self.identities.items():
            if int(frame_index) - int(rec.get('last_seen', 0)) > self.ttl_frames:
                continue

            gallery = rec.get('gallery', [])
            if not gallery:
                continue

            local_min = float('inf')
            for saved in gallery:
                d = 1.0 - float(np.dot(feature, saved))
                if d < local_min:
                    local_min = d
            if local_min < absolute_min_distance:
                absolute_min_distance = local_min
                matched_identity = identity_id
                matched_local_min = local_min

        if matched_identity is None or absolute_min_distance >= self.distance_threshold:
            return None, 0.0, None

        return matched_identity, 1.0 - absolute_min_distance, matched_local_min

    def route_gallery_feature(self, identity_id, feature, frame_index, force=False, crop=None):
        rec = self.identities.get(identity_id)
        if rec is None:
            return

        rec['last_seen'] = int(frame_index)
        rec['hits'] = rec.get('hits', 0) + 1

        if not force:
            last_update = int(rec.get('last_gallery_update', 0))
            if int(frame_index) - last_update < self.gallery_update_interval_frames:
                return

        gallery = rec.get('gallery', [])
        if not gallery:
            rec['gallery'] = [feature.copy()]
            rec['last_gallery_update'] = int(frame_index)
            evidence_path = self.save_evidence_snapshot(identity_id, 1, crop, frame_index, "Baseline")
            if evidence_path is not None:
                rec.setdefault("evidence", {})["slot_1"] = evidence_path
            self.save_database()
            return

        local_distances = [1.0 - float(np.dot(feature, saved)) for saved in gallery]
        local_min = float(min(local_distances))
        closest_index = int(np.argmin(local_distances))

        if local_min < self.morph_threshold:
            gallery[closest_index] = 0.9 * gallery[closest_index] + 0.1 * feature
            gallery[closest_index] /= max(1e-6, np.linalg.norm(gallery[closest_index]))
            rec['last_gallery_update'] = int(frame_index)
            evidence_path = self.save_evidence_snapshot(identity_id, closest_index + 1, crop, frame_index, "Refined")
            if evidence_path is not None:
                rec.setdefault("evidence", {})[f"slot_{closest_index + 1}"] = evidence_path
            if self.verbose:
                print(f"AppearanceIdentityMemory: refined ID {identity_id} slot {closest_index} (dist={local_min:.3f})")
            self.save_database()
        elif self.morph_threshold <= local_min <= self.distance_threshold:
            gallery.append(feature.copy())
            if len(gallery) > self.max_gallery_size:
                gallery.pop(0)
            slot_index = len(gallery)
            rec['last_gallery_update'] = int(frame_index)
            evidence_path = self.save_evidence_snapshot(identity_id, slot_index, crop, frame_index, "Angle")
            if evidence_path is not None:
                rec.setdefault("evidence", {})[f"slot_{slot_index}"] = evidence_path
            if self.verbose:
                print(f"AppearanceIdentityMemory: new angle for ID {identity_id} gallery={len(gallery)}/{self.max_gallery_size} (dist={local_min:.3f})")
            self.save_database()

    def assign_fingerprint(self, track_id, feature, frame_index, evidence_crop=None):
        matched_identity, similarity, _ = self.find_matching_identity(feature, frame_index)
        if matched_identity is not None:
            identity_id = matched_identity
            self.route_gallery_feature(identity_id, feature, frame_index, force=True, crop=evidence_crop)
            self.track_to_identity[track_id] = identity_id
            self.pending_intake.pop(track_id, None)
            return identity_id, similarity, True

        identity_id = self.next_identity_id
        self.next_identity_id += 1
        self.identities[identity_id] = {
            'gallery': [feature.copy()],
            'last_seen': int(frame_index),
            'last_gallery_update': int(frame_index),
            'hits': 1,
        }
        evidence_path = self.save_evidence_snapshot(identity_id, 1, evidence_crop, frame_index, "Baseline")
        if evidence_path is not None:
            self.identities[identity_id]["evidence"] = {"slot_1": evidence_path}
        self.track_to_identity[track_id] = identity_id
        self.pending_intake.pop(track_id, None)
        if self.verbose:
            print(f"AppearanceIdentityMemory: created new ID {identity_id} from {self.intake_frames}-crop denoised intake")

        self.save_database()
        return identity_id, 0.0, False

    def assign(self, track_id, crop, frame_index):
        """Assign or lookup an identity for `track_id` given the `crop` at `frame_index`.

        Returns: (identity_id or None, similarity (dot), reidentified_bool)
        """
        if crop is None or crop.size == 0:
            if self.verbose:
                print(f"AppearanceIdentityMemory: no crop for track {track_id}")
            return self.track_to_identity.get(track_id), 0.0, False

        if track_id in self.track_to_identity:
            identity_id = self.track_to_identity[track_id]
            rec = self.identities.get(identity_id)
            feature = self.extract_feature(crop)
            if rec is not None and feature is not None:
                self.route_gallery_feature(identity_id, feature, frame_index, crop=crop)
            elif rec is not None:
                rec['last_seen'] = int(frame_index)
                rec['hits'] = rec.get('hits', 0) + 1
            if self.verbose:
                print(f"AppearanceIdentityMemory: existing mapping track {track_id} -> ID {identity_id}")
            return identity_id, 1.0, False

        if self.intake_frames <= 1:
            feature = self.extract_feature(crop)
            if feature is None:
                if self.verbose:
                    print(f"AppearanceIdentityMemory: no feature for track {track_id}")
                return None, 0.0, False
            return self.assign_fingerprint(track_id, feature, frame_index, evidence_crop=crop)

        state = self.pending_intake.setdefault(track_id, {"crops": [], "last_frame": None})
        if state["last_frame"] != int(frame_index):
            state["crops"].append(crop.copy())
            state["last_frame"] = int(frame_index)

        if len(state["crops"]) < self.intake_frames:
            return None, 0.0, False

        fingerprint = self.extract_denoised_fingerprint(state["crops"][: self.intake_frames])
        if fingerprint is None:
            if self.verbose:
                print(f"AppearanceIdentityMemory: no fingerprint for track {track_id}")
            return None, 0.0, False

        evidence_crop = state["crops"][min(len(state["crops"]) - 1, self.intake_frames // 2)]
        return self.assign_fingerprint(track_id, fingerprint, frame_index, evidence_crop=evidence_crop)

