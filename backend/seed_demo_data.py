from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

import crud
from config import settings
from database import SessionLocal, init_db
from evacuees import repository as evacuee_repository
from evacuees.storage import (
    PUBLIC_UPLOAD_PREFIX as EVACUEE_UPLOAD_PREFIX,
    delete_gallery_image,
    ensure_upload_dir as ensure_evacuee_upload_dir,
)
from models import EvacueeIdentityUpsert, MetricLogCreate, SystemAlertCreate
from observation_storage import PUBLIC_UPLOAD_PREFIX, clear_observation_images, ensure_upload_dir


DEMO_RUN_ID = "demo_assistance_001"

OBSERVATIONS = [
    {"camera_id": "cam_1", "track_id": "T-001", "age": 8, "gender": "female", "age_confidence": 0.82, "gender_confidence": 0.91, "tone": (92, 143, 240)},
    {"camera_id": "cam_1", "track_id": "T-002", "age": 14, "gender": "male", "age_confidence": 0.76, "gender_confidence": 0.88, "tone": (45, 212, 191)},
    {"camera_id": "cam_2", "track_id": "T-003", "age": 31, "gender": "female", "age_confidence": 0.89, "gender_confidence": 0.93, "tone": (244, 114, 182)},
    {"camera_id": "cam_2", "track_id": "T-004", "age": 42, "gender": "male", "age_confidence": 0.86, "gender_confidence": 0.95, "tone": (251, 191, 36)},
    {"camera_id": "cam_1", "track_id": "T-005", "age": 67, "gender": "female", "age_confidence": 0.71, "gender_confidence": 0.84, "tone": (168, 85, 247)},
    {"camera_id": "cam_2", "track_id": "T-006", "age": 73, "gender": "unknown", "age_confidence": 0.68, "gender_confidence": 0.52, "tone": (148, 163, 184)},
]

GALLERY_VIEWS = [
    ("baseline", "front", "back", "left_side", "right_side"),
    ("baseline", "back", "left_side", "right_side"),
    ("baseline",),
    ("baseline", "front"),
    ("baseline", "right_side"),
    ("baseline", "back", "left_side"),
]


def make_crop(
    path: Path,
    *,
    age: int,
    gender: str,
    track_id: str,
    tone: tuple[int, int, int],
    view_type: str = "baseline",
) -> None:
    frame = np.zeros((360, 280, 3), dtype=np.uint8)
    frame[:, :] = (15, 23, 42)

    color = (int(tone[0]), int(tone[1]), int(tone[2]))
    cv2.rectangle(frame, (38, 42), (242, 318), color, -1)
    cv2.rectangle(frame, (56, 64), (224, 296), (248, 250, 252), 3)
    cv2.circle(frame, (140, 132), 44, (255, 255, 255), -1)
    cv2.rectangle(frame, (88, 190), (192, 282), (255, 255, 255), -1)
    cv2.putText(frame, track_id, (68, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (226, 232, 240), 2)
    cv2.putText(frame, view_type.replace("_", " ").title(), (64, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (226, 232, 240), 2)
    cv2.putText(frame, f"Age {age}", (70, 338), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (226, 232, 240), 2)
    cv2.putText(frame, gender.title(), (152, 338), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (226, 232, 240), 2)
    cv2.imwrite(str(path), frame)


def seed(reset_observations: bool) -> None:
    init_db()
    ensure_upload_dir()
    ensure_evacuee_upload_dir()

    if reset_observations:
        with SessionLocal() as db:
            deleted_rows = crud.clear_passenger_observations(db)
            deleted_identities, gallery_image_paths = evacuee_repository.clear_identities(db)
        deleted_images = clear_observation_images()
        deleted_gallery_images = sum(1 for path in gallery_image_paths if delete_gallery_image(path))
        print(
            f"Cleared {deleted_rows} observation rows, {deleted_identities} identities, "
            f"{deleted_images} legacy crops, and {deleted_gallery_images} gallery images."
        )

    upload_dir = settings.observation_upload_path
    now = datetime.now(timezone.utc)

    with SessionLocal() as db:
        for index, item in enumerate(OBSERVATIONS):
            filename = f"seed_{item['track_id'].lower().replace('-', '_')}.jpg"
            image_path = upload_dir / filename
            make_crop(
                image_path,
                age=item["age"],
                gender=item["gender"],
                track_id=item["track_id"],
                tone=item["tone"],
            )
            crud.create_passenger_observation(
                db,
                timestamp=now - timedelta(minutes=index * 2),
                run_id=DEMO_RUN_ID,
                camera_id=item["camera_id"],
                track_id=item["track_id"],
                age=float(item["age"]),
                gender=item["gender"],
                age_confidence=float(item["age_confidence"]),
                gender_confidence=float(item["gender_confidence"]),
                image_path=str(image_path),
                image_url=f"{PUBLIC_UPLOAD_PREFIX}/{filename}",
            )

            master_identity_id = index + 1
            identity = evacuee_repository.upsert_identity(
                db,
                run_id=DEMO_RUN_ID,
                master_identity_id=master_identity_id,
                payload=EvacueeIdentityUpsert(
                    age=float(item["age"]),
                    gender=item["gender"],
                    last_camera_id=item["camera_id"],
                    first_seen_at=now - timedelta(minutes=index * 2 + 1),
                    last_seen_at=now - timedelta(minutes=index * 2),
                    current_status="inside",
                ),
            )
            identity_dir = settings.evacuee_upload_path / DEMO_RUN_ID / f"master_{master_identity_id:04d}"
            identity_dir.mkdir(parents=True, exist_ok=True)
            for view_index, view_type in enumerate(GALLERY_VIEWS[index]):
                gallery_filename = f"seed_{view_type}.jpg"
                gallery_path = identity_dir / gallery_filename
                make_crop(
                    gallery_path,
                    age=item["age"],
                    gender=item["gender"],
                    track_id=f"ID {master_identity_id}",
                    tone=item["tone"],
                    view_type=view_type,
                )
                evacuee_repository.upsert_gallery_view(
                    db,
                    identity=identity,
                    view_type=view_type,
                    image_path=str(gallery_path),
                    image_url=(
                        f"{EVACUEE_UPLOAD_PREFIX}/{DEMO_RUN_ID}/"
                        f"master_{master_identity_id:04d}/{gallery_filename}"
                    ),
                    feature_blob=None,
                    feature_dimension=None,
                    feature_space_id=None,
                    feature_source=None,
                    digest=None,
                    captured_at=now - timedelta(minutes=index * 2, seconds=view_index),
                    captured_frame=view_index,
                    camera_id=item["camera_id"],
                    sharpness=100.0 + view_index * 15,
                    detection_confidence=0.9,
                )

        for offset, count in enumerate([82, 96, 113, 121, 128]):
            cam_1_count = count // 2 + offset
            cam_2_count = count - cam_1_count
            crud.create_metric_log(
                db,
                MetricLogCreate(
                    timestamp=now - timedelta(minutes=(4 - offset) * 3),
                    run_id=DEMO_RUN_ID,
                    passenger_count=count,
                    zone_counts={"cam_1": cam_1_count, "cam_2": cam_2_count},
                    camera_online_count=2,
                ),
            )

        for severity, message, minutes_ago in [
            ("info", "Demo assistance observations seeded.", 8),
            ("warning", "Zone B crowding trend rising.", 4),
            ("critical", "Capacity threshold demo alert active.", 1),
        ]:
            crud.create_system_alert(
                db,
                SystemAlertCreate(
                    timestamp=now - timedelta(minutes=minutes_ago),
                    run_id=DEMO_RUN_ID,
                    severity=severity,
                    message=message,
                ),
            )

    print(f"Seeded {len(OBSERVATIONS)} identities and legacy observations, 5 metrics, and 3 alerts.")
    print(f"Run ID: {DEMO_RUN_ID}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed CAG dashboard demo data.")
    parser.add_argument(
        "--reset-observations",
        action="store_true",
        help="Clear saved passenger observations and crop images before seeding.",
    )
    args = parser.parse_args()
    seed(reset_observations=args.reset_observations)


if __name__ == "__main__":
    main()
