#!/usr/bin/env python3
"""
Raw fingerprint image storage.

Enrollment used to persist only SIFT descriptors, which meant that changing the
matching algorithm forced every member to re-enroll.  Keeping the raw sensor
image means any future engine (NBIS, a commercial SDK, a CNN) can be rebuilt
from what is already on disk.

Layout (under fingerprint_storage/images/):
    enrolled/<client_id>/f<finger>_s<sample>_<timestamp>.png
    queries/<YYYYMMDD>/<HHMMSS>_<source>_<outcome>.png

Images are written as lossless PNG - JPEG compression artefacts damage ridge
structure and cost minutiae.

NOTE: these files are biometric data. They live next to the database, not in
it, so back them up together and keep the directory access-controlled.
"""

import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

BASE_DIR = Path(os.environ.get(
    "FINGERPRINT_IMAGE_DIR",
    str(Path(__file__).resolve().parent / "fingerprint_storage" / "images"),
))

ENROLLED_DIR = BASE_DIR / "enrolled"
QUERIES_DIR = BASE_DIR / "queries"

# Enrollment images are what make an algorithm migration possible - on by default.
SAVE_ENROLLMENT_IMAGES = os.environ.get("FINGERPRINT_SAVE_ENROLLMENT_IMAGES", "1") not in ("0", "false", "False")

# Query images are useful for tuning thresholds, but they pile up and are
# biometric data of everyone who touches the sensor - off by default.
SAVE_QUERY_IMAGES = os.environ.get("FINGERPRINT_SAVE_QUERY_IMAGES", "0") not in ("0", "false", "False")

# Delete query images older than this many days (0 = keep forever).
QUERY_RETENTION_DAYS = int(os.environ.get("FINGERPRINT_QUERY_RETENTION_DAYS", "30"))

_SAFE = re.compile(r"[^A-Za-z0-9_-]")


def _safe(text: str) -> str:
    return _SAFE.sub("_", str(text))[:40]


def _write(image: np.ndarray, path: Path) -> Optional[str]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            print(f"⚠️  Could not write fingerprint image to {path}")
            return None
        return str(path)
    except Exception as exc:  # noqa: BLE001 - storage must never break a scan
        print(f"⚠️  Error saving fingerprint image: {exc}")
        return None


def save_enrollment_image(image: np.ndarray, client_id, finger_index: int, sample_number: int) -> Optional[str]:
    """Persist one enrollment capture. Returns the path, or None if not saved."""
    if image is None or not SAVE_ENROLLMENT_IMAGES:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ENROLLED_DIR / _safe(client_id) / f"f{int(finger_index)}_s{int(sample_number)}_{stamp}.png"
    return _write(image, path)


def save_query_image(image: np.ndarray, source: str = "unknown", outcome: str = "") -> Optional[str]:
    """Persist one identification capture (off unless FINGERPRINT_SAVE_QUERY_IMAGES=1)."""
    if image is None or not SAVE_QUERY_IMAGES:
        return None
    now = datetime.now()
    name = f"{now.strftime('%H%M%S_%f')[:13]}_{_safe(source)}"
    if outcome:
        name += f"_{_safe(outcome)}"
    path = QUERIES_DIR / now.strftime("%Y%m%d") / f"{name}.png"
    return _write(image, path)


def load_image(path: str) -> Optional[np.ndarray]:
    """Read a stored fingerprint image back as grayscale."""
    if not path:
        return None
    try:
        if not os.path.isfile(path):
            return None
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Error loading fingerprint image {path}: {exc}")
        return None


def client_enrollment_images(client_id) -> list:
    """All stored enrollment images for one client, oldest first."""
    directory = ENROLLED_DIR / _safe(client_id)
    if not directory.is_dir():
        return []
    return sorted(str(p) for p in directory.glob("*.png"))


def prune_query_images(days: int = QUERY_RETENTION_DAYS) -> int:
    """Delete query images older than `days`. Returns how many were removed."""
    if days <= 0 or not QUERIES_DIR.is_dir():
        return 0
    cutoff = time.time() - days * 86400
    removed = 0
    for day_dir in QUERIES_DIR.iterdir():
        if not day_dir.is_dir():
            continue
        for image in day_dir.glob("*.png"):
            try:
                if image.stat().st_mtime < cutoff:
                    image.unlink()
                    removed += 1
            except OSError:
                continue
        try:
            next(day_dir.iterdir())
        except StopIteration:
            day_dir.rmdir()
        except OSError:
            pass
    if removed:
        print(f"🧹 Removed {removed} query images older than {days} days")
    return removed


def storage_report() -> dict:
    """Counts and disk usage, for logging at startup."""
    def measure(directory: Path) -> tuple:
        if not directory.is_dir():
            return 0, 0
        count = size = 0
        for path in directory.rglob("*.png"):
            count += 1
            try:
                size += path.stat().st_size
            except OSError:
                pass
        return count, size

    enrolled_count, enrolled_size = measure(ENROLLED_DIR)
    query_count, query_size = measure(QUERIES_DIR)
    return {
        "enrolled_images": enrolled_count,
        "enrolled_mb": round(enrolled_size / 1_048_576, 1),
        "query_images": query_count,
        "query_mb": round(query_size / 1_048_576, 1),
        "base_dir": str(BASE_DIR),
    }


if __name__ == "__main__":
    print(f"Fingerprint image store: {BASE_DIR}")
    print(f"  save enrollment images: {SAVE_ENROLLMENT_IMAGES}")
    print(f"  save query images     : {SAVE_QUERY_IMAGES} (retention {QUERY_RETENTION_DAYS} days)")
    for key, value in storage_report().items():
        print(f"  {key}: {value}")
