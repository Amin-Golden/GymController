#!/usr/bin/env python3
"""
Accuracy benchmark: NBIS (MINDTCT + BOZORTH3) vs the original SIFT engine.

Runs both matchers over the same stored captures, holding out one sample per
client as the query, and reports rank-1 identification accuracy plus the
false accept / false reject rates at a range of thresholds. Use it to pick
NBIS_MATCH_THRESHOLD for your own sensor and members.

Usage:
    py -3 test/test_nbis_pipeline.py                      # both engines
    py -3 test/test_nbis_pipeline.py --engine nbis
    py -3 test/test_nbis_pipeline.py --images some/dir

Image sources (in order):
    fingerprint_storage/images/enrolled/<client_id>/f<finger>_s<n>_*.png
    fingerprint_storage/images/<client>_sample_<n>.png      (older layout)
"""

import argparse
import collections
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fingerprint_nbis  # noqa: E402
from fingerprint_processor_improved import ImprovedFingerprintProcessor  # noqa: E402

DEFAULT_IMAGES = Path(__file__).resolve().parent.parent / "fingerprint_storage" / "images"


def collect_samples(images_dir: Path):
    """Return {client_key: [image paths]} from either storage layout."""
    samples = collections.defaultdict(list)

    enrolled = images_dir / "enrolled"
    if enrolled.is_dir():
        for client_dir in sorted(enrolled.iterdir()):
            if client_dir.is_dir():
                files = sorted(client_dir.glob("*.png"))
                if files:
                    samples[client_dir.name] = files

    if not samples:
        for path in sorted(images_dir.glob("*.png")):
            if "_sample_" in path.name and not path.name.startswith("query"):
                samples[path.name.split("_sample_")[0]].append(path)

    return {k: v for k, v in samples.items() if len(v) >= 2}


def report(name, genuine, impostor, rank1, total, elapsed):
    genuine, impostor = np.array(genuine, dtype=float), np.array(impostor, dtype=float)
    print(f"\n{'=' * 62}\n{name}\n{'=' * 62}")
    if genuine.size == 0:
        print("  no genuine comparisons - not enough data")
        return
    print(f"  rank-1 identification : {rank1}/{total}  ({rank1 / max(total, 1) * 100:.1f}%)")
    print(f"  genuine scores        : mean {genuine.mean():.2f}  min {genuine.min():.2f}  max {genuine.max():.2f}")
    print(f"  impostor scores       : mean {impostor.mean():.2f}  max {impostor.max():.2f}  "
          f"p99 {np.percentile(impostor, 99):.2f}")
    print(f"  identification time   : {elapsed / max(total, 1) * 1000:.0f} ms per query")

    candidates = np.unique(np.concatenate([genuine, impostor]))
    best = min(
        ((abs((impostor >= t).mean() - (genuine < t).mean()), t,
          (impostor >= t).mean(), (genuine < t).mean()) for t in candidates),
        key=lambda item: item[0],
    )
    print(f"  equal error rate      : ~{(best[2] + best[3]) / 2 * 100:.1f}% at threshold {best[1]:.2f}")
    print("\n  threshold      FAR (wrong person let in)    FRR (member rejected)")
    low = float(np.percentile(impostor, 50))
    high = float(max(genuine.max(), impostor.max()))
    grid = np.unique(np.round(np.linspace(low, high, 10), 2))
    for t in grid:
        print(f"    {t:8.2f}            {(impostor >= t).mean() * 100:6.2f}%                  "
              f"{(genuine < t).mean() * 100:6.2f}%")


def run_nbis(samples, engine):
    print("\nExtracting NBIS templates...")
    templates = {}
    for client, files in samples.items():
        extracted = []
        for path in files:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            template = engine.extract_template(image)
            extracted.append(template)
        usable = [t for t in extracted if t]
        counts = [t["minutiae_count"] for t in usable]
        print(f"  {client}: {len(usable)}/{len(files)} templates"
              + (f", {min(counts)}-{max(counts)} minutiae" if counts else ""))
        templates[client] = extracted

    genuine, impostor = [], []
    rank1 = total = 0
    started = time.time()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Gallery: every sample except the last of each client
        gallery = []
        for client, extracted in templates.items():
            for index, template in enumerate(extracted[:-1]):
                if not template:
                    continue
                path = tmp_path / f"{client}_{index}.xyt"
                path.write_text(template["xyt"])
                gallery.append((client, path))

        for client, extracted in templates.items():
            probe_template = extracted[-1]
            if not probe_template:
                continue
            total += 1
            probe = tmp_path / "probe.xyt"
            probe.write_text(probe_template["xyt"])
            scores = engine.match_files(probe, [g[1] for g in gallery])

            per_client = {}
            for (gallery_client, _path), score in zip(gallery, scores):
                per_client[gallery_client] = max(per_client.get(gallery_client, 0), score)

            for other, score in per_client.items():
                (genuine if other == client else impostor).append(score)

            if per_client and max(per_client, key=per_client.get) == client:
                rank1 += 1

    report("NBIS  -  MINDTCT + BOZORTH3", genuine, impostor, rank1, total, time.time() - started)
    return genuine, impostor


def run_sift(samples):
    processor = ImprovedFingerprintProcessor(embedding_dim=128, use_pca=True)
    print("\nExtracting SIFT features...")
    features = {}
    for client, files in samples.items():
        features[client] = [
            processor.extract_features(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
            for path in files
        ]

    genuine, impostor = [], []
    rank1 = total = 0
    started = time.time()

    for client, extracted in features.items():
        probe = extracted[-1]
        if probe is None:
            continue
        total += 1
        per_client = {}
        for other, other_features in features.items():
            scores = [
                processor.match_fingerprints_detailed(probe, enrolled)
                for enrolled in other_features[:-1] if enrolled is not None
            ]
            per_client[other] = max(scores) if scores else 0.0

        for other, score in per_client.items():
            (genuine if other == client else impostor).append(score)

        if per_client and max(per_client, key=per_client.get) == client:
            rank1 += 1

    report("SIFT  -  ratio test on descriptors (original engine)",
           genuine, impostor, rank1, total, time.time() - started)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default=str(DEFAULT_IMAGES))
    parser.add_argument("--engine", choices=["both", "nbis", "sift"], default="both")
    args = parser.parse_args()

    images_dir = Path(args.images)
    samples = collect_samples(images_dir)
    if not samples:
        print(f"No usable captures in {images_dir}")
        print("Enroll a few members first (raw images are kept automatically), or pass --images")
        return 1

    print(f"{len(samples)} clients, {sum(len(v) for v in samples.values())} images from {images_dir}")
    print("Protocol: last sample of each client is the query, the rest are enrolled.")

    if args.engine in ("both", "nbis"):
        engine = fingerprint_nbis.NbisEngine()
        if engine.available:
            run_nbis(samples, engine)
        else:
            print("\nNBIS binaries not found - see NBIS_SETUP.md")
            if args.engine == "nbis":
                return 1

    if args.engine in ("both", "sift"):
        run_sift(samples)

    print("\nPick NBIS_MATCH_THRESHOLD from the table above: the lowest threshold whose "
          "FAR is 0% on your data, then add some headroom for members not in this test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
