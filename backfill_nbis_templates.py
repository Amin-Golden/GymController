#!/usr/bin/env python3
"""
Generate NBIS templates for enrollments that were made before the migration.

A sample can be upgraded when its raw image is still on disk (image_path, or a
file under fingerprint_storage/images/enrolled/<client_id>/). Samples enrolled
before raw image retention existed cannot be recovered - those members keep
working through the SIFT fallback until they enroll again.

Usage:
    py -3 backfill_nbis_templates.py --dry-run     # report what can be upgraded
    py -3 backfill_nbis_templates.py               # write templates to the database
    py -3 backfill_nbis_templates.py --client 42   # one client only
"""

import argparse
import sys
from pathlib import Path

import fingerprint_image_store
import fingerprint_nbis
from db_helper import DatabaseHelper


def find_image(sample, client_id, finger_index, sample_number):
    """Locate the raw image for one sample, by stored path or by convention."""
    path = sample.get("image_path")
    if path and Path(path).is_file():
        return path

    directory = fingerprint_image_store.ENROLLED_DIR / fingerprint_image_store._safe(client_id)
    candidates = sorted(directory.glob(f"f{finger_index}_s{sample_number}_*.png"))
    if candidates:
        return str(candidates[-1])

    # Older layout: fingerprint_storage/images/<client_id>_sample_<n>.png
    legacy = fingerprint_image_store.BASE_DIR / f"{client_id}_sample_{sample_number}.png"
    if legacy.is_file():
        return str(legacy)

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    parser.add_argument("--client", type=int, default=None, help="restrict to one client id")
    parser.add_argument("--force", action="store_true", help="regenerate templates that already exist")
    args = parser.parse_args()

    engine = fingerprint_nbis.NbisEngine()
    if not engine.available:
        print("❌ NBIS binaries not found - see NBIS_SETUP.md")
        return 1

    db = DatabaseHelper()
    columns = db.get_fingerprint_columns()
    if "xyt_template" not in columns:
        print("❌ Run migrate_fingerprint_nbis.sql first")
        return 1

    stats = db.get_fingerprint_template_stats()
    print(f"Database: {stats['samples']} samples from {stats['clients']} clients, "
          f"{stats['with_template']} already have a template, {stats['with_image']} have an image path")

    profiles = db.get_all_fingerprint_features()
    if args.client is not None:
        profiles = [p for p in profiles if p["client_id"] == args.client]

    upgraded = skipped_existing = missing_image = failed = 0

    for profile in profiles:
        client_id = profile["client_id"]
        fingers = profile.get("fingers") or [{"finger_index": 1, "samples": profile.get("fingerprints", [])}]

        for finger in fingers:
            finger_index = finger.get("finger_index", 1)
            for sample in finger.get("samples", []):
                sample_number = sample.get("sample_number")

                if sample.get("xyt") and not args.force:
                    skipped_existing += 1
                    continue

                image_path = find_image(sample, client_id, finger_index, sample_number)
                if not image_path:
                    missing_image += 1
                    continue

                image = fingerprint_image_store.load_image(image_path)
                if image is None:
                    missing_image += 1
                    continue

                template = engine.extract_template(image)
                if template is None:
                    print(f"  ✗ client {client_id} f{finger_index} s{sample_number}: no minutiae in {image_path}")
                    failed += 1
                    continue

                print(f"  ✓ client {client_id} f{finger_index} s{sample_number}: "
                      f"{template['minutiae_count']} minutiae (quality {template['quality']:.0f})")

                if not args.dry_run:
                    ok = db.update_fingerprint_template(
                        client_id, finger_index, sample_number,
                        template["xyt"], template["minutiae_count"], image_path,
                    )
                    if not ok:
                        failed += 1
                        continue
                upgraded += 1

    print("\n" + "=" * 60)
    print(f"{'Would upgrade' if args.dry_run else 'Upgraded'}: {upgraded} samples")
    print(f"Already had a template : {skipped_existing}")
    print(f"No raw image available : {missing_image}  (these clients need to re-enroll for NBIS)")
    print(f"Failed                 : {failed}")
    if args.dry_run:
        print("\nRe-run without --dry-run to write the templates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
