#!/usr/bin/env python3
"""
Build whole-finger templates for members enrolled before multi-press stitching.

Enrollment now stitches a finger's presses into one template covering more of
the finger (fingerprint_local.fuse_presses) and stores it as sample 0. Members
enrolled earlier already have their presses' minutiae in the database, so their
whole-finger template can be built from those - no re-enrollment needed.

Run it once on the OrangePi (it needs the database), then restart RetinaFaceL.py
so the gallery reloads.

Usage:
    python3 backfill_fused_templates.py --dry-run          # report only
    python3 backfill_fused_templates.py                    # write templates
    python3 backfill_fused_templates.py --client 42        # one member only
    python3 backfill_fused_templates.py --force            # rebuild existing ones too
"""

import argparse
import sys

import fingerprint_local
from db_helper import DatabaseHelper


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db_host", default="192.168.1.3", help="PostgreSQL host")
    parser.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    parser.add_argument("--client", type=int, default=None, help="restrict to one client id")
    parser.add_argument("--force", action="store_true", help="rebuild whole-finger templates that already exist")
    args = parser.parse_args()

    db = DatabaseHelper(host=args.db_host, database='gym-db', user='postgres', password='123456')
    if "xyt_template" not in db.get_fingerprint_columns():
        print("❌ Run migrate_fingerprint_nbis.sql first")
        return 1

    profiles = db.get_all_fingerprint_features()
    if args.client is not None:
        profiles = [p for p in profiles if p.get("client_id") == args.client]
    print(f"{len(profiles)} member(s) with fingerprints")

    built = skipped = existing = failed = 0
    sizes = []
    for profile in profiles:
        cid = profile["client_id"]
        for finger in profile.get("fingers", []):
            fidx = finger["finger_index"]
            samples = finger.get("samples", [])
            if not args.force and any(s.get("sample_number") == db.FUSED_SAMPLE_NUMBER for s in samples):
                existing += 1
                continue
            presses = [fingerprint_local.parse_xyt(s["xyt"]) for s in samples
                       if s.get("xyt") and s.get("sample_number") != db.FUSED_SAMPLE_NUMBER]
            if len(presses) < 2:
                skipped += 1
                continue

            fused, used = fingerprint_local.fuse_presses(presses)
            if fused is None:
                print(f"  client {cid} finger {fidx}: presses do not overlap enough - left as is")
                failed += 1
                continue
            single = max(len(p) for p in presses)
            print(f"  client {cid} finger {fidx}: {len(fused)} minutiae from {len(used)}/{len(presses)} "
                  f"presses (largest single press {single})")
            sizes.append(len(fused) / single)
            if not args.dry_run:
                if not db.save_fused_template(cid, fidx, fingerprint_local.minutiae_to_xyt(fused), len(fused)):
                    failed += 1
                    continue
            built += 1

    verb = "would build" if args.dry_run else "built"
    print(f"\n{verb} {built} whole-finger template(s); {existing} already had one, "
          f"{skipped} finger(s) had fewer than 2 minutiae presses, {failed} could not be stitched")
    if sizes:
        print(f"whole-finger templates hold {sum(sizes) / len(sizes):.1f}x the minutiae of the best single press")
    db.close_all_connections()
    return 0


if __name__ == "__main__":
    sys.exit(main())
