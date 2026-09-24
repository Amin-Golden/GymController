#!/usr/bin/env python3
"""
Capture LABELLED fingerprints from the R503, for measuring the matcher.

test/r503_captures holds timestamped captures with no identity attached, so
accuracy can only be estimated by guessing which presses came from the same
finger.  This script records who each press belongs to, which is what
test/test_local_engine.py needs to report a real d' and EER.

Run it once per finger.  It takes no keyboard input, so it works over ssh and
inside tooling:

    py -3 test/capture_labelled.py --label alice_right_index --samples 5
    py -3 test/capture_labelled.py --label alice_left_index  --samples 5
    py -3 test/capture_labelled.py --label bob_right_index   --samples 5

Then measure:

    py -3 test/test_local_engine.py

Collect at least 6-8 different FINGERS (people or different fingers of the same
person both count) with 5 presses each.  Lift the finger fully between presses
and vary the position slightly, the way a real member would - pressing the exact
same spot five times flatters the matcher and hides the real error rate.

Images land in fingerprint_storage/images/<label>_sample_<n>.png, the layout the
benchmark already understands.  They are biometric data: keep the directory
access-controlled and out of git.
"""

import argparse
import re
import socket
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAFE = re.compile(r"[^A-Za-z0-9_-]")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True,
                    help="identity of this finger, e.g. alice_right_index")
    ap.add_argument("--samples", type=int, default=5, help="presses to record")
    ap.add_argument("--esp", default="192.168.1.120", help="ESP32 IP (for prompts)")
    ap.add_argument("--cmd-port", type=int, default=4210)
    ap.add_argument("--image-port", type=int, default=8889)
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="seconds to wait for each press")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "fingerprint_storage" / "images"))
    args = ap.parse_args()

    label = SAFE.sub("_", args.label)[:40]
    if not label:
        print("[ERROR] --label must contain letters or digits")
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob(f"{label}_sample_*.png"))
    if existing:
        print(f"[WARNING] {len(existing)} captures already exist for '{label}' - "
              f"they will be overwritten")

    from fingerprint_gym import UDPFingerprintReceiver
    min_minutiae = 16
    try:
        from fingerprint_local import LocalEngine, MIN_ENROLL_MINUTIAE
        engine = LocalEngine(verbose=False)
        min_minutiae = MIN_ENROLL_MINUTIAE
    except Exception:                                   # noqa: BLE001
        engine = None

    notify = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def tell(text):
        """Mirror the prompt to the ESP32 so its LED/display follows along."""
        try:
            notify.sendto(text.encode("utf-8"), (args.esp, args.cmd_port))
        except OSError:
            pass

    try:
        receiver = UDPFingerprintReceiver(args.image_port)
    except OSError as exc:
        print(f"[ERROR] cannot listen on port {args.image_port}: {exc}")
        print("        Stop the main gym application first - it holds that port.")
        return 1

    print(f"\nRecording {args.samples} presses for '{label}'")
    print("Lift the finger fully between presses and shift it slightly each time.\n")

    saved, counts = 0, []
    try:
        for i in range(1, args.samples + 1):
            print(f"  [{i}/{args.samples}] place finger on the sensor...", flush=True)
            tell(f"Sample {i} of {args.samples}")
            image = receiver.receive_image(timeout=args.timeout)
            if image is None:
                print(f"  [{i}/{args.samples}] timed out - no image received")
                continue

            path = out / f"{label}_sample_{i}.png"
            cv2.imwrite(str(path), image)
            saved += 1

            detail = ""
            if engine is not None:
                template = engine.extract_template(image)
                n = template["minutiae_count"] if template else 0
                counts.append(n)
                detail = f", {n} minutiae" + (
                    f" - TOO FEW (need {min_minutiae}), press firmer" if n < min_minutiae else "")
            print(f"  [{i}/{args.samples}] saved {path.name} "
                  f"({image.shape[1]}x{image.shape[0]}{detail})")
            if i < args.samples:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        receiver.close()

    print(f"\n{saved}/{args.samples} captures saved for '{label}' in {out}")
    if counts:
        print(f"minutiae: min {min(counts)} / mean {sum(counts)/len(counts):.1f} / max {max(counts)}")
    labels = {p.name.split("_sample_")[0] for p in out.glob("*_sample_*.png")}
    print(f"dataset now holds {len(labels)} labelled finger(s): {', '.join(sorted(labels))}")
    if len(labels) < 6:
        print(f"capture {6 - len(labels)} more finger(s), then run: py -3 test/test_local_engine.py")
    else:
        print("enough to measure - run: py -3 test/test_local_engine.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
