#!/usr/bin/env python3
"""
Record the R503's own feature file next to the images it came from.

Needs the PAIR command in finger_UDP/finger_UDP.ino. For each press the ESP32
uploads three things from the sensor, unmodified:

    <label>_s<n>_raw.png    UpImage before Img2Tz - the image as captured
    <label>_s<n>_proc.png   UpImage after Img2Tz  - what the normal path sends
    <label>_s<n>.char       UpChar(1) - the sensor's feature file (384 bytes)

The first press settles one question outright: if raw.png has more than two
grey levels while proc.png has two, Img2Tz is what binarises every normal
capture, and uploading before Img2Tz gives the PC real grayscale.

The .char files, paired with proc.png (the skeleton the sensor extracted them
from), are the material for decoding the template format. Collect several
fingers, several presses each, lifting and shifting between presses:

    py -3 test/capture_template_pairs.py --label alice_right_index --samples 5
    py -3 test/capture_template_pairs.py --label bob_right_index   --samples 5

On some firmwares UpChar after a single Img2Tz returns only zeros. --probe
checks whether this module exports templates at all: it reads the module's
product info, takes two presses, runs the sensor's own Match, merges them with
RegModel and uploads both buffers before and after. Nothing is stored:

    py -3 test/capture_template_pairs.py --probe

STOP the main gym application first: a press it sees would be treated as a
real entry attempt, and its enrollment prompts would redirect the replies.

The output is biometric data: test/template_pairs/ is git-ignored.
"""

import argparse
import re
import socket
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from r503_link import decode_image, run_command  # noqa: E402  (protocol shared with the gym system)

SAFE = re.compile(r"[^A-Za-z0-9_-]")

# ReadProdInfo (0x3C): model, batch, serial, hw version, sensor, w, h, template size, db size
PROD_INFO = struct.Struct(">16s4s8s2s8sHHHH")


def describe_levels(img):
    levels = np.unique(img)
    shown = ", ".join(str(v) for v in levels[:8]) + (" ..." if len(levels) > 8 else "")
    return len(levels), shown


def _ascii(b):
    return b.split(b"\0", 1)[0].decode("ascii", errors="replace")


def probe(sock, target, out, timeout):
    """Does this module export templates at all? Two presses, nothing stored."""
    def status(text):
        prompts = {"PROBE:PLACE": "place a finger on the sensor...",
                   "PROBE:LIFT": "lift the finger",
                   "PROBE:PLACE_AGAIN": "place the SAME finger again..."}
        if text in prompts:
            print(f"  {prompts[text]}", flush=True)
        elif text.startswith("PROBE:REJECT"):
            print(f"      sensor rejected the press ({text[13:]}) - press again")
        elif text.startswith("PROBE:INFO") and text != "PROBE:INFO code=0":
            print(f"  ReadProdInfo not supported ({text[11:]})")

    reply, blobs = run_command(sock, target, "PROBE", timeout, status)
    if reply is None:
        print("no answer from the ESP32 - is the PROBE firmware flashed?")
        return 1
    print(f"\n{reply}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    for kind, data in sorted(blobs.items()):
        (out / f"probe_{stamp}_{kind}.bin").write_bytes(data)

    if "I" in blobs and len(blobs["I"]) >= PROD_INFO.size:
        model, batch, serial, hw, sensor, w, h, tsize, total = PROD_INFO.unpack_from(blobs["I"])
        print(f"\nmodule  : {_ascii(model)}  batch {_ascii(batch)}  serial {_ascii(serial)}  "
              f"hw {hw[0]}.{hw[1]}")
        print(f"sensor  : {_ascii(sensor)}  {w}x{h}")
        print(f"template: {tsize} bytes, library {total}")
    if "S" in blobs and len(blobs["S"]) >= 16:
        s = struct.unpack(">HHHHIHH", blobs["S"][:16])
        print(f"syspara : status {s[0]:#06x} id {s[1]:#06x} library {s[2]} security {s[3]} "
              f"packet {32 << s[5]} baud {9600 * s[6]}")
    if "F" in blobs:
        text = "".join(chr(c) if 32 <= c < 127 else "." for c in blobs["F"])
        print(f"info pg : {len(blobs['F'])} bytes | {re.sub(r'[.]{3,}', ' .. ', text)[:200]}")

    print()
    names = {"a": "press 1, after Img2Tz(1)", "b": "press 2, after Img2Tz(2)",
             "T": "after RegModel, buffer 1", "U": "after RegModel, buffer 2"}
    exported = {}
    for kind, what in names.items():
        data = blobs.get(kind)
        if data is None:
            print(f"  UpChar {what:26s}: not received")
            continue
        nz = np.count_nonzero(np.frombuffer(data, np.uint8))
        exported[kind] = nz
        print(f"  UpChar {what:26s}: {len(data)} bytes, {nz} non-zero | {data[:24].hex(' ')}")

    m = re.search(r"match=(-?\d+) score=(\d+) reg=(-?\d+)", reply)
    if m:
        match, score, reg = (int(g) for g in m.groups())
        print(f"\n  sensor Match(1,2): {'MATCH' if match == 0 else f'code {match}'}, score {score}; "
              f"RegModel: {'ok' if reg == 0 else f'code {reg}'}")

    if any(exported.get(k) for k in "TU"):
        print("\nVERDICT: merged templates export. Decoding can proceed on RegModel output.")
    elif any(exported.get(k) for k in "ab"):
        print("\nVERDICT: single feature files export after all - compare with the PAIR captures.")
    elif exported:
        print("\nVERDICT: every upload is zero although the sensor holds real features.")
        print("         This firmware does not export templates; decoding is not possible.")
    print(f"\nsaved to {out / f'probe_{stamp}_*.bin'}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", help="identity of this finger, e.g. alice_right_index")
    ap.add_argument("--probe", action="store_true",
                    help="instead of recording: check whether the module exports templates "
                         "(product info, two presses, RegModel, UpChar of both buffers)")
    ap.add_argument("--samples", type=int, default=5, help="presses to record")
    ap.add_argument("--esp", default="192.168.1.120", help="ESP32 IP")
    ap.add_argument("--cmd-port", type=int, default=4210)
    ap.add_argument("--timeout", type=float, default=40.0,
                    help="seconds to wait for each press and its uploads")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "template_pairs"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.probe:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("", 0))
        try:
            return probe(sock, (args.esp, args.cmd_port), out, args.timeout + 30)
        finally:
            sock.close()

    label = SAFE.sub("_", args.label or "")[:40]
    if not label:
        print("[ERROR] --label must contain letters or digits")
        return 2

    try:
        from fingerprint_local import LocalEngine
        engine = LocalEngine(verbose=False)
    except Exception:                             # noqa: BLE001
        engine = None

    def minutiae(img):
        if engine is None:
            return "-"
        t = engine.extract_template(img)
        return t["minutiae_count"] if t else 0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 0))                            # replies come back to this port
    target = (args.esp, args.cmd_port)

    print(f"\nRecording {args.samples} presses for '{label}' -> {out}")
    print("Lift the finger fully between presses and shift it slightly each time.\n")

    chars, verdicts = [], []
    saved, attempts = 0, 0
    try:
        while saved < args.samples and attempts < args.samples * 3:
            attempts += 1
            n = saved + 1
            print(f"  [{n}/{args.samples}] place finger on the sensor...", flush=True)

            def status(text):
                if text.startswith("PAIR:REJECT"):
                    print(f"      sensor rejected the press ({text[12:]}) - press again")

            reply, blobs = run_command(sock, target, "PAIR", args.timeout, status)
            if reply is None:
                print("      no answer from the ESP32 - is the PAIR firmware flashed?")
                continue
            if not reply.startswith("PAIR:DONE") or not {"R", "P", "C"} <= blobs.keys():
                print(f"      {reply} (got {sorted(blobs)}) - retrying")
                continue

            raw, proc = decode_image(blobs["R"]), decode_image(blobs["P"])
            char = blobs["C"]
            if raw is None or proc is None:
                print(f"      unknown image size raw={len(blobs['R'])} proc={len(blobs['P'])} - "
                      f"saving the bytes only")
            stem = out / f"{label}_s{n}"
            for kind, suffix in (("R", "_raw.bin"), ("P", "_proc.bin")):
                if decode_image(blobs[kind]) is None:
                    (stem.parent / (stem.name + suffix)).write_bytes(blobs[kind])
            if raw is not None:
                cv2.imwrite(str(stem) + "_raw.png", raw)
            if proc is not None:
                cv2.imwrite(str(stem) + "_proc.png", proc)
            stem.with_suffix(".char").write_bytes(char)
            saved += 1
            chars.append(char)

            nz = np.flatnonzero(np.frombuffer(char, np.uint8))
            used = int(nz[-1]) + 1 if len(nz) else 0
            print(f"      char  : {len(char)} bytes, last non-zero byte at {used - 1} | "
                  f"{char[:24].hex(' ')}")
            if raw is not None and proc is not None:
                rl, rs = describe_levels(raw)
                pl, ps = describe_levels(proc)
                same = raw.shape == proc.shape and np.array_equal(raw, proc)
                print(f"      raw   : {raw.shape[1]}x{raw.shape[0]}, {rl} grey levels ({rs}), "
                      f"{minutiae(raw)} minutiae (local engine)")
                print(f"      proc  : {proc.shape[1]}x{proc.shape[0]}, {pl} grey levels ({ps}), "
                      f"{minutiae(proc)} minutiae (local engine)"
                      + ("  [identical to raw]" if same else ""))
                verdicts.append((rl, pl, same))
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        sock.close()

    print(f"\n{saved}/{args.samples} pairs saved for '{label}'")

    if verdicts:
        raw_levels = min(v[0] for v in verdicts)
        proc_levels = max(v[1] for v in verdicts)
        if raw_levels > 2 and proc_levels <= 2:
            print("\nVERDICT: Img2Tz overwrites the image buffer. The raw image is real grayscale -")
            print("         uploading BEFORE Img2Tz gives the PC far more ridge detail than it gets today.")
        elif all(v[2] for v in verdicts):
            print("\nVERDICT: raw and processed images are identical - Img2Tz leaves the buffer alone,")
            print("         and the sensor itself delivers the image this way.")
        else:
            print(f"\nVERDICT: raw has >= {raw_levels} grey levels, processed <= {proc_levels}.")

    all_chars = sorted(out.glob("*.char"))
    sizes = {p.stat().st_size for p in all_chars}
    if len(all_chars) >= 2 and len(sizes) == 1:
        stack = np.stack([np.frombuffer(p.read_bytes(), np.uint8) for p in all_chars])
        fixed = np.flatnonzero((stack == stack[0]).all(axis=0))
        print(f"\nacross all {len(all_chars)} feature files ({sizes.pop()} bytes each): "
              f"{len(fixed)} offsets never change")
        if len(fixed):
            print("  " + " ".join(f"{i}={stack[0, i]:02x}" for i in fixed[:48])
                  + (" ..." if len(fixed) > 48 else ""))
    labels = {p.name.rsplit("_s", 1)[0] for p in all_chars}
    print(f"dataset: {len(all_chars)} feature files from {len(labels)} finger(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
