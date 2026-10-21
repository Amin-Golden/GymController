#!/usr/bin/env python3
"""
Drive the R503's own extractor and matcher from the PC, and match its
templates on the PC with r503_template.py.

Needs the IMG2TMPL / MATCHTMPL / TMPL commands in finger_UDP/finger_UDP.ino.

    build     Upload images to the sensor (DownImage) and get back the template
              its extractor makes from each, plus the skeleton it drew. No
              finger needed. Default input: the raw presses recorded by
              capture_template_pairs.py.
    match     The SENSOR's matcher on every ordered pair of built templates:
              template i is rebuilt from its image (that leaves it on the
              sensor's side 1 - Match ignores DownChar(1)), then every other
              template is sent with DownChar(2) and matched.
    evaluate  The PC matcher (r503_template.match) on the same templates,
              offline, with the same statistics - compare it with `match`.
    capture   Live presses -> templates, saved under a label. TMPL takes a
              second capture of the same press (hold the finger ~1 s; no image
              transfer); --raw uses one capture sent back with DownImage, the
              way `build` makes templates (~2.2 s slower).
    identify  Live press -> template -> identified on the PC against a gallery
              of templates; --confirm also has the sensor Match the live press
              against the PC's best candidates.
    confirmtest  The sensor double-check (fingerprint_gym.py) on recorded presses:
              each query press becomes the sensor's side (as CONFIRM does) and is
              matched against the stored templates of its own finger and of the
              closest wrong finger the PC matcher finds. Needs the ESP32.
    compare   The template matcher vs SIFT on the grayscale image, on the SAME
              presses. Record with --raw (each press keeps the image its
              template was built from); enroll in one session, query in
              another, and include fingers that were never enrolled.

    py -3 test/r503_template_lab.py build
    py -3 test/r503_template_lab.py match
    py -3 test/r503_template_lab.py evaluate
    py -3 test/r503_template_lab.py capture --label ali_right_index --samples 5
    py -3 test/r503_template_lab.py identify --trials 10 --confirm
    py -3 test/r503_template_lab.py --raw capture --label ali_right_index --out test/template_lab/session_a
    py -3 test/r503_template_lab.py compare --gallery test/template_lab/session_a --queries test/template_lab/session_b

Nothing is written to the sensor's library. Stop the gym application while a
live command runs: a press it sees would be treated as an entry attempt.

The output is biometric data: test/template_lab/ is git-ignored.
"""

import argparse
import csv
import glob
import re
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))
import fingerprint_sift  # noqa: E402
import r503_template as rt  # noqa: E402
from r503_link import SensorLink, decode_image, pack_image, run_command, upload  # noqa: E402

HERE = Path(__file__).resolve().parent
LAB = HERE / "template_lab"
SAFE = re.compile(r"[^A-Za-z0-9_-]")


# --------------------------------------------------------------------------
# ESP32 link
# --------------------------------------------------------------------------

def image_to_template(sock, target, img, depth, timeout):
    """(reply, blobs) for one image through IMG2TMPL; leaves its template on the sensor's side 1."""
    if not upload(sock, target, "I", pack_image(img, depth)):
        return None, {}
    return run_command(sock, target, "IMG2TMPL", timeout, lambda _t: None)


def sensor_match(sock, target, tmpl, timeout):
    """Sensor Match of tmpl against its side 1. (matched, score) or None."""
    if not upload(sock, target, "B", tmpl):
        return None
    reply, _ = run_command(sock, target, "MATCHTMPL", timeout, lambda _t: None)
    if not reply or not reply.startswith("MATCHTMPL:DONE"):
        return None
    fields = dict(kv.split("=") for kv in reply.split()[1:] if "=" in kv)
    return int(fields["match"]) == 0, int(fields["score"])


def live_template(sock, target, timeout, raw=False):
    """One press -> (template bytes, image or None) via TMPL (or TMPL RAW), or None.

    Only TMPL RAW returns the image: it is the one capture the template was built from."""
    def status(text):
        if text.startswith("TMPL:REJECT"):
            print(f"      sensor rejected the press ({text[12:]}) - press again")
        elif text == "TMPL:PLACE_AGAIN":
            print("      finger lifted too soon - place the SAME finger again")
    reply, blobs = run_command(sock, target, "TMPL RAW" if raw else "TMPL", timeout, status,
                               prefix="TMPL")
    if reply is None:
        print("      no answer from the ESP32 - is the TMPL firmware flashed?")
        return None
    if not reply.startswith("TMPL:DONE") or "T" not in blobs:
        print(f"      {reply}")
        return None
    fields = dict(kv.split("=") for kv in reply.split()[1:] if "=" in kv)
    print(f"      template in {fields.get('ms', '?')} ms ({fields.get('mode', '?')} mode, "
          f"{fields.get('captures', '?')} capture(s))")
    image = decode_image(blobs["R"]) if "R" in blobs else None
    return blobs["T"], image


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def label_of(stem):
    return stem.rsplit("_s", 1)[0]


def report(title, names, S, note=None, thresholds=None):
    """Identification as the gym does it: each template queried against every
    finger's OTHER samples, the finger scoring its best sample. S[q, k] is the
    score of query q against gallery template k (need not be symmetric)."""
    L = np.array([label_of(n) for n in names])
    labels = sorted(set(L))
    n = len(names)
    gen, imp, rank1 = [], [], 0
    for q in range(n):
        per = {}
        for lab in labels:
            vals = [S[q, k] for k in range(n) if L[k] == lab and k != q and not np.isnan(S[q, k])]
            if vals:
                per[lab] = max(vals)
        if L[q] not in per:
            continue
        gen.append(per[L[q]])
        imp += [v for lab, v in per.items() if lab != L[q]]
        rank1 += max(per, key=per.get) == L[q]
    gen, imp = np.array(gen), np.array(imp)
    print(f"\n{title}: {len(names)} templates, {len(labels)} fingers")
    if not len(gen) or not len(imp):
        print("  need at least two samples of two fingers")
        return
    print(f"  rank-1 {rank1}/{len(gen)}")
    print(f"  genuine  (best of the finger's other samples): mean {gen.mean():.2f}  min {gen.min():.2f}")
    print(f"  impostor (best sample of another finger):      mean {imp.mean():.2f}  max {imp.max():.2f}")
    print(f"  genuine at or below the best impostor: {np.sum(gen <= imp.max())}/{len(gen)}")
    if note:
        print(f"  {note}")
    if thresholds is not None:
        for t in thresholds:
            print(f"    accept >= {t:5.1f}: impostors accepted {np.sum(imp >= t):3d}/{len(imp)}   "
                  f"genuine rejected {np.sum(gen < t):2d}/{len(gen)}")


def evaluate(args):
    paths = sorted(Path(args.dir).glob("*.tmpl"))
    names = [p.stem for p in paths]
    tmpls = [rt.prepare(p.read_bytes()) for p in paths]
    n = len(names)
    S = np.full((n, n), np.nan)
    t0 = time.perf_counter()
    for i in range(n):
        for j in range(n):
            if i != j:
                S[i, j] = rt.match(tmpls[i], tmpls[j])[0]   # query i against gallery j
    pairs = n * (n - 1)
    ms = 1000 * (time.perf_counter() - t0) / max(pairs, 1)
    report(f"PC matcher (r503_template), {ms:.1f} ms/comparison", names, S,
           thresholds=[2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    print(f"  current threshold R503_MATCH_THRESHOLD = {rt.R503_MATCH_THRESHOLD}")
    return 0


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def build(sock, target, args):
    paths = [Path(p) for pattern in args.images for p in sorted(glob.glob(pattern))]
    if not args.images:
        paths = sorted((HERE / "template_pairs").glob("*_raw.png"))
    if not paths:
        print("no input images")
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    made = 0
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  {p.name}: unreadable")
            continue
        stem = p.stem[:-4] if p.stem.endswith("_raw") else p.stem
        reply, blobs = image_to_template(sock, target, img, args.depth, args.timeout)
        if reply is None:
            print(f"  {stem}: no answer - is the template-lab firmware flashed?")
            continue

        note = ""
        if "P" in blobs:
            skel = decode_image(blobs["P"])
            if skel is not None:
                cv2.imwrite(str(out / f"{stem}_dproc.png"), skel)
                pressed = p.with_name(f"{stem}_proc.png")
                if pressed.exists():
                    ref = cv2.imread(str(pressed), cv2.IMREAD_GRAYSCALE)
                    if ref is not None and ref.shape == skel.shape:
                        same = float(np.mean((ref < 128) == (skel < 128)))
                        note = f", skeleton {100 * same:.1f}% same as the live press"
        if "T" in blobs:
            t = blobs["T"]
            (out / f"{stem}.tmpl").write_bytes(t)
            made += 1
            try:
                count = len(rt.decode(t)["minutiae"])
                print(f"  {stem}: {count} minutiae{note}")
            except ValueError as exc:
                print(f"  {stem}: saved, but {exc}")
        else:
            print(f"  {stem}: {reply}{note}")

    print(f"\n{made}/{len(paths)} templates in {out}")
    return 0 if made else 1


def source_image(stem, images):
    for cand in (images / f"{stem}_raw.png", images / f"{stem}.png"):
        if cand.exists():
            return cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE)
    return None


def match(sock, target, args):
    paths = sorted(Path(args.dir).glob("*.tmpl"))
    names = [p.stem for p in paths]
    data = [p.read_bytes() for p in paths]
    n = len(names)
    if n < 2:
        print(f"need at least two .tmpl files in {args.dir} - run build first")
        return 1
    images = Path(args.images)
    S = np.full((n, n), np.nan)
    decided = np.zeros((n, n), bool)
    rows, done, t0 = [], 0, time.time()
    total = n * (n - 1)

    for i in range(n):
        img = source_image(names[i], images)
        if img is None:
            print(f"  {names[i]}: no source image in {images} - skipped as a query")
            continue
        reply, blobs = image_to_template(sock, target, img, args.depth, args.timeout)
        if reply is None or not reply.startswith("IMG2TMPL:DONE"):
            print(f"  {names[i]}: could not rebuild ({reply})")
            continue
        if blobs.get("T") != data[i]:
            print(f"  {names[i]}: note - the rebuilt template differs from the stored one")
        upload(sock, target, "A", data[i])           # older firmware needs an 'A'; newer rejects it
        for j in range(n):
            if j == i:
                continue
            res = sensor_match(sock, target, data[j], args.timeout)
            done += 1
            if res is None:
                print(f"  {names[i]} vs {names[j]}: no result")
                continue
            decided[i, j], S[i, j] = res
            rows.append((names[i], names[j], int(res[0]), res[1]))
        rate = (time.time() - t0) / max(done, 1)
        print(f"  {names[i]}: done ({done}/{total} pairs, {rate:.2f} s/pair)", flush=True)

    out = Path(args.dir) / "sensor_match_scores.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side1", "downchar", "matched", "score"])
        w.writerows(rows)

    L = np.array([label_of(s) for s in names])
    same = L[:, None] == L[None, :]
    off = ~np.eye(n, dtype=bool) & ~np.isnan(S)
    far = decided[off & ~same].mean() if (off & ~same).any() else float("nan")
    frr = 1 - decided[off & same].mean() if (off & same).any() else float("nan")
    report("sensor matcher (DownChar + Match)", names, S,
           note=f"sensor's own decision on single pairs: FAR {100 * far:.1f}%  FRR {100 * frr:.1f}% "
                f"(security level as set on the sensor)")
    print(f"  scores saved to {out}")
    return 0


def capture(sock, target, args):
    label = SAFE.sub("_", args.label or "")[:40]
    if not label:
        print("[ERROR] --label must contain letters or digits")
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\nRecording {args.samples} presses for '{label}' -> {out}")
    print("Lift the finger fully between presses and shift it slightly each time.\n")
    saved = attempts = 0
    while saved < args.samples and attempts < args.samples * 3:
        attempts += 1
        n = saved + 1
        print(f"  [{n}/{args.samples}] place finger on the sensor...", flush=True)
        got = live_template(sock, target, args.timeout, args.raw)
        if got is None:
            continue
        t, image = got
        try:
            count = len(rt.decode(t)["minutiae"])
        except ValueError as exc:
            print(f"      {exc}")
            continue
        (out / f"{label}_s{n}.tmpl").write_bytes(t)
        if image is not None:                       # TMPL RAW: keep the press for image matchers
            cv2.imwrite(str(out / f"{label}_s{n}_raw.png"), image)
        saved += 1
        print(f"      saved, {count} minutiae" + (" + image" if image is not None else ""))
    print(f"\n{saved}/{args.samples} templates saved for '{label}'")
    return 0


def identify(sock, target, args):
    paths = sorted(Path(args.gallery).glob("*.tmpl"))
    if not paths:
        print(f"no templates in {args.gallery} - run capture first")
        return 1
    gallery = [(label_of(p.stem), rt.prepare(p.read_bytes()), p.read_bytes()) for p in paths]
    print(f"gallery: {len(paths)} templates, {len({g[0] for g in gallery})} fingers, "
          f"threshold {rt.R503_MATCH_THRESHOLD}\n")
    for trial in range(1, args.trials + 1):
        print(f"[{trial}/{args.trials}] place finger on the sensor...", flush=True)
        got = live_template(sock, target, args.timeout, args.raw)
        if got is None:
            continue
        t = got[0]
        t0 = time.perf_counter()
        query = rt.prepare(t)
        best = {}
        for lab, tmpl, raw in gallery:
            s = rt.match(query, tmpl)[0]
            if s > best.get(lab, (-1.0,))[0]:
                best[lab] = (s, raw)
        ms = 1000 * (time.perf_counter() - t0)
        ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
        top, (score, _raw) = ranked[0]
        runner = ranked[1][1][0] if len(ranked) > 1 else 0.0
        verdict = top if score >= rt.R503_MATCH_THRESHOLD else "NO MATCH"
        print(f"   PC: {verdict}  (best {top} {score:.2f}, runner-up {runner:.2f}, "
              f"{len(rt.decode(t)['minutiae'])} minutiae, {ms:.0f} ms)")
        if args.confirm:
            # the live press is still the sensor's side 1: let its own matcher check the top candidates
            for lab, (s, raw) in ranked[:args.confirm_top]:
                res = sensor_match(sock, target, raw, args.timeout)
                if res is None:
                    print(f"   sensor vs {lab}: no result")
                    continue
                print(f"   sensor vs {lab}: {'MATCH' if res[0] else 'no match'} (score {res[1]})")
    return 0


# --------------------------------------------------------------------------
# compare: the template matcher vs SIFT on the same presses
# --------------------------------------------------------------------------

_SIFT = None
_BF = None


def sift_descriptors(img):
    """SIFT exactly as fingerprint_gym.FingerprintProcessor runs it."""
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create()
    im = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(img)
    im = cv2.fastNlMeansDenoising(im, None, h=10)
    im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX)
    return _SIFT.detectAndCompute(im, None)[1]


def sift_score(d1, d2):
    """FingerprintProcessor.match_fingerprints: ratio-test matches over the mean descriptor count."""
    global _BF
    if _BF is None:
        _BF = cv2.BFMatcher(cv2.NORM_L2)
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return 0.0
    good = sum(1 for pair in _BF.knnMatch(d1, d2, k=2)
               if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance)
    score = good / ((len(d1) + len(d2)) / 2.0)
    return min(score * (1.1 if good > 20 else 1.0), 1.0)


def load_presses(folder, images=None):
    """[(name, finger, prepared template, SIFT descriptors or None, fingerprint_sift.Features or None)]
    for each .tmpl in folder."""
    out = []
    for p in sorted(Path(folder).glob("*.tmpl")):
        img_path = Path(images or folder) / f"{p.stem}_raw.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE) if img_path.exists() else None
        out.append((p.stem, label_of(p.stem), rt.prepare(p.read_bytes()),
                    sift_descriptors(img) if img is not None else None,
                    fingerprint_sift.extract(img) if img is not None else None))
    return out


def summarise(queries, gallery, score, same_set):
    """Each query against the gallery; each finger scores its best sample.

    Queries whose finger is not enrolled count only as impostors - the case
    a stranger at the door is."""
    gen, imp, missed, enrolled = [], [], [], 0
    for qi, q in enumerate(queries):
        per = {}
        for gi, g in enumerate(gallery):
            if same_set and qi == gi:
                continue
            s = score(q, g)
            if s > per.get(g[1], -np.inf):
                per[g[1]] = s
        imp += [v for lab, v in per.items() if lab != q[1]]
        if q[1] in per:
            enrolled += 1
            gen.append(per[q[1]])
            top = max(per.values())
            if [lab for lab, v in per.items() if v == top] != [q[1]]:   # a tie is a miss
                missed.append(q[0])
    gen, imp = np.array(gen), np.array(imp)
    d = (gen.mean() - imp.mean()) / np.sqrt((gen.var() + imp.var()) / 2 + 1e-12) if len(gen) and len(imp) else float("nan")
    below = int(np.sum(gen <= imp.max())) if len(gen) and len(imp) else 0
    return {"rank1": enrolled - len(missed), "enrolled": enrolled, "below": below,
            "d": d, "missed": missed, "strangers": len(queries) - enrolled}


def compare(args):
    gallery = load_presses(args.gallery, args.images)
    same_set = not args.queries
    queries = gallery if same_set else load_presses(args.queries, args.images)
    if not gallery or not queries:
        print("no templates found")
        return 1
    print(f"gallery {args.gallery}: {len(gallery)} presses, {len({g[1] for g in gallery})} fingers")
    if same_set:
        print("queries: the same presses, each against the others (one session only)")
    else:
        print(f"queries {args.queries}: {len(queries)} presses")

    methods = [("template (r503_template)", lambda q, g: rt.match(q[2], g[2])[0])]
    if all(x[3] is not None for x in gallery + queries):
        methods.append(("SIFT on the grayscale image", lambda q, g: sift_score(q[3], g[3])))
        methods.append(("SIFT + geometry (production)", lambda q, g: fingerprint_sift.inliers(q[4], g[4])))
    else:
        print("SIFT skipped: not every press has its _raw.png - record with --raw")

    results = [(title, summarise(queries, gallery, fn, same_set)) for title, fn in methods]
    first = results[0][1]
    if first["strangers"]:
        print(f"{first['strangers']} query presses are from fingers not in the gallery - "
              f"they count as strangers")
    print(f"\n{'method':30s} {'right finger first':>19s} {'genuine at/below':>17s} {'separation':>11s}")
    print(f"{'':30s} {'':>19s} {'best impostor':>17s} {'(d-prime)':>11s}")
    for title, r in results:
        print(f"{title:30s} {r['rank1']:>12d}/{r['enrolled']:<6d} {r['below']:>10d}/{r['enrolled']:<6d} {r['d']:>11.2f}")
    for title, r in results:
        if r["missed"]:
            print(f"  {title} misidentified: {', '.join(r['missed'])}")
    return 0


def confirmtest(args):
    """The sensor double-check measured on recorded presses, as fingerprint_gym runs it."""
    gal = load_presses(args.gallery)
    qry = load_presses(args.queries)
    if not gal or not qry or any(x[4] is None for x in gal + qry):
        print("need .tmpl + _raw.png presses in both folders (record with --raw)")
        return 1
    # the templates enrollment would keep: each finger's presses with the most features
    stored = {}
    for name, label, _t, _d, f in sorted(gal, key=lambda x: -x[4].n):
        if len(stored.setdefault(label, [])) < args.templates:
            stored[label].append((Path(args.gallery) / f"{name}.tmpl").read_bytes())
    # the PC matcher picks the candidate; its closest WRONG finger is the impostor to test
    gallery = fingerprint_sift.SiftGallery(index_dir=None, verbose=False)
    ids = {label: i + 1 for i, label in enumerate(sorted(stored))}
    gallery.sync([{"client_id": ids[label], "name": label, "fingers": [{"finger_index": 1, "samples": [
        {"sample_number": i + 1, "sift_gray": fingerprint_sift.pack(x[4])}
        for i, x in enumerate(p for p in gal if p[1] == label)]}]} for label in sorted(stored)])
    label_of_id = {v: k for k, v in ids.items()}

    gen = imp = gen_ok = imp_ok = 0
    times = []
    with SensorLink(args.esp, args.cmd_port, timeout=args.timeout) as link:
        for name, label, _t, _d, f in qry:
            img = cv2.imread(str(Path(args.queries) / f"{name}_raw.png"), cv2.IMREAD_GRAYSCALE)
            t0 = time.time()
            if link.image_template(img) is None:          # leaves this press as the sensor's side 1
                print(f"  {name}: the sensor could not build its template")
                continue
            ranking = gallery.identify(f, threshold=10 ** 9)[1]
            wrong = next((label_of_id[r[1]] for r in ranking if label_of_id[r[1]] != label), None)
            tests = [("own finger", label)] if label in stored else []
            if wrong:
                tests.append(("closest other", wrong))
            line = []
            for kind, finger in tests:
                results = [link.match(t) for t in stored[finger]]
                accepted = any(r and r[0] for r in results)
                scores = "/".join(str(r[1]) if r else "-" for r in results)
                line.append(f"{kind} {finger}: {'MATCH' if accepted else 'no'} ({scores})")
                if kind == "own finger":
                    gen += 1
                    gen_ok += accepted
                else:
                    imp += 1
                    imp_ok += accepted
            times.append(time.time() - t0)
            print(f"  {name:24s} " + " | ".join(line))
    print(f"\nsensor double-check, {args.templates} stored templates per finger:")
    print(f"  own finger confirmed:            {gen_ok}/{gen}")
    print(f"  closest wrong finger confirmed:  {imp_ok}/{imp}   (must be 0)")
    if times:
        print(f"  time per press (template + matches): median {np.median(times):.1f} s")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--esp", default="192.168.1.120", help="ESP32 IP")
    ap.add_argument("--cmd-port", type=int, default=4210)
    ap.add_argument("--timeout", type=float, default=20.0, help="seconds per sensor operation")
    ap.add_argument("--depth", type=int, choices=(4, 8), default=4,
                    help="DownImage format: 4-bit packed (as UpImage sends) or 8-bit")
    ap.add_argument("--raw", action="store_true",
                    help="capture/identify: TMPL RAW - one capture, via DownImage (slower)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="images -> sensor templates")
    b.add_argument("images", nargs="*", help="image files or globs (default: recorded raw presses)")
    b.add_argument("--out", default=str(LAB))

    m = sub.add_parser("match", help="the sensor's matcher on every ordered pair of templates")
    m.add_argument("--dir", default=str(LAB))
    m.add_argument("--images", default=str(HERE / "template_pairs"),
                   help="where each template's source image is (<name>_raw.png or <name>.png)")

    e = sub.add_parser("evaluate", help="the PC matcher on every pair of templates (offline)")
    e.add_argument("--dir", default=str(LAB))

    c = sub.add_parser("capture", help="live presses -> labelled templates")
    c.add_argument("--label", required=True)
    c.add_argument("--samples", type=int, default=5)
    c.add_argument("--out", default=str(LAB / "live"))

    k = sub.add_parser("compare", help="template matcher vs SIFT on the same presses (offline)")
    k.add_argument("--gallery", required=True, help="enrolled presses (.tmpl + _raw.png)")
    k.add_argument("--queries", help="presses from another session; omit to test the gallery against itself")
    k.add_argument("--images", help="folder with the _raw.png images, if not next to the templates")

    ct = sub.add_parser("confirmtest", help="the sensor double-check on recorded presses (needs the ESP32)")
    ct.add_argument("--gallery", default=str(LAB / "session_a"))
    ct.add_argument("--queries", default=str(LAB / "session_b"))
    ct.add_argument("--templates", type=int, default=3, help="stored sensor templates per finger")

    i = sub.add_parser("identify", help="live press -> identified on the PC")
    i.add_argument("--gallery", default=str(LAB / "live"))
    i.add_argument("--trials", type=int, default=5)
    i.add_argument("--confirm", action="store_true",
                   help="also run the sensor's own Match against the best candidates")
    i.add_argument("--confirm-top", type=int, default=2)

    args = ap.parse_args()
    if args.cmd == "evaluate":
        return evaluate(args)
    if args.cmd == "compare":
        return compare(args)
    if args.cmd == "confirmtest":
        return confirmtest(args)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 0))
    target = (args.esp, args.cmd_port)
    try:
        return {"build": build, "match": match, "capture": capture, "identify": identify}[args.cmd](
            sock, target, args)
    finally:
        sock.close()


if __name__ == "__main__":
    sys.exit(main())
