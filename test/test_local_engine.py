#!/usr/bin/env python3
"""
Accuracy benchmark for the pure-Python engine (fingerprint_local.py).

It reports the two numbers that decide whether a fingerprint system is usable:

    d'   separation between genuine and impostor scores (>3 usable, >4 good)
    EER  equal error rate, where false accepts and false rejects balance

Two sources of labelled data are understood:

  1. test/r503_captures/monitor_<date>_<time>_<ms>.png
     Unlabelled captures from the live sensor.  Identity is inferred from the
     capture CLOCK, deliberately in the pessimistic direction:
        genuine  - two captures in the same burst, <= 20 s apart.  Nobody
                   swaps fingers between presses that close together.
        impostor - captures from bursts more than 60 s apart.  If two such
                   bursts are really the same finger, that inflates the
                   impostor scores, so the true result can only be better
                   than what this prints - never worse.

  2. fingerprint_storage/images/<label>_sample_<n>.png
     Explicitly labelled enrollment captures; <label> is the identity.

Usage:
    py -3 test/test_local_engine.py                 # both datasets
    py -3 test/test_local_engine.py --dir <folder>
"""

import datetime
import itertools
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(str(Path(__file__).resolve().parent.parent))

from fingerprint_local import (  # noqa: E402
    LocalEngine, build_template, parse_xyt, match_templates, is_pre_thinned,
)


def _metrics(gen, imp):
    gen, imp = np.asarray(gen, float), np.asarray(imp, float)
    if len(gen) == 0 or len(imp) == 0:
        return None
    pooled = np.sqrt((gen.var() + imp.var()) / 2) or 1e-9
    d = abs(gen.mean() - imp.mean()) / pooled
    best = None
    for t in np.unique(np.concatenate([gen, imp])):
        far, frr = (imp >= t).mean(), (gen < t).mean()
        if best is None or abs(far - frr) < abs(best[1] - best[2]):
            best = (t, far, frr)
    return {
        "d": d, "eer": (best[1] + best[2]) / 2 * 100, "eer_th": best[0],
        "gen": gen, "imp": imp,
    }


def _report(name, gen, imp, threshold):
    m = _metrics(gen, imp)
    if m is None:
        print(f"  {name}: not enough pairs")
        return
    gen, imp = m["gen"], m["imp"]
    print(f"\n  {name}: {len(gen)} genuine / {len(imp)} impostor pairs")
    print(f"    genuine  matched minutiae: mean {gen.mean():5.1f}  "
          f"p5 {np.percentile(gen,5):.0f}  median {np.percentile(gen,50):.0f}")
    print(f"    impostor matched minutiae: mean {imp.mean():5.1f}  "
          f"p95 {np.percentile(imp,95):.0f}  p99 {np.percentile(imp,99):.0f}  max {imp.max():.0f}")
    print(f"    d' = {m['d']:.2f}   EER = {m['eer']:.1f}% (at {m['eer_th']:.0f} matched)")
    print(f"    at LOCAL_MATCH_THRESHOLD={threshold}: "
          f"FAR = {(imp >= threshold).mean()*100:.2f}%   "
          f"FRR = {(gen < threshold).mean()*100:.1f}%")
    print("    operating points:")
    for th in range(int(max(4, imp.mean())), int(gen.max()) + 2):
        far, frr = (imp >= th).mean() * 100, (gen < th).mean() * 100
        print(f"      accept at >={th:2d} matched:  FAR {far:6.2f}%   FRR {frr:6.1f}%")


def _templates(paths, engine):
    t0 = time.time()
    tpls, counts = [], []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        t = engine.extract_template(img)
        tpls.append(build_template(parse_xyt(t["xyt"])) if t else None)
        counts.append(t["minutiae_count"] if t else 0)
    el = (time.time() - t0) / max(len(paths), 1)
    print(f"  {len(paths)} images, {el*1000:.0f} ms/image, "
          f"minutiae min {min(counts)} / mean {np.mean(counts):.1f} / max {max(counts)}"
          f"  [{sum(1 for t in tpls if t is None)} unusable]")
    return tpls


def bench_r503(folder, engine):
    paths = sorted(Path(folder).glob("monitor_*.png"))
    if not paths:
        return
    print(f"\n{'='*66}\nR503 live captures ({folder}) - labels inferred from capture time\n{'='*66}")
    times = [datetime.datetime.strptime(p.name[8:23], "%Y%m%d_%H%M%S") for p in paths]
    burst = [0]
    for k in range(1, len(paths)):
        burst.append(burst[-1] + (1 if (times[k] - times[k - 1]).total_seconds() > 40 else 0))
    print(f"  {len(set(burst))} capture bursts: "
          f"{ {b: burst.count(b) for b in sorted(set(burst))} }")
    tpls = _templates(paths, engine)

    gen, imp = [], []
    for i, j in itertools.combinations(range(len(paths)), 2):
        if tpls[i] is None or tpls[j] is None:
            continue
        dt = abs((times[j] - times[i]).total_seconds())
        same = burst[i] == burst[j]
        if same and dt <= 20:
            gen.append(match_templates(tpls[i], tpls[j]))
        elif not same and dt >= 60:
            imp.append(match_templates(tpls[i], tpls[j]))
    _report("R503 (time-inferred labels, pessimistic)", gen, imp, engine.threshold)


def bench_labelled(folder, engine):
    groups = defaultdict(list)
    for p in sorted(Path(folder).glob("*_sample_*.png")):
        groups[p.name.split("_sample_")[0]].append(p)
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not groups:
        return
    paths = [p for v in groups.values() for p in v]
    sample = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
    print(f"\n{'='*66}\nLabelled captures ({folder})\n{'='*66}")
    print(f"  {len(groups)} identities, {sample.shape[1]}x{sample.shape[0]}, "
          f"{'pre-thinned' if is_pre_thinned(sample) else 'grayscale'}")
    tpls = {}
    flat = []
    for gid, ps in groups.items():
        flat.extend(ps)
    allt = _templates(flat, engine)
    k = 0
    for gid, ps in groups.items():
        tpls[gid] = allt[k:k + len(ps)]
        k += len(ps)

    gen, imp = [], []
    for gid, ts in tpls.items():
        for a, b in itertools.combinations(range(len(ts)), 2):
            if ts[a] is not None and ts[b] is not None:
                gen.append(match_templates(ts[a], ts[b]))
    for g1, g2 in itertools.combinations(tpls, 2):
        for t1 in tpls[g1]:
            for t2 in tpls[g2]:
                if t1 is not None and t2 is not None:
                    imp.append(match_templates(t1, t2))
    _report("labelled", gen, imp, engine.threshold)

    # rank-1 identification, leave one out
    ids = list(tpls)
    correct = total = 0
    for pid in ids:
        for k in range(len(tpls[pid])):
            if tpls[pid][k] is None:
                continue
            total += 1
            best, who = -1, None
            for gid in ids:
                for j, t in enumerate(tpls[gid]):
                    if t is None or (gid == pid and j == k):
                        continue
                    s = match_templates(tpls[pid][k], t)
                    if s > best:
                        best, who = s, gid
            correct += (who == pid)
    if total:
        print(f"    rank-1 identification: {correct}/{total} = {correct/total*100:.1f}%")


if __name__ == "__main__":
    engine = LocalEngine()
    if "--dir" in sys.argv:
        d = sys.argv[sys.argv.index("--dir") + 1]
        bench_r503(d, engine)
        bench_labelled(d, engine)
    else:
        bench_r503("test/r503_captures", engine)
        bench_labelled("fingerprint_storage/images", engine)
    print("\nRaise LOCAL_MATCH_THRESHOLD for fewer false accepts, lower it for "
          "fewer false rejects.\n")
