#!/usr/bin/env python3
"""
Decoder for the R503's own fingerprint template, so matching can run on the
PC / OrangePi with fingerprint_local's matcher instead of on the sensor.

Measured on this unit: ReadProdInfo "R5xxS", firmware "Ver 2.01 GROW", sensor
BF5325S, 160x160. Another firmware may lay the template out differently -
decode() checks the fixed bytes and refuses anything that does not fit.

How it was decoded (2026-10-19, test/r503_template_lab.py): the sensor built
templates from 25 recorded presses, and each template was compared with the
skeleton the sensor drew from the same image.

* x/y: every decoded minutia lies on average 0.6 px from a ridge ending or
  bifurcation of that skeleton (265 minutiae).
* type: bit 0 separates endings from bifurcations without exception.
* direction: agrees with the skeleton's ridge direction to R = 0.994
  (residual spread 6.5 degrees).
* orientation map: agrees with the image's block ridge direction to R = 0.974.

Layout (UpChar after RegModel returns 2048 bytes; everything after the last
record is zero). Img2Tz alone exports only zeros on this firmware - a template
exists only after RegModel, even of one image with itself.

    0..12    header, constant: 71 04 50 50 41 03 0a be 2c 43 34 0a 0a
             (the last two bytes are the orientation grid size, 10 x 10)
    13..62   ridge orientation per 16x16-px block, one nibble each, row-major,
             high nibble first: ridge direction ~= value * 11.25 deg (mod 180),
             0xF = no print in that block
    63..66   44 01 00 45  (constant)
    67       N, the number of minutiae
    68..     N records of 5 bytes, each read as one 40-bit big-endian number:
               bits 39..31  x, pixels
               bits 30..22  y, pixels (downward)
               bits 21..16  quality, 26..61 observed (records are always stored
                            in descending order of it)
               bits 15..8   direction, 1.5 degree units (0..239), pointing into
                            the ridge
               bits  7..1   meaning unknown (1..88 observed)
               bit   0      0 = ridge ending, 1 = bifurcation

decode() returns minutiae in fingerprint_local's convention, (x, y, theta,
kind, quality) with theta in radians pointing away from the ridge body and
kind 1 = ending, 3 = bifurcation, so its matcher and .xyt storage take them
directly.

match() is a matcher written for these templates: ~10 minutiae with exact
angles, plus the orientation map. It aligns the two prints on every minutia
pair, counts the minutiae that land on a compatible partner (position,
direction, type and the 7-bit field), and multiplies that count by how well
the two orientation maps agree once aligned - a chance alignment of two
different fingers rarely survives the last check. On 25 presses of 5 fingers,
5 samples per finger, it identified 25/25 with no genuine press scoring below
the best impostor; fingerprint_local's matcher on the same minutiae identified
24/25 with six. One person's fingers - not yet a population measurement.

    py -3 r503_template.py test/template_lab/*.tmpl     # print what decodes
"""

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# Accept when match() scores at least this. Measured on 5 fingers x 5 presses
# (one person), each press against every finger's other samples: the best
# impostor scored 2.89, and no impostor was accepted anywhere from 3.0 to 4.5;
# at 4.0, 1 of 25 genuine presses was rejected. 4.0 sits mid-gap for headroom.
# Provisional - 100 impostor comparisons cannot show a false-accept rate below
# a few percent. Re-measure on more people (r503_template_lab.py evaluate).
R503_MATCH_THRESHOLD = float(os.environ.get("R503_MATCH_THRESHOLD", "4.0"))

# A minutia pair counts when it lands within these after alignment.
R503_DIST_TOL = float(os.environ.get("R503_DIST_TOL", "10.0"))
R503_ANG_TOL = math.radians(float(os.environ.get("R503_ANG_TOL_DEG", "20.0")))

# ...and only if the two minutiae also agree on what the sensor recorded about
# them: the same type, and the 7-bit field (bits 7..1) within this. Measured on
# minutiae paired across presses of the same finger, that field correlates
# 0.94 (median difference 1) against 0.04 for chance pairings across fingers,
# and the type agrees 96% vs 57%. Together they cut the best impostor score
# from 4.3 to 2.8 while genuine scores barely moved. -1 disables the field check.
R503_EXTRA_TOL = float(os.environ.get("R503_EXTRA_TOL", "3"))
R503_SAME_TYPE = os.environ.get("R503_SAME_TYPE", "1") == "1"

TEMPLATE_BYTES = 2048
HEADER = bytes.fromhex("71 04 50 50 41 03 0a be 2c 43 34 0a 0a")
SECTION = bytes.fromhex("44010045")
GRID = 10                       # orientation blocks per side
RECORDS_AT = 68
RECORD_BYTES = 5
ANGLE_UNIT = math.radians(1.5)


def decode(data: bytes) -> Dict:
    """Template bytes -> {'minutiae', 'orientation', 'records'}. Raises ValueError."""
    data = bytes(data)
    if len(data) < RECORDS_AT or data[:len(HEADER)] != HEADER or data[63:67] != SECTION:
        raise ValueError("not an R503 (R5xxS, Ver 2.01) template - empty buffer or another firmware?")
    n = data[67]
    end = RECORDS_AT + RECORD_BYTES * n
    if end > len(data):
        raise ValueError(f"template claims {n} minutiae but holds only {len(data)} bytes")

    grid = np.frombuffer(data[13:63], np.uint8)
    orientation = np.stack([grid >> 4, grid & 0x0F], axis=1).reshape(GRID, GRID)

    minutiae: List[Tuple[int, int, float, int, int]] = []
    records = []
    for i in range(n):
        v = int.from_bytes(data[RECORDS_AT + RECORD_BYTES * i: RECORDS_AT + RECORD_BYTES * (i + 1)], "big")
        x, y = (v >> 31) & 0x1FF, (v >> 22) & 0x1FF
        quality, angle, extra, bif = (v >> 16) & 0x3F, (v >> 8) & 0xFF, (v >> 1) & 0x7F, v & 1
        theta = (angle * ANGLE_UNIT + math.pi) % (2 * math.pi)
        minutiae.append((x, y, theta, 3 if bif else 1, round(quality * 100 / 63)))
        records.append({"x": x, "y": y, "angle": angle, "quality": quality,
                        "extra": extra, "bifurcation": bool(bif)})
    return {"minutiae": minutiae, "orientation": orientation, "records": records}


def is_template(data: bytes) -> bool:
    try:
        decode(data)
        return True
    except ValueError:
        return False


def to_xyt(data: bytes) -> str:
    """Template bytes -> NIST .xyt text, the format fingerprint_local stores."""
    from fingerprint_local import minutiae_to_xyt
    return minutiae_to_xyt(decode(data)["minutiae"])


# ==========================================================================
# Matching
# ==========================================================================

def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def prepare(data: bytes) -> Dict:
    """Decode once into the arrays match() needs; cache this per enrolled template."""
    t = decode(data)
    m = np.array([(x, y, th) for x, y, th, _k, _q in t["minutiae"]], float).reshape(-1, 3)
    o = t["orientation"].astype(float)
    return {"xy": m[:, :2], "theta": m[:, 2],
            "bif": np.array([r["bifurcation"] for r in t["records"]], bool),
            "extra": np.array([r["extra"] for r in t["records"]], float),
            "ori": np.where(o == 0x0F, np.nan, o * math.radians(11.25))}


def _orientation_agreement(a: Dict, b: Dict, rot: float, tx: float, ty: float) -> float:
    """Mean cos(2 * difference) of the block orientations both prints cover, after aligning a onto b."""
    c, s = math.cos(rot), math.sin(rot)
    yy, xx = np.mgrid[0:10, 0:10]
    cx, cy = xx * 16 + 8.0, yy * 16 + 8.0
    bi = np.floor((s * cx + c * cy + ty) / 16).astype(int)
    bj = np.floor((c * cx - s * cy + tx) / 16).astype(int)
    inside = (bi >= 0) & (bi < 10) & (bj >= 0) & (bj < 10) & ~np.isnan(a["ori"])
    oa = a["ori"][inside] + rot
    ob = b["ori"][bi[inside], bj[inside]]
    both = ~np.isnan(ob)
    if both.sum() < 4:
        return 0.0
    return float(np.mean(np.cos(2 * (oa[both] - ob[both]))))


def _pair_count(ok: np.ndarray, d: np.ndarray) -> int:
    """One-to-one pairs among the candidates in ok, closest first."""
    pairs, used_a, used_b = 0, set(), set()
    for k in np.argsort(np.where(ok, d, np.inf), axis=None):
        p, q = divmod(int(k), ok.shape[1])
        if not ok[p, q]:
            break
        if p in used_a or q in used_b:
            continue
        used_a.add(p)
        used_b.add(q)
        pairs += 1
    return pairs


def match(a: Dict, b: Dict, dist_tol: float = R503_DIST_TOL, ang_tol: float = R503_ANG_TOL,
          extra_tol: float = R503_EXTRA_TOL, same_type: bool = R503_SAME_TYPE) -> Tuple[float, int, float]:
    """(score, paired minutiae, orientation agreement) for two prepare()d templates."""
    n, m = len(a["theta"]), len(b["theta"])
    if n < 3 or m < 3:
        return 0.0, 0, 0.0
    compat = np.ones((n, m), bool)                  # minutiae that may pair at all
    if same_type:
        compat &= a["bif"][:, None] == b["bif"][None, :]
    if extra_tol >= 0:
        compat &= np.abs(a["extra"][:, None] - b["extra"][None, :]) <= extra_tol
    # every alignment that puts one minutia of a exactly on a compatible one of b
    ai, bj = np.nonzero(compat)
    if len(ai) == 0:
        return 0.0, 0, 0.0
    rot = _wrap(b["theta"][bj] - a["theta"][ai])
    c, s = np.cos(rot)[:, None], np.sin(rot)[:, None]
    ax0, ay0 = a["xy"][:, 0][None, :], a["xy"][:, 1][None, :]
    tx = b["xy"][bj, 0][:, None] - (c * a["xy"][ai, 0][:, None] - s * a["xy"][ai, 1][:, None])
    ty = b["xy"][bj, 1][:, None] - (s * a["xy"][ai, 0][:, None] + c * a["xy"][ai, 1][:, None])
    ax, ay = c * ax0 - s * ay0 + tx, s * ax0 + c * ay0 + ty           # (K, n)
    at = a["theta"][None, :] + rot[:, None]
    d = np.hypot(ax[:, :, None] - b["xy"][:, 0], ay[:, :, None] - b["xy"][:, 1])   # (K, n, m)
    ok = compat & (d <= dist_tol) & (np.abs(_wrap(at[:, :, None] - b["theta"])) <= ang_tol)
    upper = np.minimum(ok.any(axis=2).sum(axis=1), ok.any(axis=1).sum(axis=1))

    best = (0, 0.0)
    for k in np.argsort(-upper, kind="stable"):
        if upper[k] < max(best[0], 2):
            break
        pairs = _pair_count(ok[k], d[k])
        if pairs < max(best[0], 2):
            continue
        agree = _orientation_agreement(a, b, float(rot[k]), float(tx[k, 0]), float(ty[k, 0]))
        if (pairs, agree) > best:
            best = (pairs, agree)
    pairs, agree = best
    return pairs * max(agree, 0.0), pairs, agree


def identify(query: Dict, gallery, threshold: float = R503_MATCH_THRESHOLD):
    """gallery: iterable of (key, prepare()d template). Returns (key or None, best score, runner-up score).

    Keys may repeat (several samples per finger); each key scores its best sample.
    """
    best: Dict = {}
    for key, tmpl in gallery:
        score = match(query, tmpl)[0]
        if score > best.get(key, -1.0):
            best[key] = score
    if not best:
        return None, 0.0, 0.0
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    top_key, top = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    return (top_key if top >= threshold else None), top, runner_up


def main(paths) -> int:
    if not paths:
        print(__doc__)
        return 2
    for p in paths:
        try:
            t = decode(open(p, "rb").read())
        except (OSError, ValueError) as exc:
            print(f"{p}: {exc}")
            continue
        m = t["minutiae"]
        bif = sum(k == 3 for *_, k, _q in m)
        print(f"{p}: {len(m)} minutiae ({len(m) - bif} endings, {bif} bifurcations)")
        for x, y, th, k, q in m:
            print(f"   x {x:3d}  y {y:3d}  {math.degrees(th):5.1f} deg  "
                  f"{'bifurcation' if k == 3 else 'ending':11s}  q {q}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
