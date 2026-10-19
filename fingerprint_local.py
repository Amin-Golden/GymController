#!/usr/bin/env python3
"""
Pure-Python fingerprint engine: minutiae extraction + local-structure matching.

Why this module exists
----------------------
fingerprint_nbis.py needs NIST's mindtct/bozorth3 native binaries, which cannot
be installed on every target (an OrangePi 5B / arm64 board in particular).  This
module does the same job with nothing but numpy and opencv-python, so it runs
anywhere the rest of the system already runs.

Matching algorithm
-------------------
Two prints are compared by finding one confidently-matching pair of minutiae
(a "root"), then growing the match outward: a neighbour of one minutia is
accepted as corresponding to a neighbour of the other only if ITS geometric
relationship to an ALREADY-CONFIRMED pair also lines up. This is the same
family of algorithm real AFIS systems use (NIST's own Bozorth3 included),
and it is deliberately NOT a single global rotation+translation fit: a small
optical sensor captures a different patch of skin on every press, and that
skin stretches and shears differently in different parts of the same image,
so demanding one rigid transform for every minutia throws away correct
correspondences a local model would accept.

IMPORTANT - read before trusting this for access control: this was measured
against a global-transform matcher on the same real captures and only won by
about 30% relative (d' 0.69 -> 0.91, rank-1 58% -> 64% on 10 fingers). That
modest gap, not a dramatic one, is itself evidence: the ceiling here is set by
how few minutiae this sensor's output contains (~15 per press), not by which
matching algorithm reads them. See FINGERPRINT_DIAGNOSIS.md.

It is a drop-in replacement for fingerprint_nbis:

    engine = LocalEngine()
    template = engine.extract_template(image)     # {'xyt', 'minutiae_count', 'quality'}
    result  = identify(engine, image, profiles, gallery)

Templates are written in NIST .xyt text format ("x y theta quality" per line,
theta in degrees), so they share the xyt_template database column with NBIS and
the two engines can read each other's enrollments.

Input images
------------
Two very different kinds of image reach us, and they need different front ends:

  * ALREADY THINNED - the R503 returns a binary ridge skeleton (the sensor runs
    its own enhancement).  Detected by the pixel histogram; used directly.
  * GRAYSCALE - older sensors (FPM383C) send a real gray image.  It gets
    normalisation, a Gabor ridge filter, binarisation and thinning first.

Self test:
    py -3 fingerprint_local.py --selftest
    py -3 fingerprint_local.py --bench test/r503_captures
"""

import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Tunables (environment overridable)
# --------------------------------------------------------------------------

# Matched-minutiae count required to accept. Measured on 10 real fingers / 50
# presses (test/test_enroll_identify.py evaluate): genuine pairs average 6.1
# matched minutiae, impostor pairs average 3.5, and NO threshold gives both a
# low false-accept and a low false-reject rate - at >=6 FAR is 3.1% but FRR is
# 58%; at >=8 FAR reaches 0% but FRR is 76%. 6 is the best balance found, not
# a safe operating point. See FINGERPRINT_DIAGNOSIS.md section "Why this has a
# hard ceiling" before trusting this threshold for real access control.
LOCAL_MATCH_THRESHOLD = int(os.environ.get("LOCAL_MATCH_THRESHOLD", "6"))

# Minutiae required to ACCEPT a sample during enrollment (a stricter, separate
# gate from the 6-minutiae floor extract_template() applies to everything,
# including a live identify query). Raised from 12 to 16 on 2026-09:
# cross-checking the 10-person / 50-sample dataset, samples below 16 minutiae
# misidentify at 40% vs 34% at >=16 - real, but a smaller effect than minutiae
# count alone explains (several wrong matches happen at 20+ minutiae too, e.g.
# a finger named 'r3' in that dataset is wrong at 21 minutiae). Raising this
# reduces, but does not eliminate, wrong matches - re-enrolling an existing
# weak finger is still worth doing, but is not by itself a fix. A finger that
# consistently can't clear this within ENROLL_RETRY_BUDGET presses (in
# fingerprint_gym.py) is one this sensor genuinely cannot capture well.
# Lowered to 10 on 2026-10 with multi-press enrollment: presses are now
# stitched into one whole-finger template (fuse_presses), so a press that only
# shows a smooth part of the finger - genuinely few minutiae, e.g. the core -
# still adds coverage instead of being rejected. 160x160 R503 presses measure
# 12-57 minutiae (median 18) on the labelled dataset.
MIN_ENROLL_MINUTIAE = int(os.environ.get("MIN_ENROLL_MINUTIAE", "10"))

# The winner must beat the runner-up client by this ratio (1.0 disables).
LOCAL_MATCH_MARGIN = float(os.environ.get("LOCAL_MATCH_MARGIN", "1.25"))

# Neighbours per minutia in the local-structure graph (see match_templates).
LOCAL_NEIGHBORS = int(os.environ.get("LOCAL_NEIGHBORS", "6"))

# Edge-compatibility tolerances for the graph matcher.
LOCAL_DIST_TOL = float(os.environ.get("LOCAL_DIST_TOL", "6.0"))
LOCAL_ANG_TOL_DEG = float(os.environ.get("LOCAL_ANG_TOL_DEG", "20.0"))

# Ignore minutiae this close to the image border: a ridge cut by the edge of
# the capture window looks exactly like a ridge ending and is not repeatable.
LOCAL_BORDER = int(os.environ.get("LOCAL_BORDER", "6"))

# Minutiae closer together than this are a binarisation artefact, not two
# separate features. 8 px is about one ridge period on a 500 dpi sensor.
LOCAL_DEDUPE_DIST = float(os.environ.get("LOCAL_DEDUPE_DIST", "8.0"))

# These matching defaults come from evaluate() in test/test_enroll_identify.py
# against 10 real fingers / 50 presses. Re-run it after changing sensor, and
# after enrolling a meaningfully larger population.

_NB = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


# ==========================================================================
# Image front end
# ==========================================================================

def _neighbors(img: np.ndarray) -> List[np.ndarray]:
    p = np.pad(img, 1)
    h, w = img.shape
    return [p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in _NB]


def crossing_number(skel: np.ndarray) -> np.ndarray:
    """Rutovitz crossing number: 1 = ridge ending, >=3 = bifurcation."""
    P = _neighbors(skel)
    cn = 0.5 * sum(np.abs(P[i].astype(np.int16) - P[(i + 1) % 8].astype(np.int16))
                   for i in range(8))
    return cn.astype(np.int8)


def is_pre_thinned(image: np.ndarray) -> bool:
    """True when the sensor already returned a binary ridge skeleton."""
    vals = np.unique(image)
    if len(vals) > 4:
        return False
    # a skeleton is mostly background: thin lines cover well under half the frame
    dark = float((image < 128).mean())
    return dark < 0.45


def _normalize(img: np.ndarray, m0=100.0, v0=10000.0) -> np.ndarray:
    f = img.astype(np.float32)
    m, v = f.mean(), f.var()
    if v < 1e-6:
        return np.full_like(f, m0)
    d = np.sqrt(v0 * (f - m) ** 2 / v)
    return np.where(f > m, m0 + d, m0 - d)


def _orientation(img: np.ndarray, block: int = 8) -> np.ndarray:
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    k = block * 2 + 1
    gxx = cv2.GaussianBlur(gx * gx, (k, k), block / 2.0)
    gyy = cv2.GaussianBlur(gy * gy, (k, k), block / 2.0)
    gxy = cv2.GaussianBlur(gx * gy, (k, k), block / 2.0)
    ang2 = np.arctan2(2 * gxy, gxx - gyy)
    cs = cv2.GaussianBlur(np.cos(ang2), (k, k), block / 2.0)
    sn = cv2.GaussianBlur(np.sin(ang2), (k, k), block / 2.0)
    return np.mod(np.arctan2(sn, cs) / 2.0 + np.pi / 2.0, np.pi)


def _ridge_period(img: np.ndarray) -> float:
    """Dominant ridge wavelength in pixels, from the image power spectrum."""
    f = img - img.mean()
    win = np.outer(np.hanning(f.shape[0]), np.hanning(f.shape[1]))
    mag = np.abs(np.fft.fftshift(np.fft.fft2(f * win)))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - cy, xx - cx).astype(int)
    prof = np.bincount(r.ravel(), mag.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    lo, hi = 3, min(cy, cx)
    band = prof[lo:hi]
    if band.size == 0:
        return 9.0
    k = lo + int(np.argmax(band))
    period = max(h, w) / max(k, 1)
    return float(np.clip(period, 5.0, 16.0))


def _thin(binary: np.ndarray, max_iter: int = 60) -> np.ndarray:
    """Zhang-Suen thinning, vectorised."""
    img = (binary > 0).astype(np.uint8)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            P = _neighbors(img)
            B = sum(P)
            seq = P + [P[0]]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c1, c2 = P[0] * P[2] * P[4], P[2] * P[4] * P[6]
            else:
                c1, c2 = P[0] * P[2] * P[6], P[0] * P[4] * P[6]
            cond = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & (c1 == 0) & (c2 == 0)
            if cond.any():
                img[cond] = 0
                changed = True
        if not changed:
            break
    return img


def to_skeleton(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (skeleton {0,1}, foreground mask {0,1})."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.uint8)

    if is_pre_thinned(image):
        skel = (image < 128).astype(np.uint8)
        mask = np.ones_like(skel)
        return skel, mask

    # grayscale path: normalise -> Gabor along the ridge flow -> binarise -> thin
    norm = _normalize(image)
    period = _ridge_period(norm)
    freq = 1.0 / period
    orient = _orientation(norm)

    sigma = 0.65 * period
    half = int(np.ceil(2.5 * sigma))
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    n_or = 16
    bank = []
    for i in range(n_or):
        th = np.pi * i / n_or
        xr = x * np.cos(th) + y * np.sin(th)
        yr = -x * np.sin(th) + y * np.cos(th)
        g = np.exp(-0.5 * (xr ** 2 + yr ** 2) / sigma ** 2) * np.cos(2 * np.pi * freq * xr)
        bank.append((g - g.mean()).astype(np.float32))
    resp = np.stack([cv2.filter2D(norm, cv2.CV_32F, k) for k in bank])
    idx = (np.round(orient / np.pi * n_or).astype(np.int32) % n_or)
    yy, xx = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    enhanced = resp[idx, yy, xx]

    # foreground: where there is real ridge contrast
    sd = cv2.GaussianBlur(np.abs(enhanced), (0, 0), period)
    mask = (sd > 0.25 * sd.max()).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    binary = ((enhanced > 0) & (mask > 0)).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= max(8, period)
    binary = keep[lbl].astype(np.uint8)
    return _thin(binary), mask


def prune_spurs(skel: np.ndarray, spur_len: int = 8, min_component: int = 6) -> np.ndarray:
    """Remove islands and spurs (short branches hanging off a bifurcation).

    Only branches that run from a free end INTO a bifurcation within spur_len
    pixels are cut.  Shaving every endpoint instead would also eat the genuine
    ridge endings, which are the minutiae we are trying to find.
    """
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(skel, 8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_component
    skel = keep[lbl].astype(np.uint8)

    h, w = skel.shape
    for _ in range(3):
        cn = crossing_number(skel)
        ends = np.argwhere((skel == 1) & (cn == 1))
        if len(ends) == 0:
            break
        remove = []
        for y, x in ends:
            path = [(y, x)]
            prev, cur = (y, x), (y, x)
            hit = False
            for _ in range(spur_len):
                nxt = None
                for dy, dx in _NB:
                    ny, nx = cur[0] + dy, cur[1] + dx
                    if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] and (ny, nx) != prev:
                        nxt = (ny, nx)
                        break
                if nxt is None:
                    break
                if cn[nxt] >= 3:
                    hit = True
                    break
                path.append(nxt)
                prev, cur = cur, nxt
            if hit:
                remove.extend(path)
        if not remove:
            break
        for y, x in remove:
            skel[y, x] = 0
    return skel


def _walk(skel: np.ndarray, x: int, y: int, steps: int = 10):
    h, w = skel.shape
    prev, cur, n = (y, x), (y, x), 0
    for _ in range(steps):
        nxt = None
        for dy, dx in _NB:
            ny, nx = cur[0] + dy, cur[1] + dx
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] and (ny, nx) != prev:
                nxt = (ny, nx)
                break
        if nxt is None:
            break
        prev, cur, n = cur, nxt, n + 1
    if cur == (y, x):
        return None
    # point away from the ridge body, image coords (y grows downward)
    return float(np.mod(np.arctan2(-(cur[0] - y), -(cur[1] - x)), 2 * np.pi)), n


def extract_minutiae(image: np.ndarray, border: int = LOCAL_BORDER,
                     spur_len: int = 8, min_component: int = 6,
                     dedupe_dist: float = LOCAL_DEDUPE_DIST):
    """Return (minutiae, skeleton). Minutia = (x, y, theta_rad, kind, quality)."""
    skel, mask = to_skeleton(image)
    skel = prune_spurs(skel, spur_len=spur_len, min_component=min_component)
    cn = crossing_number(skel)
    h, w = skel.shape

    inner = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=border)
    inner[:border, :] = 0
    inner[h - border:, :] = 0
    inner[:, :border] = 0
    inner[:, w - border:] = 0

    orient = _orientation(skel.astype(np.float32) * 255.0, block=10)
    out = []
    for y, x in np.argwhere((skel == 1) & (cn == 1) & (inner == 1)):
        d = _walk(skel, int(x), int(y))
        if d is None or d[1] < 4:
            continue
        out.append((int(x), int(y), d[0], 1, min(100, 40 + 6 * d[1])))
    for y, x in np.argwhere((skel == 1) & (cn >= 3) & (inner == 1)):
        out.append((int(x), int(y), float(orient[y, x]), 3, 60))

    return _dedupe(out, dedupe_dist), skel


def _dedupe(m, min_dist: float = 6.0):
    """Two minutiae closer than one ridge width are a binarisation artefact."""
    if not m:
        return []
    pts = np.array([(a[0], a[1]) for a in m], float)
    drop = np.zeros(len(m), bool)
    for i in range(len(m)):
        if drop[i]:
            continue
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        for j in np.where(d < min_dist)[0]:
            if j != i:
                drop[j] = True
    return [m[i] for i in range(len(m)) if not drop[i]]


# ==========================================================================
# Template (NIST .xyt text) and matching
# ==========================================================================

def minutiae_to_xyt(minutiae) -> str:
    """NIST .xyt: 'x y theta quality', theta in degrees."""
    lines = []
    for x, y, th, _kind, q in minutiae:
        lines.append(f"{int(x)} {int(y)} {int(round(np.degrees(th))) % 360} {int(q)}")
    return "\n".join(lines) + ("\n" if lines else "")


def parse_xyt(text: str):
    """Parse .xyt text into (x, y, theta_rad, kind, quality). Kind is unknown -> 0."""
    out = []
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            x, y, t = int(parts[0]), int(parts[1]), float(parts[2])
            q = int(parts[3]) if len(parts) > 3 else 50
        except ValueError:
            continue
        out.append((x, y, np.radians(t), 0, q))
    return out


def _wrap(a):
    return np.mod(a + np.pi, 2 * np.pi) - np.pi


def build_template(minutiae, k_neighbors: int = LOCAL_NEIGHBORS,
                   max_radius: float = 100.0) -> Optional[Dict]:
    """Local-structure template: for every minutia, its k nearest neighbours
    described by (length, bearing-from-self, bearing-from-neighbour) - all
    invariant to the print's rotation and translation, so matching needs no
    rotation search (contrast the old global-pair-table version, which did).
    """
    n = len(minutiae)
    if n < 3:
        return None
    m = np.asarray([(x, y, t) for x, y, t, _k, _q in minutiae], float)
    xy, th = m[:, :2], m[:, 2]

    dx = xy[:, 0][None, :] - xy[:, 0][:, None]
    dy = xy[:, 1][None, :] - xy[:, 1][:, None]
    dist = np.hypot(dx, dy)
    np.fill_diagonal(dist, np.inf)
    bearing = np.arctan2(dy, dx)                        # i -> j
    ref_i = _wrap(bearing - th[:, None])                 # relative to i's own ridge direction
    ref_j = _wrap((bearing + np.pi) - th[None, :])        # relative to j's own ridge direction

    k = min(k_neighbors, n - 1)
    order = np.argsort(dist, axis=1)[:, :k]
    nb_len = np.take_along_axis(dist, order, axis=1)
    nb_ai = np.take_along_axis(ref_i, order, axis=1)
    nb_aj = np.take_along_axis(ref_j, order, axis=1)
    nb_valid = nb_len <= max_radius
    nb_len = np.where(nb_valid, nb_len, np.nan)

    # Python-side adjacency for the breadth-first growth phase, which only
    # ever touches a handful of neighbours per step and is not worth
    # vectorising; the expensive all-pairs ranking below is.
    adjacency = []
    for i in range(n):
        d = {}
        for p in range(k):
            if nb_valid[i, p]:
                d[int(order[i, p])] = (float(nb_len[i, p]), float(nb_ai[i, p]), float(nb_aj[i, p]))
        adjacency.append(d)

    return {
        "xy": xy, "theta": th, "n": n, "k": k,
        "nb_len": nb_len, "nb_ai": nb_ai, "nb_aj": nb_aj, "nb_valid": nb_valid,
        "adjacency": adjacency,
    }


def _root_candidates(t1, t2, dist_tol, dist_rel, ang_tol, min_sim=2, top=12):
    """Rank every minutia pair by how many of minutia i's edges have a
    length/angle-compatible counterpart among minutia j's edges - vectorised
    over every minutia pair and every neighbour-slot pair at once, which is
    what makes this fast enough to search a real gallery (see match_templates
    docstring): the naive per-edge Python-loop version of this exact check
    took ~26 ms/comparison; this takes ~2-4 ms.
    """
    L1 = t1["nb_len"][:, None, :, None]
    L2 = t2["nb_len"][None, :, None, :]
    A1 = t1["nb_ai"][:, None, :, None]
    A2 = t2["nb_ai"][None, :, None, :]
    B1 = t1["nb_aj"][:, None, :, None]
    B2 = t2["nb_aj"][None, :, None, :]
    V1 = t1["nb_valid"][:, None, :, None]
    V2 = t2["nb_valid"][None, :, None, :]

    with np.errstate(invalid="ignore"):
        len_ok = np.abs(L1 - L2) <= (dist_tol + dist_rel * np.minimum(L1, L2))
        a_ok = np.abs(_wrap(A1 - A2)) <= ang_tol
        b_ok = np.abs(_wrap(B1 - B2)) <= ang_tol
    compat = len_ok & a_ok & b_ok & V1 & V2
    compat = np.nan_to_num(compat, nan=False).astype(bool)

    # For each (i, j): how many of i's neighbour slots have >=1 compatible
    # j-slot. An upper-bound approximation (not an exact 1:1 assignment) -
    # fine for RANKING candidate roots; _grow() below enforces exact 1:1.
    sim = compat.any(axis=3).sum(axis=2)

    i_idx, j_idx = np.where(sim >= min_sim)
    if len(i_idx) == 0:
        return []
    order = np.argsort(-sim[i_idx, j_idx])[:top]
    return [(int(i_idx[o]), int(j_idx[o])) for o in order]


def _edge_compatible(e1, e2, dist_tol, dist_rel, ang_tol):
    L1, ai1, aj1 = e1
    L2, ai2, aj2 = e2
    if abs(L1 - L2) > dist_tol + dist_rel * min(L1, L2):
        return False
    if abs(_wrap(ai1 - ai2)) > ang_tol:
        return False
    if abs(_wrap(aj1 - aj2)) > ang_tol:
        return False
    return True


def match_templates(t1, t2, dist_tol: float = LOCAL_DIST_TOL, dist_rel: float = 0.08,
                    ang_tol_deg: float = LOCAL_ANG_TOL_DEG,
                    root_candidates: int = 12) -> int:
    """Number of minutiae placed in correspondence between the two templates.

    Finds the best-looking root pair by local geometry (rotation-free, no
    search needed), then grows a match outward from it: a candidate pair is
    accepted only when its edge to an ALREADY-MATCHED pair is compatible, so
    consistency is always checked against real nearby structure - never one
    rigid transform assumed to explain every minutia in the image at once.
    """
    return len(match_pairs(t1, t2, dist_tol, dist_rel, ang_tol_deg, root_candidates))


def match_pairs(t1, t2, dist_tol: float = LOCAL_DIST_TOL, dist_rel: float = 0.08,
                ang_tol_deg: float = LOCAL_ANG_TOL_DEG,
                root_candidates: int = 12) -> Dict[int, int]:
    """The correspondence behind match_templates(): {minutia in t1: minutia in t2}."""
    if t1 is None or t2 is None:
        return {}
    ang_tol = np.radians(ang_tol_deg)
    tol = dict(dist_tol=dist_tol, dist_rel=dist_rel, ang_tol=ang_tol)

    roots = _root_candidates(t1, t2, dist_tol, dist_rel, ang_tol, top=root_candidates)
    best = {}
    for ri, rj in roots:
        matched = _grow(ri, rj, t1, t2, tol)
        if len(matched) > len(best):
            best = matched
    return best


def _grow(root_i, root_j, t1, t2, tol):
    """Breadth-first expansion from one confirmed root pair."""
    matched = {root_i: root_j}
    used2 = {root_j}
    queue = [root_i]
    adj1, adj2 = t1["adjacency"], t2["adjacency"]

    while queue:
        a = queue.pop()
        a2 = matched[a]
        edges_a = adj1[a]
        edges_a2 = adj2[a2]

        candidates = []
        for b, e_ab in edges_a.items():
            if b in matched:
                continue
            for b2, e_a2b2 in edges_a2.items():
                if b2 in used2:
                    continue
                if _edge_compatible(e_ab, e_a2b2, **tol):
                    candidates.append((abs(e_ab[0] - e_a2b2[0]), b, b2))

        candidates.sort(key=lambda c: c[0])
        for _score, b, b2 in candidates:
            if b in matched or b2 in used2:
                continue                       # a better candidate may have claimed it
            matched[b] = b2
            used2.add(b2)
            queue.append(b)

    return matched


# ==========================================================================
# Multi-press fusion ("move your finger around", as phone sensors enroll)
# ==========================================================================
# The R503 sees only ~8 mm of finger per press, ~10-20 minutiae. A live press
# can land anywhere on the finger, so it often shares too few minutiae with any
# single enrolled press. Enrollment therefore asks for presses at different
# positions and stitches them into one template covering the whole finger:
# each press is aligned onto the template through the minutiae it shares with
# it (rotation + shift), then its new minutiae are added.

# Shared minutiae needed to place a press onto the template. Fewer than this
# and the alignment could be a coincidence - the press stays a separate sample.
FUSE_MIN_PAIRS = int(os.environ.get("FUSE_MIN_PAIRS", "5"))

# Max RMS alignment error (px) for the shared minutiae; above it the press is
# not stitched (one ridge period is ~8 px at 508 dpi).
FUSE_MAX_RMS = float(os.environ.get("FUSE_MAX_RMS", "6.0"))

# A stitched minutia this close to an existing one, pointing the same way, is
# the same feature seen again: it is merged, not added twice.
FUSE_MERGE_DIST = float(os.environ.get("FUSE_MERGE_DIST", "8.0"))
FUSE_MERGE_ANGLE_DEG = float(os.environ.get("FUSE_MERGE_ANGLE_DEG", "30.0"))


def _rigid_fit(src: np.ndarray, dst: np.ndarray):
    """Least-squares rotation + shift taking src points onto dst (2-D Kabsch)."""
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    h = (src - cs).T @ (dst - cd)
    angle = np.arctan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shift = cd - rot @ cs
    return angle, rot, shift


def align_minutiae(new, base, min_pairs: int = FUSE_MIN_PAIRS, max_rms: float = FUSE_MAX_RMS):
    """Rotation/shift placing `new` minutiae onto `base`, or None if they don't overlap enough.

    Returns (angle, rot, shift, pairs_used, rms). Pairs that disagree with the
    fitted alignment are dropped and the fit repeated, so one wrong
    correspondence from the matcher cannot skew the whole placement.
    """
    pairs = match_pairs(build_template(new), build_template(base))
    if len(pairs) < min_pairs:
        return None
    src = np.array([new[i][:2] for i in pairs], float)
    dst = np.array([base[j][:2] for j in pairs.values()], float)

    keep = np.ones(len(src), bool)
    for _ in range(3):
        if keep.sum() < min_pairs:
            return None
        angle, rot, shift = _rigid_fit(src[keep], dst[keep])
        err = np.hypot(*((src @ rot.T + shift) - dst).T)
        new_keep = err <= max(2.0 * np.median(err[keep]), max_rms)
        if (new_keep == keep).all():
            break
        keep = new_keep
    if keep.sum() < min_pairs:
        return None
    rms = float(np.sqrt(np.mean(err[keep] ** 2)))
    if rms > max_rms:
        return None
    return angle, rot, shift, int(keep.sum()), rms


def fuse_minutiae(base, new, merge_dist: float = FUSE_MERGE_DIST,
                  merge_angle_deg: float = FUSE_MERGE_ANGLE_DEG):
    """Stitch `new` onto `base`. Returns (fused minutiae, info) or (None, None) if it can't be placed.

    info = {"pairs", "rms", "added"}: shared minutiae used for the alignment,
    its error in px, and how many minutiae the press contributed that the
    template did not have yet (0 = this part of the finger was already covered).
    """
    placed = align_minutiae(new, base)
    if placed is None:
        return None, None
    angle, rot, shift, pairs, rms = placed

    merge_ang = np.radians(merge_angle_deg)
    fused = list(base)
    base_xy = np.array([m[:2] for m in base], float)
    base_th = np.array([m[2] for m in base], float)
    added = 0
    for x, y, th, kind, q in new:
        nx, ny = rot @ np.array([x, y], float) + shift
        nth = float(np.mod(th + angle, 2 * np.pi))
        d = np.hypot(base_xy[:, 0] - nx, base_xy[:, 1] - ny)
        same = (d <= merge_dist) & (np.abs(_wrap(base_th - nth)) <= merge_ang)
        if same.any():
            continue                     # already in the template
        fused.append((float(nx), float(ny), nth, kind, q))
        added += 1

    # Keep coordinates non-negative: matching is shift-invariant, but .xyt
    # consumers and the LCD preview expect image-like coordinates.
    min_x = min(m[0] for m in fused)
    min_y = min(m[1] for m in fused)
    if min_x < 0 or min_y < 0:
        dx, dy = max(0.0, -min_x), max(0.0, -min_y)
        fused = [(mx + dx, my + dy, mt, mk, mq) for mx, my, mt, mk, mq in fused]
    return fused, {"pairs": pairs, "rms": rms, "added": added}


def fuse_presses(press_minutiae: Sequence[Sequence], min_presses: int = 2):
    """Stitch several presses of one finger into one whole-finger template.

    Starts from the press with the most minutiae and keeps adding whichever
    remaining press aligns onto the growing template with the most shared
    minutiae. Returns (fused minutiae, indices of the presses used); presses
    that never overlap enough are left out (they stay separate samples).
    Returns (None, []) when fewer than `min_presses` could be stitched.
    """
    presses = [list(p) for p in press_minutiae if p]
    if len(presses) < min_presses:
        return None, []
    start = max(range(len(presses)), key=lambda i: len(presses[i]))
    fused, used = presses[start], [start]
    remaining = [i for i in range(len(presses)) if i != start]
    while remaining:
        best = None
        for i in remaining:
            placed = align_minutiae(presses[i], fused)
            if placed is not None and (best is None or placed[3] > best[1]):
                best = (i, placed[3])
        if best is None:
            break
        fused, _info = fuse_minutiae(fused, presses[best[0]])
        used.append(best[0])
        remaining.remove(best[0])
    if len(used) < min_presses:
        return None, []
    return fused, used


# ==========================================================================
# Engine / gallery / identification  (mirrors fingerprint_nbis)
# ==========================================================================

class LocalEngine:
    """Same surface as fingerprint_nbis.NbisEngine, no native binaries."""

    def __init__(self, threshold: int = LOCAL_MATCH_THRESHOLD, verbose: bool = True):
        self.threshold = threshold
        self.available = True
        self.verbose = verbose
        self._lock = threading.RLock()
        if verbose:
            print(f"[OK] Local fingerprint engine (pure Python), threshold {threshold}")

    def extract_template(self, image: np.ndarray) -> Optional[Dict]:
        try:
            minutiae, _skel = extract_minutiae(image)
        except Exception as exc:  # noqa: BLE001 - a bad scan must not kill the loop
            print(f"[ERROR] minutiae extraction failed: {exc}")
            return None
        if len(minutiae) < 6:
            return None
        return {
            "xyt": minutiae_to_xyt(minutiae),
            "minutiae_count": len(minutiae),
            "quality": float(np.mean([m[4] for m in minutiae])),
        }

    def match(self, xyt1: str, xyt2: str) -> int:
        return match_templates(build_template(parse_xyt(xyt1)),
                               build_template(parse_xyt(xyt2)))


class GalleryCache:
    """Holds enrolled pair tables in memory, rebuilt when enrollment changes."""

    def __init__(self, directory: str = ""):
        self.entries: List[Tuple[int, str, Dict]] = []
        self._signature = None
        self._lock = threading.RLock()

    @staticmethod
    def _samples_of(profile: Dict) -> List[Dict]:
        if "fingers" in profile:
            out = []
            for finger in profile["fingers"]:
                out.extend(finger.get("samples", []))
            return out
        return profile.get("fingerprints", [])

    @staticmethod
    def _signature_of(profiles) -> str:
        import hashlib
        h = hashlib.md5()
        for p in profiles:
            usable = sum(1 for s in GalleryCache._samples_of(p)
                         if isinstance(s, dict) and s.get("xyt"))
            h.update(f"{p.get('client_id')}:{usable}|".encode())
        return h.hexdigest()

    def invalidate(self) -> None:
        with self._lock:
            self._signature = None

    def build(self, profiles, force: bool = False):
        with self._lock:
            sig = self._signature_of(profiles)
            if not force and sig == self._signature and self.entries:
                return self.entries
            entries, skipped = [], 0
            for p in profiles:
                cid = p.get("client_id")
                name = p.get("name", f"Client {cid}")
                for s in self._samples_of(p):
                    if not isinstance(s, dict) or not s.get("xyt"):
                        skipped += 1
                        continue
                    t = build_template(parse_xyt(s["xyt"]))
                    if t is not None:
                        entries.append((cid, name, t))
            self.entries = entries
            self._signature = sig
            clients = len({e[0] for e in entries})
            print(f"[OK] Local gallery: {len(entries)} templates from {clients} clients"
                  + (f" ({skipped} samples have no template yet)" if skipped else ""))
            return entries


def identify(engine: LocalEngine, image_or_xyt, profiles, gallery: GalleryCache,
             threshold: Optional[int] = None,
             margin: float = LOCAL_MATCH_MARGIN):
    """1:N identification. Returns (client_id, name, score, details) or None."""
    threshold = engine.threshold if threshold is None else threshold

    if isinstance(image_or_xyt, str):
        query = build_template(parse_xyt(image_or_xyt))
        count = len(parse_xyt(image_or_xyt))
    else:
        tpl = engine.extract_template(image_or_xyt)
        if tpl is None:
            print("[ERROR] no usable minutiae in the query image")
            return None
        query = build_template(parse_xyt(tpl["xyt"]))
        count = tpl["minutiae_count"]
    if query is None:
        return None

    entries = gallery.build(profiles)
    if not entries:
        print("[WARNING] local gallery is empty - no enrolled templates")
        return None

    started = time.time()
    per_client: Dict[int, Tuple[str, int]] = {}
    for cid, name, tmpl in entries:
        n = match_templates(query, tmpl)
        cur = per_client.get(cid)
        if cur is None or n > cur[1]:
            per_client[cid] = (name, n)
    elapsed = time.time() - started

    ranking = sorted(((c, nm, sc) for c, (nm, sc) in per_client.items()),
                     key=lambda r: r[2], reverse=True)
    details = {"minutiae": count, "compared": len(entries),
               "elapsed": elapsed, "ranking": ranking[:5]}
    if not ranking:
        return None

    cid, name, score = ranking[0]
    runner = ranking[1][2] if len(ranking) > 1 else 0
    if score < threshold:
        # The single most useful line for diagnosing "nothing ever matches":
        # without it, every rejection looks identical whether the real match
        # missed by 1 or by 10.
        print(f"[INFO] no match: best was {name} at {score} "
              f"(need {threshold}, {count} minutiae in the query, "
              f"{len(entries)} templates compared in {elapsed*1000:.0f} ms)")
        return None
    if margin > 1.0 and runner > 0 and score < runner * margin:
        print(f"[WARNING] ambiguous match rejected: {name} {score} vs "
              f"{ranking[1][1]} {runner} (margin {margin})")
        return None
    return (cid, name, score, details)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _selftest() -> int:
    print("=" * 60)
    print("Local fingerprint engine self test")
    print("=" * 60)
    engine = LocalEngine()
    rng = np.random.default_rng(3)
    noise = (rng.random((160, 160)) * 255).astype(np.uint8)
    print(f"  noise image -> {engine.extract_template(noise) and 'template' or 'no template (expected)'}")

    for folder in ("test/r503_captures", "fingerprint_storage/images"):
        files = sorted(Path(folder).glob("*.png"))[:3]
        if not files:
            continue
        print(f"  {folder}:")
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            t = engine.extract_template(img)
            kind = "pre-thinned" if is_pre_thinned(img) else "grayscale"
            print(f"    {f.name}: {kind}, "
                  + (f"{t['minutiae_count']} minutiae, quality {t['quality']:.0f}"
                     if t else "no template"))
    print("\nRESULT: engine usable. Benchmark with test/test_local_engine.py")
    return 0


def _bench(folder: str) -> int:
    files = sorted(Path(folder).glob("*.png"))
    if not files:
        print(f"no images in {folder}")
        return 1
    engine = LocalEngine()
    t0 = time.time()
    tpls = []
    for f in files[:40]:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        t = engine.extract_template(img)
        tpls.append(build_template(parse_xyt(t["xyt"])) if t else None)
    ext = (time.time() - t0) / max(len(tpls), 1)
    ok = [t for t in tpls if t is not None]
    t0 = time.time()
    n = 0
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            match_templates(ok[i], ok[j])
            n += 1
    print(f"extraction {ext*1000:.0f} ms/image, matching "
          f"{(time.time()-t0)/max(n,1)*1000:.2f} ms/pair over {n} pairs")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    if "--bench" in sys.argv:
        i = sys.argv.index("--bench")
        raise SystemExit(_bench(sys.argv[i + 1] if len(sys.argv) > i + 1 else "test/r503_captures"))
    print(__doc__)
