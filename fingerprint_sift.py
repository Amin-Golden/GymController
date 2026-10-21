#!/usr/bin/env python3
"""
Fingerprint engine for the R503's grayscale image: SIFT keypoints verified by
geometry, with a FAISS index so one press is identified among 1000+ members.

Why this engine (measured on the R503, test/r503_template_lab.py compare):
  18 fingers enrolled in one session, queried in the next with shifted,
  rotated, light and firm presses, plus fingers never enrolled:
    SIFT ratio score alone        right finger first 72/72, but the impostor
                                  tail is heavy - best impostor 0.079 against
                                  a genuine 5th percentile of 0.120
    + geometric verification      matched keypoints must agree on ONE rotation
      (this engine)               and shift of the finger: impostors never got
                                  more than 5 such inliers in 2898 finger
                                  comparisons; genuine presses median 170,
                                  5th percentile 45
  A stranger at the door is compared with every enrolled finger, so the score
  that matters is how far the impostor tail stays below the threshold. Random
  matches agreeing on one rigid motion become rarer very fast as their number
  grows, which is what makes 1:N over thousands of fingers workable.

  The sensor's own minutiae template (r503_template.py) identified 65/72 on the
  same test; the minutiae engines that read the sensor's skeleton image did worse.

Pipeline
  extract()      grayscale press -> keypoints (x, y, angle) + SIFT descriptors
                 (uint8 - SIFT values are whole numbers 0..255), strongest first
  inliers()      two presses -> matched keypoints consistent with one rotation
                 + shift (rigid RANSAC; each match's own SIFT orientation change
                 must agree with that rotation)
  SiftGallery    1:N. Every press's strongest INDEX_CAP descriptors sit in a
                 FAISS index; a query's descriptors vote for fingers, the
                 SHORTLIST best are verified with inliers().

Storage: pack()/unpack(), one bytea per press (fingerprint_features.sift_gray,
see migrate_fingerprint_sift_gray.sql), ~75 KB per press.

The image must be the sensor's GRAYSCALE capture (finger_UDP.ino uploads it
before Img2Tz). A two-level skeleton - what older firmware sent - is refused.

    py -3 fingerprint_sift.py --selftest
    py -3 fingerprint_sift.py --bench --images test/template_lab/session_a --members 1000
    py -3 fingerprint_sift.py --report --db-host 192.168.1.3
"""

import math
import os
import pickle
import struct
import sys
import threading
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:  # noqa: BLE001
    faiss = None
    FAISS_AVAILABLE = False

# --------------------------------------------------------------------------
# Tunables (environment overridable)
# --------------------------------------------------------------------------

# Accept when the best finger has at least this many geometrically consistent
# matches. Measured: impostors max 5 (2898 finger comparisons), genuine next-
# session presses 5th percentile 45; one weak press of 72 scored 6. Every value
# from 10 to 25 rejected 1/72 genuine and accepted 0 impostors - 20 keeps 4x
# headroom over the worst impostor for a gallery of thousands of fingers.
SIFT_MATCH_INLIERS = int(os.environ.get("SIFT_MATCH_INLIERS", "20"))

# Enrollment: a press with fewer keypoints is rejected (measured presses had
# 287..948, median 566).
SIFT_MIN_KEYPOINTS = int(os.environ.get("SIFT_MIN_KEYPOINTS", "150"))

RATIO = 0.75                    # Lowe's ratio test for a descriptor match
RANSAC_TOL_PX = float(os.environ.get("SIFT_RANSAC_TOL_PX", "8"))   # skin stretches between presses
RANSAC_ANGLE_TOL = math.radians(30)
RANSAC_ITERS = 300

INDEX_CAP = 200                 # strongest descriptors per press that go into the index
QUERY_CAP = int(os.environ.get("SIFT_QUERY_CAP", "200"))   # strongest query descriptors looked up
INDEX_K = 8                     # neighbours fetched per query descriptor
VOTE_RATIO = 0.8                # a vote needs a clearly nearer neighbour than any other finger's
SHORTLIST = int(os.environ.get("SIFT_SHORTLIST", "5"))   # fingers verified per query

# Exact search up to this many indexed descriptors (~1500 presses, ~95 members
# at 2 fingers x 8 presses). Beyond it an approximate IVF index is trained once,
# in the background, while the exact one keeps serving.
FLAT_MAX_VECTORS = int(os.environ.get("SIFT_FLAT_MAX_VECTORS", "300000"))
IVF_SPEC = "IVF4096_HNSW32,SQ8"
# Lists searched per descriptor. Measured at 1000 members (72 genuine + 8
# stranger presses): 16, 8 and 4 all shortlisted 72/72 and let no stranger in;
# 8 took 186 ms per press on the dev PC against 286 ms for 16. Looking up only
# 100 query descriptors lost a finger from the shortlist, so QUERY_CAP stays 200.
IVF_NPROBE = int(os.environ.get("SIFT_NPROBE", "8"))

INDEX_DIR = os.environ.get("SIFT_INDEX_DIR",
                           str(Path(__file__).resolve().parent / "fingerprint_storage" / "sift_index"))

MIN_GRAY_LEVELS = 8             # a skeleton has 2; a real R503 capture has 16
SLOT_IDS = 1024                 # index id = slot * SLOT_IDS + keypoint rank
MAGIC = b"SG1\0"
MANIFEST_VERSION = 1

# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

_local = threading.local()      # OpenCV objects are not shared between threads


def _tools():
    if not hasattr(_local, "sift"):
        _local.sift = cv2.SIFT_create()
        _local.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        _local.bf = cv2.BFMatcher(cv2.NORM_L2)
    return _local


class Features:
    """Keypoints of one press: pts (n,2) float32, ang (n,) radians, desc (n,128) uint8."""
    __slots__ = ("pts", "ang", "desc")

    def __init__(self, pts, ang, desc):
        self.pts, self.ang, self.desc = pts, ang, desc

    @property
    def n(self):
        return len(self.desc)


def is_grayscale(image: np.ndarray) -> bool:
    """False for the two-level skeleton older firmware sent after Img2Tz."""
    return len(np.unique(image)) >= MIN_GRAY_LEVELS


def extract(image: np.ndarray) -> Optional[Features]:
    """Grayscale press -> Features (strongest keypoints first), or None."""
    if image is None:
        return None
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.ascontiguousarray(image, dtype=np.uint8)
    if not is_grayscale(image):
        return None
    t = _tools()
    im = t.clahe.apply(image)
    im = cv2.fastNlMeansDenoising(im, None, h=10)
    im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX)
    kp, desc = t.sift.detectAndCompute(im, None)
    if desc is None or not kp:
        return None
    order = np.argsort([-k.response for k in kp], kind="stable")
    pts = np.float32([kp[i].pt for i in order])
    ang = np.float32([math.radians(kp[i].angle) for i in order])
    return Features(pts, ang, np.clip(np.rint(desc[order]), 0, 255).astype(np.uint8))


def pack(f: Features) -> bytes:
    """Features -> bytes for the database: magic, count, (x, y, angle) float32, descriptors uint8."""
    geo = np.column_stack([f.pts, f.ang]).astype("<f4")
    return MAGIC + struct.pack("<I", f.n) + geo.tobytes() + np.ascontiguousarray(f.desc, np.uint8).tobytes()


def unpack(blob) -> Features:
    """bytes from pack() -> Features (zero-copy views into blob)."""
    blob = bytes(blob)
    if blob[:4] != MAGIC or len(blob) < 8:
        raise ValueError("not a sift_gray record")
    n = struct.unpack_from("<I", blob, 4)[0]
    if len(blob) != 8 + n * 12 + n * 128:
        raise ValueError(f"sift_gray record of {len(blob)} bytes does not hold {n} keypoints")
    geo = np.frombuffer(blob, "<f4", count=n * 3, offset=8).reshape(n, 3)
    desc = np.frombuffer(blob, np.uint8, count=n * 128, offset=8 + n * 12).reshape(n, 128)
    return Features(geo[:, :2], geo[:, 2], desc)


# --------------------------------------------------------------------------
# Verification: matches that agree on one rotation + shift
# --------------------------------------------------------------------------

def _rigid_inliers(src, dst, dang, rng):
    n = len(src)
    if n < 3:
        return 0
    i = rng.integers(0, n, RANSAC_ITERS)
    j = rng.integers(0, n, RANSAC_ITERS)
    vs, vd = src[j] - src[i], dst[j] - dst[i]
    ls, ld = np.hypot(vs[:, 0], vs[:, 1]), np.hypot(vd[:, 0], vd[:, 1])
    ok = (ls > 10) & (np.abs(ls - ld) < RANSAC_TOL_PX)        # a rigid motion keeps lengths
    if not ok.any():
        return 0
    i, vs, vd = i[ok], vs[ok], vd[ok]
    th = np.arctan2(vd[:, 1], vd[:, 0]) - np.arctan2(vs[:, 1], vs[:, 0])
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    tx = dst[i, 0][:, None] - (c * src[i, 0][:, None] - s * src[i, 1][:, None])
    ty = dst[i, 1][:, None] - (s * src[i, 0][:, None] + c * src[i, 1][:, None])
    px = c * src[None, :, 0] - s * src[None, :, 1] + tx
    py = s * src[None, :, 0] + c * src[None, :, 1] + ty
    close = np.hypot(px - dst[None, :, 0], py - dst[None, :, 1]) <= RANSAC_TOL_PX
    turned = np.abs((dang[None, :] - th[:, None] + np.pi) % (2 * np.pi) - np.pi) <= RANSAC_ANGLE_TOL
    return int((close & turned).sum(axis=1).max())


def inliers(a: Features, b: Features, a_f32: Optional[np.ndarray] = None) -> int:
    """Geometrically consistent keypoint matches between two presses (the match score).

    a_f32: a's descriptors as float32, when the caller compares one query with many presses.
    """
    if a.n < 2 or b.n < 2:
        return 0
    qa = a_f32 if a_f32 is not None else a.desc.astype(np.float32)
    good = [p[0] for p in _tools().bf.knnMatch(qa, b.desc.astype(np.float32), k=2)
            if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    if len(good) < 3:
        return 0
    qi = np.fromiter((m.queryIdx for m in good), np.int64, len(good))
    ti = np.fromiter((m.trainIdx for m in good), np.int64, len(good))
    return _rigid_inliers(a.pts[qi], b.pts[ti], b.ang[ti] - a.ang[qi], np.random.default_rng(0))


def confidence(n_inliers: int) -> float:
    """0..1 for callers that log a percentage: inliers / 100, capped."""
    return min(1.0, n_inliers / 100.0)


# --------------------------------------------------------------------------
# Gallery: 1:N with a FAISS shortlist
# --------------------------------------------------------------------------

def sample_key(client_id, finger_index, sample_number, blob) -> tuple:
    """Identity of one stored press; changes whenever the press is re-enrolled."""
    blob = bytes(blob) if not isinstance(blob, bytes) else blob
    return (client_id, finger_index, sample_number, len(blob), zlib.crc32(blob[:512] + blob[-512:]))


class SiftGallery:
    """Enrolled presses + their descriptor index. Thread safe.

    sync(profiles) brings it in line with the profiles the gym system holds
    (the BiometricCache structure: [{client_id, name, fingers: [{finger_index,
    samples: [{sample_number, sift_gray}]}]}]), adding and removing only what
    changed. With index_dir set, the index is saved there so a restart does
    not rebuild it.
    """

    def __init__(self, index_dir: Optional[str] = INDEX_DIR, verbose: bool = True):
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss is not installed (pip install faiss-cpu)")
        self.index_dir = Path(index_dir) if index_dir else None
        self.verbose = verbose
        self.lock = threading.RLock()
        self.slots: Dict[int, dict] = {}          # slot -> {key, client_id, finger, name, f}
        self.by_key: Dict[tuple, int] = {}
        self.by_finger: Dict[tuple, List[int]] = {}
        self.next_slot = 0
        self.index = None
        self.kind = None                          # "flat" | "ivf"
        self.vectors = 0
        self._upgrading = False
        self._loaded = False

    # ---- persistence -----------------------------------------------------

    def _paths(self):
        return self.index_dir / "index.faiss", self.index_dir / "manifest.pkl"

    def _new_flat(self):
        return faiss.IndexIDMap2(faiss.IndexFlatL2(128))

    def _load(self):
        """Restore a saved index; its slots get their features on the next sync."""
        self._loaded = True
        self.index, self.kind = self._new_flat(), "flat"
        if not self.index_dir:
            return
        index_path, manifest_path = self._paths()
        if not (index_path.exists() and manifest_path.exists()):
            return
        try:
            with open(manifest_path, "rb") as fh:
                m = pickle.load(fh)
            if m.get("version") != MANIFEST_VERSION or m.get("cap") != INDEX_CAP:
                raise ValueError("saved index was built with other settings")
            index = faiss.read_index(str(index_path))
            if m["kind"] == "ivf":
                faiss.extract_index_ivf(index).nprobe = IVF_NPROBE
            self.index, self.kind, self.vectors = index, m["kind"], m["vectors"]
            self.next_slot = m["next_slot"]
            for slot, (key, n_vec) in m["slots"].items():
                self.slots[slot] = {"key": key, "client_id": key[0], "finger": key[1],
                                    "name": "", "f": None, "nvec": n_vec}
                self.by_key[key] = slot
            self._log(f"[SIFT] loaded saved index: {len(self.slots)} presses, {self.vectors} descriptors ({self.kind})")
        except Exception as e:  # noqa: BLE001 - a damaged file just means a rebuild
            self._log(f"[SIFT] saved index unusable ({e}) - rebuilding")
            self.slots, self.by_key, self.next_slot, self.vectors = {}, {}, 0, 0
            self.index, self.kind = self._new_flat(), "flat"

    def _save(self):
        if not self.index_dir:
            return
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            index_path, manifest_path = self._paths()
            manifest = {"version": MANIFEST_VERSION, "cap": INDEX_CAP, "kind": self.kind,
                        "vectors": self.vectors, "next_slot": self.next_slot,
                        "slots": {s: (v["key"], v["nvec"]) for s, v in self.slots.items()}}
            faiss.write_index(self.index, str(index_path) + ".tmp")
            with open(str(manifest_path) + ".tmp", "wb") as fh:
                pickle.dump(manifest, fh)
            os.replace(str(index_path) + ".tmp", index_path)
            os.replace(str(manifest_path) + ".tmp", manifest_path)
        except Exception as e:  # noqa: BLE001 - matching still works, only the restart is slower
            self._log(f"[SIFT] could not save the index: {e}")

    def _log(self, text):
        if self.verbose:
            print(text)

    # ---- maintenance -----------------------------------------------------

    @staticmethod
    def _vectors_of(f: Features) -> np.ndarray:
        return np.ascontiguousarray(f.desc[:INDEX_CAP], dtype=np.float32)

    def _add(self, slot, f: Features):
        x = self._vectors_of(f)
        ids = slot * SLOT_IDS + np.arange(len(x), dtype=np.int64)
        self.index.add_with_ids(x, ids)
        self.vectors += len(x)
        return len(x)

    def _remove(self, slot):
        info = self.slots.pop(slot)
        self.by_key.pop(info["key"], None)
        fk = (info["client_id"], info["finger"])
        if fk in self.by_finger:
            self.by_finger[fk] = [s for s in self.by_finger[fk] if s != slot]
            if not self.by_finger[fk]:
                del self.by_finger[fk]
        self.index.remove_ids(faiss.IDSelectorRange(slot * SLOT_IDS, (slot + 1) * SLOT_IDS))
        self.vectors -= info.get("nvec", 0)

    def sync(self, profiles) -> bool:
        """Match the gallery to profiles; True when anything changed."""
        wanted = {}
        for p in profiles or []:
            for finger in p.get("fingers", []):
                for s in finger.get("samples", []):
                    blob = s.get("sift_gray")
                    if not blob:
                        continue
                    key = sample_key(p["client_id"], finger["finger_index"], s.get("sample_number"), blob)
                    wanted[key] = (p["client_id"], finger["finger_index"], p.get("name", ""), blob)

        with self.lock:
            if not self._loaded:
                self._load()
            changed = False
            for key in [k for k in self.by_key if k not in wanted]:
                self._remove(self.by_key[key])
                changed = True
            added = 0
            for key, (cid, fidx, name, blob) in wanted.items():
                slot = self.by_key.get(key)
                if slot is not None:                  # known press: refresh name, attach features after a load
                    info = self.slots[slot]
                    info["name"] = name
                    if info["f"] is None:
                        self.by_finger.setdefault((cid, fidx), []).append(slot)
                    if info.get("blob") is not blob:
                        # Features are views into the blob: follow the caller's current copy,
                        # or a profile list the caller has replaced would stay in memory
                        info["f"], info["blob"] = unpack(blob), blob
                    continue
                try:
                    f = unpack(blob)
                except ValueError as e:
                    self._log(f"[SIFT] client {cid} finger {fidx}: unreadable press ({e}) - skipped")
                    continue
                slot = self.next_slot
                self.next_slot += 1
                nvec = self._add(slot, f)
                self.slots[slot] = {"key": key, "client_id": cid, "finger": fidx, "name": name, "f": f,
                                    "blob": blob, "nvec": nvec}
                self.by_key[key] = slot
                self.by_finger.setdefault((cid, fidx), []).append(slot)
                added += 1
                changed = True
            if changed:
                if added > 50:
                    self._log(f"[SIFT] gallery: {len(self.by_finger)} fingers, {len(self.slots)} presses, "
                              f"{self.vectors} indexed descriptors ({self.kind})")
                self._save()
            self._maybe_upgrade()        # also after a restart that stopped before the upgrade finished
            return changed

    def _maybe_upgrade(self):
        """Past FLAT_MAX_VECTORS, train the approximate index in the background."""
        if self.kind != "flat" or self.vectors <= FLAT_MAX_VECTORS or self._upgrading:
            return
        self._upgrading = True
        snapshot = {slot: info["f"] for slot, info in self.slots.items() if info["f"] is not None}
        self._log(f"[SIFT] {self.vectors} descriptors - building the approximate index in the background "
                  f"(identification keeps using the exact one meanwhile)")
        threading.Thread(target=self._build_ivf, args=(snapshot,), daemon=True).start()

    def _build_ivf(self, snapshot):
        try:
            t0 = time.time()
            x = np.concatenate([self._vectors_of(f) for f in snapshot.values()])
            rng = np.random.default_rng(0)
            index = faiss.index_factory(128, IVF_SPEC)
            index.train(x[rng.choice(len(x), min(len(x), 250_000), replace=False)])
            faiss.extract_index_ivf(index).nprobe = IVF_NPROBE
            for slot, f in snapshot.items():
                v = self._vectors_of(f)
                index.add_with_ids(v, slot * SLOT_IDS + np.arange(len(v), dtype=np.int64))
            with self.lock:
                # presses enrolled or removed while this was building
                for slot in [s for s in snapshot if s not in self.slots]:
                    index.remove_ids(faiss.IDSelectorRange(slot * SLOT_IDS, (slot + 1) * SLOT_IDS))
                for slot, info in self.slots.items():
                    if slot not in snapshot and info["f"] is not None:
                        v = self._vectors_of(info["f"])
                        index.add_with_ids(v, slot * SLOT_IDS + np.arange(len(v), dtype=np.int64))
                self.index, self.kind = index, "ivf"
                self._save()
            self._log(f"[SIFT] approximate index ready in {time.time() - t0:.0f} s")
        except Exception as e:  # noqa: BLE001 - the exact index keeps working
            self._log(f"[SIFT] approximate index build failed: {e}")
        finally:
            self._upgrading = False

    # ---- search ----------------------------------------------------------

    def stats(self) -> dict:
        with self.lock:
            return {"members": len({c for c, _ in self.by_finger}), "fingers": len(self.by_finger),
                    "presses": len(self.slots), "descriptors": self.vectors, "index": self.kind}

    def shortlist(self, query: Features, exclude_client=None, k: int = SHORTLIST) -> List[tuple]:
        """The k fingers whose descriptors the query's descriptors vote for most."""
        q = np.ascontiguousarray(query.desc[:QUERY_CAP], dtype=np.float32)
        if not len(q):
            return []
        with self.lock:
            if self.index is None or self.index.ntotal == 0:
                return []
            D, I = self.index.search(q, INDEX_K)
            owner = {}
            for slot_id in np.unique(I[I >= 0] // SLOT_IDS):
                info = self.slots.get(int(slot_id))
                if info is not None and info["client_id"] != exclude_client:
                    owner[int(slot_id)] = (info["client_id"], info["finger"])
        votes: Dict[tuple, int] = {}
        for d_row, i_row in zip(D, I):
            fingers = [(owner.get(int(i) // SLOT_IDS), d) for i, d in zip(i_row, d_row) if i >= 0]
            fingers = [(f, d) for f, d in fingers if f is not None]
            if not fingers:
                continue
            best, d1 = fingers[0]
            other = next((d for f, d in fingers if f != best), None)
            if other is None or d1 < VOTE_RATIO ** 2 * other:
                votes[best] = votes.get(best, 0) + 1
        return sorted(votes, key=votes.get, reverse=True)[:k]

    def identify(self, query: Features, exclude_client=None, threshold: int = SIFT_MATCH_INLIERS):
        """(best or None, ranking). Each entry: (inliers, client_id, finger_index, name)."""
        candidates = self.shortlist(query, exclude_client=exclude_client)
        q32 = query.desc.astype(np.float32)
        ranking = []
        for fk in candidates:
            with self.lock:
                presses = [(self.slots[s]["f"], self.slots[s]["name"])
                           for s in self.by_finger.get(fk, [])
                           if s in self.slots and self.slots[s]["f"] is not None]
            if not presses:
                continue
            best = max(inliers(query, f, q32) for f, _ in presses)
            ranking.append((best, fk[0], fk[1], presses[0][1]))
        ranking.sort(key=lambda r: r[0], reverse=True)
        best = ranking[0] if ranking and ranking[0][0] >= threshold else None
        return best, ranking

    def closest_other_members(self, per_finger: int = 1, threshold: int = SIFT_MATCH_INLIERS):
        """Self-measurement: each finger's press(es) against every OTHER member.

        Returns [(inliers, client_id, finger, other_client, other_finger)] sorted
        worst first - how close members come to being mistaken for each other."""
        with self.lock:
            fingers = {fk: list(slots[:per_finger]) for fk, slots in self.by_finger.items()}
        rows = []
        for (cid, fidx), slots in fingers.items():
            for slot in slots:
                f = self.slots.get(slot, {}).get("f")
                if f is None:
                    continue
                _, ranking = self.identify(f, exclude_client=cid, threshold=threshold)
                if ranking:
                    rows.append((ranking[0][0], cid, fidx, ranking[0][1], ranking[0][2]))
                else:
                    rows.append((0, cid, fidx, None, None))
        rows.sort(key=lambda r: r[0], reverse=True)
        return rows


# --------------------------------------------------------------------------
# Command line: self test, Orange Pi benchmark, gallery report
# --------------------------------------------------------------------------

def _load_folder(folder):
    """{finger label: [Features]} from <label>_s<n>_raw.png (as r503_template_lab captures them)."""
    out = {}
    for p in sorted(Path(folder).glob("*_raw.png")):
        f = extract(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
        if f is not None:
            out.setdefault(p.name.rsplit("_s", 1)[0], []).append(f)
    return out


def _mirror(desc):
    """Mirror-image SIFT descriptors: realistic values that no real finger in the gallery produces."""
    g = desc.reshape(-1, 4, 4, 8)[:, :, ::-1, :][..., (4 - np.arange(8)) % 8]
    return np.concatenate([np.rot90(g, r, axes=(1, 2))[..., (np.arange(8) - 2 * r) % 8].reshape(len(desc), 128)
                           for r in range(4)])


def _selftest() -> int:
    print(f"OpenCV {cv2.__version__}, SIFT: ", end="")
    try:
        cv2.SIFT_create()
        print("ok")
    except Exception as e:  # noqa: BLE001
        print(f"MISSING ({e}) - need opencv-python >= 4.4")
        return 1
    print(f"faiss: {'ok ' + faiss.__version__ if FAISS_AVAILABLE else 'MISSING - pip install faiss-cpu'}")
    rng = np.random.default_rng(0)
    img = (cv2.GaussianBlur(rng.integers(0, 16, (160, 160)).astype(np.uint8) * 17, (3, 3), 0))
    f = extract(img)
    if f is None:
        print("extraction failed")
        return 1
    g = unpack(pack(f))
    same = np.array_equal(g.desc, f.desc) and np.allclose(g.pts, f.pts)
    print(f"extract/pack/unpack: {f.n} keypoints, round trip {'ok' if same else 'MISMATCH'}, "
          f"{len(pack(f)) / 1024:.0f} KB")
    print(f"self match: {inliers(f, g)} inliers (threshold {SIFT_MATCH_INLIERS})")
    print(f"skeleton refused: {extract((img > 128).astype(np.uint8) * 255) is None}")
    return 0 if same and FAISS_AVAILABLE else 1


def _bench(images, members, presses, queries_per_finger) -> int:
    """Gallery of real fingers (from images) + synthetic members up to `members`."""
    real = _load_folder(images)
    if not real:
        print(f"no <label>_s<n>_raw.png images in {images}")
        return 1
    rng = np.random.default_rng(1)
    profiles, queries = [], []
    for cid, (label, feats) in enumerate(sorted(real.items()), start=1):
        enrolled, held_out = feats[:presses], feats[presses:presses + queries_per_finger] or feats[-1:]
        profiles.append({"client_id": cid, "name": label, "fingers": [
            {"finger_index": 1, "samples": [{"sample_number": i + 1, "sift_gray": pack(f)}
                                            for i, f in enumerate(enrolled)]}]})
        queries += [(cid, f) for f in held_out]
    base = np.concatenate([_mirror(f.desc) for fs in real.values() for f in fs])
    n_fake = max(0, members - len(profiles))
    t0 = time.time()
    for m in range(n_fake):
        fingers = []
        for fidx in (1, 2):
            samples = []
            for s in range(presses):
                n = int(rng.integers(350, 700))
                desc = np.clip(base[rng.integers(0, len(base), n)] + rng.normal(0, 8, (n, 128)), 0, 255).astype(np.uint8)
                f = Features(rng.uniform(0, 160, (n, 2)).astype(np.float32),
                             rng.uniform(-np.pi, np.pi, n).astype(np.float32), desc)
                samples.append({"sample_number": s + 1, "sift_gray": pack(f)})
            fingers.append({"finger_index": fidx, "samples": samples})
        profiles.append({"client_id": 100000 + m, "name": f"synthetic {m}", "fingers": fingers})
    print(f"gallery: {len(real)} real fingers + {n_fake} synthetic members x 2 fingers x {presses} presses "
          f"(made in {time.time() - t0:.0f} s)")

    gallery = SiftGallery(index_dir=None, verbose=True)
    t0 = time.time()
    gallery.sync(profiles)
    print(f"index built in {time.time() - t0:.1f} s: {gallery.stats()}")
    if gallery.kind == "flat" and gallery.vectors > FLAT_MAX_VECTORS:
        print("waiting for the approximate index...")
        while gallery._upgrading:
            time.sleep(1)
        print(f"now: {gallery.stats()}")
    mem_mb = sum(len(s["sift_gray"]) for p in profiles for fi in p["fingers"] for s in fi["samples"]) / 2 ** 20
    print(f"stored presses: {mem_mb:.0f} MB")

    times, hits, short = [], 0, 0
    for cid, f in queries:
        t0 = time.perf_counter()
        best, ranking = gallery.identify(f)
        times.append(time.perf_counter() - t0)
        short += any(r[1] == cid for r in ranking)
        hits += best is not None and best[1] == cid
    print(f"{len(queries)} held-out presses of real fingers: in shortlist {short}, identified {hits} "
          f"(threshold {SIFT_MATCH_INLIERS} inliers)")
    print(f"identification time: median {1000 * np.median(times):.0f} ms, max {1000 * max(times):.0f} ms")
    return 0


def _report(db_host, per_finger) -> int:
    from db_helper import DatabaseHelper
    db = DatabaseHelper(host=db_host, database=os.environ.get("GYM_DB_NAME", "gym-db"),
                        user=os.environ.get("GYM_DB_USER", "postgres"),
                        password=os.environ.get("GYM_DB_PASSWORD", "123456"), port=5432)
    profiles = db.get_all_fingerprint_features()
    gallery = SiftGallery(index_dir=None, verbose=False)
    gallery.sync(profiles)
    st = gallery.stats()
    legacy = sum(1 for p in profiles if not any(s.get("sift_gray") for fi in p.get("fingers", [])
                                                 for s in fi.get("samples", [])))
    print(f"enrolled (grayscale engine): {st['members']} members, {st['fingers']} fingers, {st['presses']} presses")
    if legacy:
        print(f"{legacy} members have only old-format fingerprints and must enroll again")
    rows = gallery.closest_other_members(per_finger=per_finger)
    if not rows:
        print("nothing to compare yet")
        return 0
    worst = np.array([r[0] for r in rows])
    print(f"\neach finger against every OTHER member ({len(rows)} fingers):")
    print(f"  closest other member: median {np.median(worst):.0f}, 99th pct {np.percentile(worst, 99):.0f}, "
          f"max {worst.max()} inliers  -  threshold {SIFT_MATCH_INLIERS}")
    for r in rows[:10]:
        flag = "  <-- AT/ABOVE THRESHOLD" if r[0] >= SIFT_MATCH_INLIERS else ""
        print(f"  client {r[1]} finger {r[2]}  ~  client {r[3]} finger {r[4]}: {r[0]} inliers{flag}")
    if worst.max() >= SIFT_MATCH_INLIERS:
        print("\nTwo different members reach the threshold: re-enroll those fingers, or raise SIFT_MATCH_INLIERS.")
    elif worst.max() * 2 > SIFT_MATCH_INLIERS:
        print("\nHeadroom is below 2x: watch this as members are added.")
    return 0


def main(argv) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", action="store_true", help="speed and memory at gym scale (run it on the OrangePi)")
    ap.add_argument("--images", default="test/template_lab/session_a",
                    help="bench: folder of <label>_s<n>_raw.png presses (real fingers)")
    ap.add_argument("--members", type=int, default=1000, help="bench: gallery size in members")
    ap.add_argument("--presses", type=int, default=4, help="bench: enrolled presses per finger")
    ap.add_argument("--queries", type=int, default=1, help="bench: held-out presses per real finger")
    ap.add_argument("--report", action="store_true", help="how close enrolled members come to each other")
    ap.add_argument("--db-host", help="report: database host")
    ap.add_argument("--per-finger", type=int, default=1, help="report: presses per finger to test")
    args = ap.parse_args(argv)
    if args.bench:
        return _bench(args.images, args.members, args.presses, args.queries)
    if args.report:
        if not args.db_host:
            print("--report needs --db-host")
            return 2
        return _report(args.db_host, args.per_finger)
    return _selftest()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
