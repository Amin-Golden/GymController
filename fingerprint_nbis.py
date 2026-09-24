#!/usr/bin/env python3
"""
NBIS fingerprint engine (MINDTCT + BOZORTH3).

NBIS is NIST's public-domain biometric software.  MINDTCT extracts a minutiae
template (an ASCII ".xyt" file) from a fingerprint image, BOZORTH3 compares two
templates and returns an integer similarity score.  Both are designed for
~500 dpi images, which matches the R503 sensor (508 dpi, 192x192).

This module does not replace fingerprint_processor_improved.py - the SIFT/ORB
engine is kept intact as a fallback.  Selection happens in fingerprint_gym.py.

Layout:
    NbisEngine          - binary discovery, template extraction, matching
    GalleryCache        - materialises enrolled templates as .xyt files on disk
    identify()          - 1:N search over fingerprint profiles

Because the exact BOZORTH3 command line differs between NBIS builds, the batch
mode is *probed* at runtime (see NbisEngine._detect_batch_mode) instead of being
hard-coded, with a correct-but-slower per-pair mode as the final fallback.

Self test (run after installing NBIS):
    py -3 fingerprint_nbis.py --selftest
    py -3 fingerprint_nbis.py --bench fingerprint_storage/images
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Configuration (all overridable through environment variables)
# --------------------------------------------------------------------------

# Directory holding mindtct/bozorth3/cwsq. Empty -> search PATH and well-known dirs.
NBIS_BIN_DIR = os.environ.get("NBIS_BIN_DIR", "")

# Scanner resolution in dots per inch. R503 = 508 dpi, FPM383 = 508 dpi.
NBIS_PPI = int(os.environ.get("NBIS_PPI", "508"))

# BOZORTH3 score threshold. NIST guidance: 24 = loose, 40 = standard operating
# point, higher = fewer false accepts. For 1:N identification over many members
# use 40 or above; tune with test/test_nbis_pipeline.py on your own captures.
NBIS_MATCH_THRESHOLD = int(os.environ.get("NBIS_MATCH_THRESHOLD", "40"))

# Optional separation requirement: the best client must beat the best *other*
# client by this ratio. 1.0 disables it. Helps a lot as the member count grows.
NBIS_MATCH_MARGIN = float(os.environ.get("NBIS_MATCH_MARGIN", "1.0"))

# mindtct -b (contrast boost). Recommended for low contrast sensor images,
# including the 4-bit greyscale images the R503 produces over UART.
NBIS_CONTRAST_BOOST = os.environ.get("NBIS_CONTRAST_BOOST", "1") not in ("0", "false", "False")

# Where enrolled templates are materialised for BOZORTH3.
NBIS_GALLERY_DIR = os.environ.get(
    "NBIS_GALLERY_DIR",
    str(Path(__file__).resolve().parent / "fingerprint_storage" / "nbis_gallery"),
)

# Max gallery files per BOZORTH3 invocation (keeps the command line sane).
_CHUNK_SIZE = 200

_SEARCH_DIRS = [
    r"C:\Program Files\NBIS\bin",
    r"C:\Program Files (x86)\NBIS\bin",
    r"C:\NBIS\bin",
    r"C:\nbis\bin",
    "/usr/local/nbis/bin",
    "/usr/local/bin",
    "/usr/bin",
]

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


class NbisNotAvailable(RuntimeError):
    """Raised when the NBIS binaries cannot be located or do not work."""


def _find_binary(name: str, bin_dir: str = "") -> Optional[str]:
    """Locate an NBIS executable in bin_dir, PATH, or the usual install dirs."""
    candidates = []
    if bin_dir:
        candidates.append(Path(bin_dir))
    candidates.extend(Path(d) for d in _SEARCH_DIRS)

    for directory in candidates:
        for exe in (name, name + ".exe"):
            path = directory / exe
            if path.is_file():
                return str(path)

    found = shutil.which(name)
    return found


def _run(cmd: Sequence[str], timeout: float = 30.0) -> subprocess.CompletedProcess:
    """Run a command without popping up a console window on Windows."""
    return subprocess.run(
        list(cmd),
        capture_output=True,
        text=True,
        timeout=timeout,
        creationflags=_CREATE_NO_WINDOW,
    )


def _write_pgm(image: np.ndarray, path: Path) -> None:
    """Write a grayscale numpy image as a binary PGM (P5)."""
    if image.ndim != 2:
        raise ValueError(f"expected a 2D grayscale image, got shape {image.shape}")
    img = np.ascontiguousarray(image.astype(np.uint8))
    height, width = img.shape
    with open(path, "wb") as handle:
        handle.write(b"P5\n%d %d\n255\n" % (width, height))
        handle.write(img.tobytes())


def parse_xyt(xyt_text: str) -> List[Tuple[int, int, int, int]]:
    """Parse an .xyt template into (x, y, theta, quality) tuples."""
    minutiae = []
    for line in xyt_text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            x, y, theta = int(parts[0]), int(parts[1]), int(parts[2])
            quality = int(parts[3]) if len(parts) > 3 else 0
        except ValueError:
            continue
        minutiae.append((x, y, theta, quality))
    return minutiae


class NbisEngine:
    """MINDTCT + BOZORTH3 wrapper.

    Everything that varies between NBIS builds (image format accepted by
    mindtct, BOZORTH3 batch syntax) is detected on first use and cached.
    """

    def __init__(
        self,
        bin_dir: str = NBIS_BIN_DIR,
        ppi: int = NBIS_PPI,
        threshold: int = NBIS_MATCH_THRESHOLD,
        contrast_boost: bool = NBIS_CONTRAST_BOOST,
        verbose: bool = True,
    ):
        self.ppi = ppi
        self.threshold = threshold
        self.contrast_boost = contrast_boost
        self.verbose = verbose

        self.mindtct = _find_binary("mindtct", bin_dir)
        self.bozorth3 = _find_binary("bozorth3", bin_dir)
        self.cwsq = _find_binary("cwsq", bin_dir)

        self._lock = threading.RLock()
        self._image_mode: Optional[str] = None   # "pgm" or "wsq"
        self._batch_mode: Optional[str] = None   # "pairs", "probe-many", "positional", "single"
        self._workdir = Path(tempfile.gettempdir()) / "gym_nbis"
        self._workdir.mkdir(parents=True, exist_ok=True)

        self.available = bool(self.mindtct and self.bozorth3)
        if self.verbose:
            if self.available:
                print(f"[OK] NBIS engine found: mindtct={self.mindtct}")
                print(f"                        bozorth3={self.bozorth3}")
                if self.cwsq:
                    print(f"                        cwsq={self.cwsq}")
                print(f"  - PPI: {self.ppi}, threshold: {self.threshold}, boost: {self.contrast_boost}")
            else:
                missing = [n for n, p in (("mindtct", self.mindtct), ("bozorth3", self.bozorth3)) if not p]
                print(f"[WARNING] NBIS not available (missing: {', '.join(missing)})")
                print("          Set NBIS_BIN_DIR or see NBIS_SETUP.md. NBIS is optional:"
                      " fingerprint_local.py is a pure-Python replacement.")

    # -- capability probing -------------------------------------------------

    def _require(self) -> None:
        if not self.available:
            raise NbisNotAvailable("mindtct/bozorth3 not found - see NBIS_SETUP.md")

    def _detect_image_mode(self) -> str:
        """Find out which image container this mindtct build reads."""
        if self._image_mode:
            return self._image_mode

        probe = (np.random.default_rng(0).random((64, 64)) * 255).astype(np.uint8)
        with tempfile.TemporaryDirectory(dir=self._workdir) as tmp:
            tmp_path = Path(tmp)
            pgm = tmp_path / "probe.pgm"
            _write_pgm(probe, pgm)
            try:
                result = _run([self.mindtct, str(pgm), str(tmp_path / "out")])
                if (tmp_path / "out.xyt").exists():
                    self._image_mode = "pgm"
                    if self.verbose:
                        print("[OK] mindtct reads PGM directly")
                    return self._image_mode
                stderr = (result.stderr or "").strip()
            except Exception as exc:  # noqa: BLE001 - probing, any failure means "no"
                stderr = str(exc)

            if self.cwsq:
                self._image_mode = "wsq"
                if self.verbose:
                    print("[OK] mindtct will be fed WSQ images via cwsq")
                return self._image_mode

            raise NbisNotAvailable(
                f"mindtct rejected the PGM probe and cwsq is not available ({stderr}). "
                "Install the full NBIS package so cwsq can convert images to WSQ."
            )

    def _detect_batch_mode(self) -> str:
        """Work out how this bozorth3 build wants a 1:N batch, and verify ordering.

        Different NBIS packages expose different multi-file syntaxes, so each
        candidate is validated against templates whose expected ranking is known
        (a copy of the probe must outscore an unrelated template).
        """
        if self._batch_mode:
            return self._batch_mode

        rng = np.random.default_rng(1234)
        probe_lines = []
        for _ in range(40):
            probe_lines.append(
                f"{rng.integers(20, 170)} {rng.integers(20, 170)} {rng.integers(0, 360)} {rng.integers(30, 100)}"
            )
        probe_xyt = "\n".join(probe_lines) + "\n"
        other_lines = []
        for _ in range(40):
            other_lines.append(
                f"{rng.integers(20, 170)} {rng.integers(20, 170)} {rng.integers(0, 360)} {rng.integers(30, 100)}"
            )
        other_xyt = "\n".join(other_lines) + "\n"

        with tempfile.TemporaryDirectory(dir=self._workdir) as tmp:
            tmp_path = Path(tmp)
            probe = tmp_path / "probe.xyt"
            mate = tmp_path / "mate.xyt"      # identical to probe -> high score
            other = tmp_path / "other.xyt"    # unrelated       -> low score
            probe.write_text(probe_xyt)
            mate.write_text(probe_xyt)
            other.write_text(other_xyt)
            gallery = [mate, other]

            pairs = tmp_path / "pairs.lst"
            pairs.write_text("".join(f"{probe} {g}\n" for g in gallery))

            candidates = [
                ("pairs", [self.bozorth3, "-A", "outfmt=s", "-M", str(pairs)]),
                ("probe-many", [self.bozorth3, "-A", "outfmt=s", "-p", str(probe)] + [str(g) for g in gallery]),
                ("positional", [self.bozorth3, "-A", "outfmt=s", str(probe)] + [str(g) for g in gallery]),
            ]

            for mode, cmd in candidates:
                try:
                    result = _run(cmd, timeout=20.0)
                except Exception:  # noqa: BLE001
                    continue
                if result.returncode != 0:
                    continue
                scores = self._parse_scores(result.stdout)
                # Correct mode: one score per gallery entry, self-match ranked first.
                if len(scores) == len(gallery) and scores[0] > scores[1]:
                    self._batch_mode = mode
                    if self.verbose:
                        print(f"[OK] bozorth3 batch mode: {mode} (self-match {scores[0]}, unrelated {scores[1]})")
                    return mode

            # Last resort: one process per comparison. Correct, just slower.
            self._batch_mode = "single"
            if self.verbose:
                print("[WARNING] bozorth3 batch mode not detected - using one process per comparison")
            return self._batch_mode

    @staticmethod
    def _parse_scores(stdout: str) -> List[int]:
        """Take the first integer on each non-empty output line."""
        scores = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            token = line.split()[0]
            try:
                scores.append(int(token))
            except ValueError:
                continue
        return scores

    # -- template extraction ------------------------------------------------

    def extract_template(self, image: np.ndarray) -> Optional[Dict]:
        """Run MINDTCT on an image.

        Returns {'xyt': str, 'minutiae_count': int, 'quality': float} or None.
        """
        self._require()
        if image is None:
            return None

        with self._lock:
            mode = self._detect_image_mode()

        try:
            with tempfile.TemporaryDirectory(dir=self._workdir) as tmp:
                tmp_path = Path(tmp)
                oroot = tmp_path / "out"

                if mode == "pgm":
                    source = tmp_path / "input.pgm"
                    _write_pgm(image, source)
                else:
                    raw = tmp_path / "input.raw"
                    height, width = image.shape
                    raw.write_bytes(np.ascontiguousarray(image.astype(np.uint8)).tobytes())
                    conv = _run([
                        self.cwsq, "2.25", "wsq", str(raw),
                        "-raw_in", f"{width},{height},8,{self.ppi}",
                    ])
                    source = tmp_path / "input.wsq"
                    if not source.exists():
                        print(f"[ERROR] cwsq failed: {(conv.stderr or conv.stdout or '').strip()}")
                        return None

                cmd = [self.mindtct]
                if self.contrast_boost:
                    cmd.append("-b")
                cmd += [str(source), str(oroot)]
                result = _run(cmd, timeout=30.0)

                xyt_file = Path(str(oroot) + ".xyt")
                if not xyt_file.exists():
                    print(f"[ERROR] mindtct produced no template: {(result.stderr or result.stdout or '').strip()}")
                    return None

                xyt_text = xyt_file.read_text()

            minutiae = parse_xyt(xyt_text)
            if not minutiae:
                return None

            qualities = [m[3] for m in minutiae]
            return {
                "xyt": xyt_text,
                "minutiae_count": len(minutiae),
                "quality": float(np.mean(qualities)) if qualities else 0.0,
            }
        except Exception as exc:  # noqa: BLE001 - never let a scan crash the loop
            print(f"[ERROR] MINDTCT extraction failed: {exc}")
            return None

    # -- matching -----------------------------------------------------------

    def match_files(self, probe_path: Path, gallery_paths: Sequence[Path]) -> List[int]:
        """Compare one probe template against many gallery templates."""
        self._require()
        if not gallery_paths:
            return []

        with self._lock:
            mode = self._detect_batch_mode()

        scores: List[int] = []
        for start in range(0, len(gallery_paths), _CHUNK_SIZE):
            chunk = list(gallery_paths[start:start + _CHUNK_SIZE])
            scores.extend(self._match_chunk(probe_path, chunk, mode))
        return scores

    def _match_chunk(self, probe_path: Path, chunk: List[Path], mode: str) -> List[int]:
        timeout = 30.0 + 0.5 * len(chunk)

        if mode == "single":
            out = []
            for gallery in chunk:
                try:
                    result = _run([self.bozorth3, str(probe_path), str(gallery)], timeout=20.0)
                    parsed = self._parse_scores(result.stdout)
                    out.append(parsed[0] if parsed else 0)
                except Exception:  # noqa: BLE001
                    out.append(0)
            return out

        try:
            if mode == "pairs":
                pairs = self._workdir / f"pairs_{os.getpid()}_{threading.get_ident()}.lst"
                pairs.write_text("".join(f"{probe_path} {g}\n" for g in chunk))
                try:
                    result = _run([self.bozorth3, "-A", "outfmt=s", "-M", str(pairs)], timeout=timeout)
                finally:
                    pairs.unlink(missing_ok=True)
            elif mode == "probe-many":
                result = _run(
                    [self.bozorth3, "-A", "outfmt=s", "-p", str(probe_path)] + [str(g) for g in chunk],
                    timeout=timeout,
                )
            else:  # positional
                result = _run(
                    [self.bozorth3, "-A", "outfmt=s", str(probe_path)] + [str(g) for g in chunk],
                    timeout=timeout,
                )

            scores = self._parse_scores(result.stdout)
            if len(scores) == len(chunk):
                return scores

            print(f"[WARNING] bozorth3 returned {len(scores)} scores for {len(chunk)} templates - retrying per pair")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] bozorth3 batch call failed ({exc}) - retrying per pair")

        return self._match_chunk(probe_path, chunk, "single")

    def match(self, probe_xyt: str, gallery_xyt: str) -> int:
        """Compare two templates held in memory. Returns a BOZORTH3 score."""
        self._require()
        with tempfile.TemporaryDirectory(dir=self._workdir) as tmp:
            tmp_path = Path(tmp)
            probe = tmp_path / "probe.xyt"
            gallery = tmp_path / "gallery.xyt"
            probe.write_text(probe_xyt)
            gallery.write_text(gallery_xyt)
            scores = self.match_files(probe, [gallery])
        return scores[0] if scores else 0


class GalleryCache:
    """Keeps enrolled .xyt templates on disk so BOZORTH3 can read them directly.

    Rebuilt only when the enrolled set changes (signature check) or when
    invalidate() is called after an enrollment.
    """

    def __init__(self, directory: str = NBIS_GALLERY_DIR):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.entries: List[Tuple[int, str, Path]] = []   # (client_id, name, path)
        self._signature: Optional[str] = None
        self._lock = threading.RLock()

    @staticmethod
    def _samples_of(profile: Dict) -> List[Dict]:
        if "fingers" in profile:
            samples = []
            for finger in profile["fingers"]:
                samples.extend(finger.get("samples", []))
            return samples
        return profile.get("fingerprints", [])

    @staticmethod
    def _signature_of(profiles: Sequence[Dict]) -> str:
        digest = hashlib.md5()
        for profile in profiles:
            samples = GalleryCache._samples_of(profile)
            usable = sum(1 for s in samples if isinstance(s, dict) and s.get("xyt"))
            digest.update(f"{profile.get('client_id')}:{usable}|".encode())
        return digest.hexdigest()

    def invalidate(self) -> None:
        with self._lock:
            self._signature = None

    def build(self, profiles: Sequence[Dict], force: bool = False) -> List[Tuple[int, str, Path]]:
        """Materialise templates for all profiles; returns the gallery entries."""
        with self._lock:
            signature = self._signature_of(profiles)
            if not force and signature == self._signature and self.entries:
                return self.entries

            for stale in self.directory.glob("*.xyt"):
                stale.unlink(missing_ok=True)

            entries: List[Tuple[int, str, Path]] = []
            skipped = 0
            for profile in profiles:
                client_id = profile.get("client_id")
                name = profile.get("name", f"Client {client_id}")

                if "fingers" in profile:
                    grouped = [
                        (finger.get("finger_index", 1), finger.get("samples", []))
                        for finger in profile["fingers"]
                    ]
                else:
                    grouped = [(1, profile.get("fingerprints", []))]

                for finger_index, samples in grouped:
                    for sample in samples:
                        if not isinstance(sample, dict):
                            continue
                        xyt = sample.get("xyt")
                        if not xyt:
                            skipped += 1
                            continue
                        sample_number = sample.get("sample_number", len(entries))
                        path = self.directory / f"c{client_id}_f{finger_index}_s{sample_number}.xyt"
                        path.write_text(xyt)
                        entries.append((client_id, name, path))

            self.entries = entries
            self._signature = signature
            clients = len({e[0] for e in entries})
            print(f"[OK] NBIS gallery: {len(entries)} templates from {clients} clients"
                  + (f" ({skipped} samples have no template yet)" if skipped else ""))
            return entries


def identify(
    engine: NbisEngine,
    image_or_xyt,
    profiles: Sequence[Dict],
    gallery: GalleryCache,
    threshold: Optional[int] = None,
    margin: float = NBIS_MATCH_MARGIN,
) -> Optional[Tuple[int, str, int, Dict]]:
    """1:N identification.

    Args:
        image_or_xyt: query image (numpy array) or an already extracted .xyt string
        profiles:     fingerprint profiles as returned by db.get_all_fingerprint_features()
        gallery:      GalleryCache instance holding the enrolled templates
        threshold:    BOZORTH3 score required to accept (default NBIS_MATCH_THRESHOLD)
        margin:       best client must beat the runner-up by this ratio (1.0 = off)

    Returns:
        (client_id, name, score, details) or None. `details` always carries the
        ranking so callers can log near misses while tuning the threshold.
    """
    threshold = engine.threshold if threshold is None else threshold

    if isinstance(image_or_xyt, str):
        query_xyt = image_or_xyt
        minutiae_count = len(parse_xyt(query_xyt))
    else:
        template = engine.extract_template(image_or_xyt)
        if template is None:
            print("[ERROR] MINDTCT found no minutiae in the query image")
            return None
        query_xyt = template["xyt"]
        minutiae_count = template["minutiae_count"]

    entries = gallery.build(profiles)
    if not entries:
        print("[WARNING] NBIS gallery is empty - no enrolled templates")
        return None

    started = time.time()
    with tempfile.TemporaryDirectory(dir=engine._workdir) as tmp:
        probe_path = Path(tmp) / "probe.xyt"
        probe_path.write_text(query_xyt)
        scores = engine.match_files(probe_path, [entry[2] for entry in entries])
    elapsed = time.time() - started

    # Best score per client (standard for multi-sample enrollment).
    per_client: Dict[int, Tuple[str, int]] = {}
    for (client_id, name, _path), score in zip(entries, scores):
        current = per_client.get(client_id)
        if current is None or score > current[1]:
            per_client[client_id] = (name, score)

    ranking = sorted(
        ((cid, name, score) for cid, (name, score) in per_client.items()),
        key=lambda item: item[2],
        reverse=True,
    )
    details = {
        "minutiae": minutiae_count,
        "compared": len(scores),
        "elapsed": elapsed,
        "ranking": ranking[:5],
    }

    if not ranking:
        return None

    best_id, best_name, best_score = ranking[0]
    runner_up = ranking[1][2] if len(ranking) > 1 else 0

    if best_score < threshold:
        return None
    if margin > 1.0 and runner_up > 0 and best_score < runner_up * margin:
        print(f"[WARNING] NBIS rejected an ambiguous match: {best_score} vs {runner_up} (margin {margin})")
        return None

    return (best_id, best_name, best_score, details)


# --------------------------------------------------------------------------
# Command line helpers
# --------------------------------------------------------------------------

def _selftest() -> int:
    print("=" * 60)
    print("NBIS engine self test")
    print("=" * 60)
    engine = NbisEngine()
    if not engine.available:
        print("\nRESULT: NBIS binaries not found. See NBIS_SETUP.md")
        return 1

    try:
        mode = engine._detect_image_mode()
        print(f"  image input mode : {mode}")
        batch = engine._detect_batch_mode()
        print(f"  bozorth3 batch   : {batch}")
    except NbisNotAvailable as exc:
        print(f"\nRESULT: {exc}")
        return 1

    rng = np.random.default_rng(7)
    image = (rng.random((192, 192)) * 255).astype(np.uint8)
    template = engine.extract_template(image)
    if template is None:
        print("  extraction on noise produced no minutiae (expected for pure noise)")
    else:
        print(f"  noise image      : {template['minutiae_count']} minutiae, quality {template['quality']:.1f}")

    print("\nRESULT: NBIS engine is usable.")
    print("Now run the accuracy benchmark on real captures:")
    print("  py -3 test/test_nbis_pipeline.py")
    return 0


def _bench(image_dir: str) -> int:
    import cv2

    engine = NbisEngine()
    if not engine.available:
        print("NBIS binaries not found - see NBIS_SETUP.md")
        return 1

    files = sorted(Path(image_dir).glob("*.png"))
    if not files:
        print(f"No images in {image_dir}")
        return 1

    started = time.time()
    count = 0
    for path in files[:20]:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        template = engine.extract_template(image)
        count += 1
        print(f"  {path.name}: "
              + (f"{template['minutiae_count']} minutiae, quality {template['quality']:.1f}"
                 if template else "no template"))
    print(f"\n{count} images in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    if "--bench" in sys.argv:
        index = sys.argv.index("--bench")
        directory = sys.argv[index + 1] if len(sys.argv) > index + 1 else "fingerprint_storage/images"
        raise SystemExit(_bench(directory))
    print(__doc__)
