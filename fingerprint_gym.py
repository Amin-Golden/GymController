#!/usr/bin/env python3
"""
Fingerprint Integration Module for Gym Management System
Integrates with RetinaFaceL.py to provide fingerprint enrollment and identification
"""

import socket
import struct
import numpy as np
import cv2
import time
from datetime import datetime
import threading
import pickle
import os
import textwrap
from fingerprint_processor_improved import ImprovedFingerprintProcessor, FAISS_AVAILABLE

# NBIS engine (MINDTCT + BOZORTH3). Needs native binaries from NIST.
try:
    import fingerprint_nbis
    NBIS_MODULE_AVAILABLE = True
except Exception as _nbis_import_error:  # noqa: BLE001
    NBIS_MODULE_AVAILABLE = False
    print(f"[WARNING] fingerprint_nbis not importable ({_nbis_import_error})")

# Pure-Python minutiae engine. Same interface as fingerprint_nbis but with no
# native dependencies, so it runs where NBIS cannot be installed (OrangePi 5B).
try:
    import fingerprint_local
    LOCAL_MODULE_AVAILABLE = True
except Exception as _local_import_error:  # noqa: BLE001
    LOCAL_MODULE_AVAILABLE = False
    print(f"[WARNING] fingerprint_local not importable ({_local_import_error})")

# Grayscale SIFT engine with geometric verification and a FAISS index - the
# production matcher (see fingerprint_sift.py for the measurements).
try:
    import fingerprint_sift
    GRAY_MODULE_AVAILABLE = fingerprint_sift.FAISS_AVAILABLE
    if not GRAY_MODULE_AVAILABLE:
        print("[WARNING] faiss is not installed - the grayscale fingerprint engine is unavailable "
              "(pip install faiss-cpu)")
except Exception as _gray_import_error:  # noqa: BLE001
    GRAY_MODULE_AVAILABLE = False
    print(f"[WARNING] fingerprint_sift not importable ({_gray_import_error})")

# Commands to the ESP32 for the sensor double-check (CONFIRM / MATCHTMPL / IMG2TMPL).
try:
    import r503_link
    R503_LINK_AVAILABLE = True
except Exception as _link_import_error:  # noqa: BLE001
    R503_LINK_AVAILABLE = False
    print(f"[WARNING] r503_link not importable ({_link_import_error}) - no sensor double-check")

# Raw image retention, so the matching algorithm can be changed later without
# asking every member to enroll again.
try:
    import fingerprint_image_store
    IMAGE_STORE_AVAILABLE = True
except Exception as _store_import_error:  # noqa: BLE001
    IMAGE_STORE_AVAILABLE = False
    print(f"[WARNING] fingerprint_image_store not importable ({_store_import_error}) - images will not be kept")

# Configuration
UDP_PORT_ENTRANCE = 8889  # Port for entrance ESP32 (enrollment & entry)
UDP_PORT_LOCKER = 8890    # Port for locker ESP32 (unlock only)
# Fallback geometry, only used when the image header is missed. R503 sends
# 192x192; set FINGERPRINT_IMAGE_WIDTH/HEIGHT if your sensor differs.
IMAGE_WIDTH = int(os.environ.get("FINGERPRINT_IMAGE_WIDTH", "192"))
IMAGE_HEIGHT = int(os.environ.get("FINGERPRINT_IMAGE_HEIGHT", "192"))
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
EXPECTED_MARKER = b"FPIMG\0\0\0"
END_MARKER = b"FPIMGEND"

# Fingerprint matching parameters
FINGERS_PER_CLIENT = 2

# Presses per finger. Like a phone sensor, enrollment asks for each press at a
# different part of the finger and stitches them into one whole-finger template
# (fingerprint_local.fuse_presses), because one R503 press sees only ~8 mm.
SAMPLES_PER_FINGER = int(os.environ.get("ENROLL_PRESSES", "8"))

# If the database refuses all presses (an admin-app limit of 5 samples per
# finger), enrollment retries with this many of the strongest presses.
FALLBACK_STORED_PRESSES = 5

# Characters per line on the entrance LCD / PC log. Longer enrollment messages
# are wrapped onto extra lines instead of running off the screen.
LCD_LINE_WIDTH = int(os.environ.get("LCD_LINE_WIDTH", "40"))

# What to ask for on each press (cycled), shown on the entrance LCD - ASCII only.
# Small shifts only: a press is stitched through the minutiae it shares with
# earlier ones, and on enrollment day presses moved to the tip / joint / a
# rolled edge mostly shared too few and were not stitched. Moving a few mm
# keeps most of the window on known ridges while still adding new area.
ENROLL_PROMPTS = [
    "CENTER of the finger",
    "CENTER again",
    "move finger a LITTLE UP",
    "move finger a LITTLE DOWN",
    "move finger a LITTLE LEFT",
    "move finger a LITTLE RIGHT",
    "a little UP and LEFT",
    "a little UP and RIGHT",
]

# Which matcher to use:
#   "auto"  - the grayscale engine (fingerprint_sift) when available (default)
#   "gray"  - the grayscale engine only
#   "nbis"  - NBIS only (needs mindtct/bozorth3 native binaries)
#   "local" - pure-Python minutiae engine only (no native dependencies)
#   "sift"  - the original SIFT/ORB + FAISS engine only
#
# The grayscale engine needs the firmware that uploads the sensor's grayscale
# image (finger_UDP.ino: UpImage before Img2Tz) and members enrolled with it;
# presses enrolled with the other engines are not read by it. Those engines
# matched the two-level skeleton older firmware sent and are kept only for
# reference: on the same presses they identified far fewer members.
FINGERPRINT_ENGINE = os.environ.get("FINGERPRINT_ENGINE", "auto").lower()
GRAY_ENGINE = FINGERPRINT_ENGINE in ("auto", "gray") and GRAY_MODULE_AVAILABLE
if FINGERPRINT_ENGINE == "gray" and not GRAY_MODULE_AVAILABLE:
    print("[ERROR] FINGERPRINT_ENGINE=gray but fingerprint_sift/faiss is unavailable - no fingerprint matching")

# When an enrolling finger matches ANOTHER member at the match threshold:
# "reject" refuses to save it (a double enrollment, or two members the door
# would confuse), "warn" saves it anyway and only reports it.
SIFT_ENROLL_COLLISION = os.environ.get("SIFT_ENROLL_COLLISION", "reject").lower()

# Sensor double-check: a match scoring at least SIFT_MATCH_INLIERS but below
# this is accepted only if the R503's own matcher - a different algorithm -
# agrees on the same press. Clear matches (genuine presses median ~170) are not
# slowed down; a double-check costs ~4 s. 0 turns it off.
SIFT_CONFIRM_BELOW = int(os.environ.get("SIFT_CONFIRM_BELOW", "40"))

# Presses per finger whose R503 template is stored at enrollment (~3.5 s each)
# and tried, one by one, at a double-check.
SENSOR_TEMPLATES_PER_FINGER = int(os.environ.get("SENSOR_TEMPLATES_PER_FINGER", "3"))

# In "auto" mode, fall back to the SIFT engine when the minutiae engine reports
# no match. Useful while members enrolled before the migration have no template.
NBIS_FALLBACK_TO_SIFT = os.environ.get("NBIS_FALLBACK_TO_SIFT", "1") not in ("0", "false", "False")

# Minutiae required to accept an enrollment sample. A weak impression enrolled
# now can never be matched reliably later, so ask for another press instead.
# Single source of truth is fingerprint_local.MIN_ENROLL_MINUTIAE - see its
# comment there for the measurement behind the current value. Falls back to a
# literal only if that module failed to import.
MIN_ENROLL_MINUTIAE = int(os.environ.get(
    "MIN_ENROLL_MINUTIAE",
    str(fingerprint_local.MIN_ENROLL_MINUTIAE) if LOCAL_MODULE_AVAILABLE else "10"))

# Extra presses allowed while rejecting weak ones, before giving up.
ENROLL_RETRY_BUDGET = int(os.environ.get("ENROLL_RETRY_BUDGET", "5"))

# SIFT engine threshold. This is a ratio-test match score, NOT a cosine
# similarity: measured genuine scores sit around 0.04-0.28, so anything above
# ~0.3 rejects everyone. Only the SIFT path uses it.
MATCH_THRESHOLD = float(os.environ.get("SIFT_MATCH_THRESHOLD", "0.12"))

# ESP32 endpoints for feedback
ENTRANCE_ESP_IP = "192.168.1.120"  # Entrance/enrollment ESP32
ENTRANCE_ESP_PORT = 4210
LOCKER_ESP_IP = "192.168.1.121"    # Locker ESP32
LOCKER_ESP_PORT = 4210
pc_ip = "192.168.1.3"            # PC
pc_port = 4211
raspi_entrance_ip = "192.168.1.111"
raspi_entrance_port = 4210

# --------------------------------------------------------------------------
# Enrollment preview for the OrangePi screen
# --------------------------------------------------------------------------
# Each enrollment capture is drawn with its minutiae and the accept/reject
# verdict, so a weak press (e.g. 15 minutiae against MIN_ENROLL_MINUTIAE) can
# be seen. Enrollment runs on a worker thread and OpenCV windows belong to the
# main thread, so this only stores the picture; RetinaFaceL's main loop shows it.

PREVIEW_SCALE = 3
_preview_lock = threading.Lock()
_preview = None          # (sequence number, BGR image)
_preview_seq = 0


def _render_whole_finger(xyt, size):
    """Square panel plotting the stitched whole-finger template, scaled to fit."""
    panel = np.full((size, size, 3), 255, np.uint8)
    pts = [tuple(map(float, l.split()[:3])) for l in (xyt or "").splitlines() if len(l.split()) >= 3]
    cv2.putText(panel, f"Whole finger: {len(pts)} minutiae", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 60, 0), 2)
    if not pts:
        return panel
    xy = np.array([p[:2] for p in pts])
    lo, span = xy.min(axis=0), max(np.ptp(xy, axis=0).max(), 1.0)
    pad = 40
    k = (size - 2 * pad) / span
    for x, y, t in pts:
        px, py = int(pad + (x - lo[0]) * k), int(pad + (y - lo[1]) * k)
        th = np.radians(t)
        cv2.circle(panel, (px, py), 5, (200, 90, 0), 2)
        cv2.line(panel, (px, py), (int(px + 12 * np.cos(th)), int(py + 12 * np.sin(th))), (200, 90, 0), 2)
    return panel


def render_fingerprint_preview(image, title, xyt=None, verdict="", ok=True, whole_xyt=None):
    """Fingerprint enlarged, minutiae marked, title and verdict above it.

    With whole_xyt, the stitched whole-finger template is drawn beside it, so
    coverage can be watched growing press by press.
    """
    s = PREVIEW_SCALE
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, (gray.shape[1] * s, gray.shape[0] * s), interpolation=cv2.INTER_CUBIC)
    big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)

    color = (0, 200, 0) if ok else (0, 0, 255)          # BGR: green accepted, red rejected
    for line in (xyt or "").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        x, y, theta = int(parts[0]) * s, int(parts[1]) * s, np.radians(float(parts[2]))
        cv2.circle(big, (x, y), 2 * s, color, 2)
        # direction in image coordinates (y grows downward), as both engines write it
        end = (int(x + 5 * s * np.cos(theta)), int(y + 5 * s * np.sin(theta)))
        cv2.line(big, (x, y), end, color, 2)

    if whole_xyt is not None:
        big = np.hstack([big, _render_whole_finger(whole_xyt, big.shape[0])])

    header = np.full((70, big.shape[1], 3), 30, np.uint8)
    cv2.putText(header, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(header, verdict, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return np.vstack([header, big])


def publish_fingerprint_preview(image, title, xyt=None, verdict="", ok=True, whole_xyt=None):
    """Render a capture for the main thread to show; never lets a drawing error stop enrollment."""
    global _preview, _preview_seq
    try:
        picture = render_fingerprint_preview(image, title, xyt, verdict, ok, whole_xyt)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  Fingerprint preview failed: {e}")
        return
    with _preview_lock:
        _preview_seq += 1
        _preview = (_preview_seq, picture)


def take_fingerprint_preview(last_seq=0):
    """Return (sequence, image) when a capture newer than last_seq is waiting, else None."""
    with _preview_lock:
        if _preview is not None and _preview[0] != last_seq:
            return _preview
    return None

# --------------------------------------------------------------------------
# NBIS engine access (shared by every fingerprint system instance)
# --------------------------------------------------------------------------

_engine_module = None
_engine = None
_engine_gallery = None
_engine_checked = False
_engine_lock = threading.Lock()


def get_minutiae_engine():
    """Return (module, engine, gallery) for the minutiae matcher.

    Both fingerprint_nbis and fingerprint_local expose the same three things -
    an engine with extract_template(), a GalleryCache and identify() - so the
    rest of this file does not care which one it got.

    Selection runs once per process, including when it fails: binary discovery
    and the bozorth3 capability probe are not worth repeating per sensor, and
    reprinting the failure on every scan only buries the real logs.
    """
    global _engine_module, _engine, _engine_gallery, _engine_checked

    if FINGERPRINT_ENGINE == "sift" or GRAY_ENGINE:
        return (None, None, None)

    with _engine_lock:
        if _engine_checked:
            return (_engine_module, _engine, _engine_gallery)
        _engine_checked = True

        if FINGERPRINT_ENGINE in ("auto", "nbis") and NBIS_MODULE_AVAILABLE:
            try:
                engine = fingerprint_nbis.NbisEngine()
                if engine.available:
                    _engine_module = fingerprint_nbis
                    _engine = engine
                    _engine_gallery = fingerprint_nbis.GalleryCache()
                    return (_engine_module, _engine, _engine_gallery)
                if FINGERPRINT_ENGINE == "nbis":
                    print("[ERROR] FINGERPRINT_ENGINE=nbis but the NBIS binaries are missing - see NBIS_SETUP.md")
                    return (None, None, None)
            except Exception as e:  # noqa: BLE001
                print(f"[WARNING] Could not start the NBIS engine: {e}")
                if FINGERPRINT_ENGINE == "nbis":
                    return (None, None, None)

        if FINGERPRINT_ENGINE in ("auto", "local") and LOCAL_MODULE_AVAILABLE:
            # Messages here stay ASCII: this runs before anything else prints,
            # and a UnicodeEncodeError on a cp1252 console would otherwise be
            # caught below and leave the system with no matcher at all.
            try:
                engine = fingerprint_local.LocalEngine()
                _engine_module = fingerprint_local
                _engine = engine
                _engine_gallery = fingerprint_local.GalleryCache()
            except Exception as e:  # noqa: BLE001
                print(f"[ERROR] Could not start the local engine: {e}")
            else:
                if FINGERPRINT_ENGINE == "auto":
                    print("[INFO] NBIS not installed - using the pure-Python minutiae engine")

    return (_engine_module, _engine, _engine_gallery)


# Kept for callers written against the NBIS-only API.
def get_nbis():
    module, engine, gallery = get_minutiae_engine()
    return (engine, gallery)


def engine_invalidate_gallery():
    """Force the enrolled-template gallery to be rebuilt (after an enrollment)."""
    if _engine_gallery is not None:
        _engine_gallery.invalidate()


# Backwards-compatible alias.
nbis_invalidate_gallery = engine_invalidate_gallery


_gray_gallery = None
_gray_lock = threading.Lock()


def get_gray_gallery():
    """The process-wide grayscale gallery (FAISS index + enrolled presses), or None."""
    global _gray_gallery
    if not GRAY_ENGINE:
        return None
    with _gray_lock:
        if _gray_gallery is None:
            _gray_gallery = fingerprint_sift.SiftGallery()
        return _gray_gallery


def _keypoints_xyt(features, n=60):
    """Strongest keypoints as .xyt lines, so the enrollment preview can mark them."""
    if features is None:
        return None
    return "".join(f"{int(x)} {int(y)} {int(np.degrees(a)) % 360} 50\n"
                   for (x, y), a in zip(features.pts[:n], features.ang[:n]))


def build_sample_record(image, processor, client_id=None, finger_index=None, sample_number=None,
                        nbis_engine=None):
    """Turn one captured image into a database-ready sample.

    Runs both engines so the stored row serves either matcher, and keeps the raw
    image on disk so a future algorithm can be rebuilt from it.

    Returns the sample dict, or None when neither engine could use the image.
    """
    record = {}

    # 1. Raw image first - it is the only artefact that survives an algorithm change
    if IMAGE_STORE_AVAILABLE and client_id is not None:
        path = fingerprint_image_store.save_enrollment_image(image, client_id, finger_index, sample_number)
        if path:
            record['image_path'] = path

    # Grayscale engine: its packed keypoints are all the database needs
    if GRAY_ENGINE:
        features = fingerprint_sift.extract(image)
        if features is None:
            return None
        record.update(sift_gray=fingerprint_sift.pack(features), num_features=features.n,
                      gray=features, keypoints=[], confidence=1.0)
        return record

    # 2. SIFT/ORB descriptors (original engine, still stored)
    features = processor.extract_features(image)
    if features:
        record.update(features)

    # 3. NBIS minutiae template
    engine = nbis_engine if nbis_engine is not None else get_nbis()[0]
    if engine is not None:
        template = engine.extract_template(image)
        if template:
            record['xyt'] = template['xyt']
            record['minutiae_count'] = template['minutiae_count']
            record['template_quality'] = template['quality']

    # Keep the sample if either engine got something usable out of it
    if record.get('descriptors') is None and not record.get('xyt'):
        return None

    record.setdefault('num_features', 0)
    record.setdefault('confidence', 1.0)
    record.setdefault('keypoints', [])
    return record


class FingerprintProcessor:
    """Processes fingerprint images and extracts features"""
    
    def __init__(self):
        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=500)
        
        # Try to initialize SIFT (more accurate)
        try:
            self.sift = cv2.SIFT_create()
            self.use_sift = True
            print("✓ Using SIFT for fingerprint feature extraction")
        except:
            self.use_sift = False
            print("✓ Using ORB for fingerprint feature extraction")
    
    def preprocess_image(self, image: np.ndarray):
        """Preprocess fingerprint image"""
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(image)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10)
        
        # Normalize
        normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized
    
    def extract_features(self, image: np.ndarray):
        """Extract features from fingerprint image"""
        try:
            # Preprocess
            processed = self.preprocess_image(image)
            
            # Extract keypoints and descriptors
            if self.use_sift:
                keypoints, descriptors = self.sift.detectAndCompute(processed, None)
            else:
                keypoints, descriptors = self.orb.detectAndCompute(processed, None)
            
            if descriptors is None or len(descriptors) < 10:
                return None
            
            return {
                'descriptors': descriptors,
                'keypoints': [(kp.pt, kp.size, kp.angle) for kp in keypoints],
                'num_features': len(descriptors),
                'confidence': 1.0
            }
        except Exception as e:
            print(f"✗ Feature extraction error: {e}")
            return None
    
    def match_fingerprints(self, features1, features2):
        """
        Match two fingerprints and return similarity score (0-1)
        """
        if features1 is None or features2 is None:
            return 0.0
        
        desc1 = features1['descriptors']
        desc2 = features2['descriptors']
        
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return 0.0
        
        # Use BFMatcher
        if self.use_sift:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        try:
            # KNN matching
            matches = bf.knnMatch(desc1, desc2, k=2)
            
            # Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            # Calculate score with improved normalization
            if len(desc1) > 0 and len(desc2) > 0:
                avg_descriptors = (len(desc1) + len(desc2)) / 2.0
                score = len(good_matches) / avg_descriptors
                
                # Boost for strong matches
                if len(good_matches) > 20:
                    score = score * 1.1
                
                return min(score, 1.0)
            return 0.0
                
        except Exception as e:
            print(f"✗ Matching error: {e}")
            return 0.0


class UDPFingerprintReceiver:
    """Receives fingerprint images over UDP"""
    
    def __init__(self, port):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', port))
        self.sock.settimeout(1.0)
        # (sender IP, header timestamp) of the last image: names the press for
        # the sensor double-check (the ESP32 keeps that image for a moment)
        self.last_press = None
        print(f"✓ Fingerprint UDP receiver listening on port {port}")
    
    def receive_image(self, timeout=30.0):
        """
        Receive a complete fingerprint image
        Returns: numpy array of image or None
        """
        # print("\n⏳ Waiting for fingerprint image...")
        start_time = time.time()
        
        header_received = False
        image_data = bytearray()
        expected_size = IMAGE_SIZE
        image_width = IMAGE_WIDTH
        image_height = IMAGE_HEIGHT
        
        while time.time() - start_time < timeout:
            try:
                data, addr = self.sock.recvfrom(2048)
                
                # Check for header
                if not header_received and data[:8] == EXPECTED_MARKER:
                    if len(data) >= 20:
                        width, height, size, timestamp = struct.unpack('<HHI I', data[8:20])
                        image_width = int(width)
                        image_height = int(height)
                        print(f"📥 Receiving fingerprint: {image_width}x{image_height}, {size} bytes")
                        header_received = True
                        expected_size = int(size)
                        self.last_press = (addr[0], int(timestamp))
                    continue
                
                # Check for end marker
                if data == END_MARKER:
                    if len(image_data) == expected_size:
                        print(f"✓ Fingerprint received ({len(image_data)} bytes)")
                        # Convert to numpy array
                        img_array = np.frombuffer(image_data, dtype=np.uint8)
                        try:
                            img_array = img_array.reshape((image_height, image_width))
                        except Exception:
                            # Fallback to default constants if reshape fails
                            img_array = img_array.reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
                        return img_array
                    else:
                        print(f"✗ Size mismatch: got {len(image_data)}, expected {expected_size}")
                        return None
                
                # Accumulate image data
                if header_received:
                    image_data.extend(data)
                    progress = len(image_data) / expected_size * 100
                    print(f"  Progress: {progress:.1f}%", end='\r')
            
            except socket.timeout:
                continue
            except Exception as e:
                print(f"✗ Error receiving data: {e}")
                return None
        
        # print("\n✗ Timeout waiting for fingerprint image")
        return None
    
    def close(self):
        """Close the socket"""
        self.sock.close()


class DualESP32FingerprintSystem:
    """
    Manages two ESP32 fingerprint sensors:
    1. Entrance ESP32 - For gym entry and enrollment
    2. Locker ESP32 - For locker unlocking
    """
    
    def __init__(self, db_helper, entrance_callback=None, locker_callback=None,
                 sock=None, fingerprint_cache=None, pause_callback=None, resume_callback=None):
        self.db = db_helper
        self.processor = ImprovedFingerprintProcessor(embedding_dim=128, use_pca=True)
        self.nbis_module, self.nbis, self.nbis_gallery = get_minutiae_engine()
        self.gray = get_gray_gallery()
        self.sock = sock
        self.fingerprint_cache = fingerprint_cache  # Shared cache from RetinaFaceL
        self.pause_callback = pause_callback
        self.resume_callback = resume_callback
        self._pause_active = False
        
        # Create two receivers for different ports
        self.entrance_receiver = UDPFingerprintReceiver(UDP_PORT_ENTRANCE)
        self.locker_receiver = UDPFingerprintReceiver(UDP_PORT_LOCKER)
        
        # Load registered fingerprints (use cache if provided, otherwise load from DB)
        if fingerprint_cache:
            self.registered_fingerprints = fingerprint_cache.get_fingerprints()
            print(f"✅ Using cached fingerprints: {len(self.registered_fingerprints)} profiles")
        else:
            self.registered_fingerprints = []
            self.reload_fingerprints()
        
        # Callbacks for different ESP32s
        self.entrance_callback = entrance_callback
        self.locker_callback = locker_callback
        
        # Threading for both ESP32s
        self.entrance_thread = None
        self.locker_thread = None
        self.entrance_running = False
        self.locker_running = False
        self.enrollment_lock = threading.RLock()
        self.enrollment_active = threading.Event()

        self._warm_gray_gallery()
        print("✅ Dual ESP32 fingerprint system initialized")

    def _warm_gray_gallery(self):
        """Load the enrolled presses into the grayscale index in the background,
        so the first member at the door does not wait for it."""
        if self.gray is None:
            return
        profiles = list(self.registered_fingerprints)

        def warm():
            try:
                self.gray.sync(profiles)
                st = self.gray.stats()
                old = sum(1 for p in profiles if not any(s.get('sift_gray') for s in p.get('fingerprints', [])))
                print(f"[SIFT] ready: {st['members']} members, {st['fingers']} fingers, {st['presses']} presses")
                if old:
                    print(f"[SIFT] {old} members have only old-format fingerprints - they must enroll again")
            except Exception as e:  # noqa: BLE001
                print(f"❌ Could not load the fingerprint gallery: {e}")
        threading.Thread(target=warm, daemon=True).start()

    def set_fingerprint_cache(self, fingerprint_cache):
        """Set or update the fingerprint cache reference"""
        self.fingerprint_cache = fingerprint_cache
        if fingerprint_cache:
            self.registered_fingerprints = fingerprint_cache.get_fingerprints()
            self._warm_gray_gallery()
    
    def _request_pause(self, reason="fingerprint enrollment"):
        """Request main pipeline pause, returns True if pause acquired."""
        if self.pause_callback and not self._pause_active:
            try:
                self.pause_callback(reason)
            except TypeError:
                self.pause_callback()
            self._pause_active = True
            return True
        return False

    def _status(self, text, console=True):
        """Show a message on the PC log and the entrance LCD, one display line per packet.

        ASCII only, and wrapped at LCD_LINE_WIDTH so nothing runs off the screen.
        console=False when the caller already printed its own (emoji) version.
        """
        if console:
            print(f"   {text}")
        lcd = text.encode("ascii", "ignore").decode()
        for line in textwrap.wrap(lcd, LCD_LINE_WIDTH) or [""]:
            for endpoint in ((pc_ip, pc_port), (raspi_entrance_ip, raspi_entrance_port)):
                try:
                    self.sock.sendto(line.encode("utf-8"), endpoint)
                except Exception:
                    pass

    def _db_error(self):
        """Why the last fingerprint save failed, as the database reported it."""
        return getattr(self.db, "last_fingerprint_error", None) or "unknown database error (see the terminal)"

    def _release_pause(self):
        """Release main pipeline pause if previously acquired."""
        if self._pause_active and self.resume_callback:
            try:
                self.resume_callback()
            except Exception:
                pass
            self._pause_active = False
    
    def start_auto_identification(self):
        """Start automatic identification for both ESP32s"""
        self.start_entrance_listener()
        self.start_locker_listener()
    
    def start_entrance_listener(self):
        """Start entrance ESP32 listener (enrollment & entry)"""
        if self.entrance_running:
            print("⚠️  Entrance listener already running")
            return
        
        self.entrance_running = True
        self.entrance_thread = threading.Thread(
            target=self._entrance_listener_loop, 
            daemon=True
        )
        self.entrance_thread.start()
        print(f"🚪 Entrance listener started (Port {UDP_PORT_ENTRANCE})")
    
    def start_locker_listener(self):
        """Start locker ESP32 listener (unlock only)"""
        if self.locker_running:
            print("⚠️  Locker listener already running")
            return
        
        self.locker_running = True
        self.locker_thread = threading.Thread(
            target=self._locker_listener_loop, 
            daemon=True
        )
        self.locker_thread.start()
        print(f"🔐 Locker listener started (Port {UDP_PORT_LOCKER})")

    def is_enrollment_in_progress(self):
        """Return True if a fingerprint enrollment flow is currently running."""
        return self.enrollment_active.is_set()

    def perform_enrollment(self, client_id, finger_index=None, use_entrance_esp=True):
        """
        Unified enrollment entry point that serializes access and pauses auto identification.
        Args:
            client_id: Target client id
            finger_index: Optional finger index (1 or 2). If None, enroll both fingers.
            use_entrance_esp: Whether to use the entrance ESP32 (default True)
        """
        if client_id is None:
            print("❌ perform_enrollment called without a valid client_id")
            return False

        result = False
        with self.enrollment_lock:
            if self.enrollment_active.is_set():
                print(f"⏳ Enrollment already in progress; waiting to process client {client_id}")
            self.enrollment_active.set()
            try:
                if finger_index is None:
                    result = self.enroll_client_fingerprint(client_id, use_entrance_esp=use_entrance_esp)
                else:
                    pause_acquired = self._request_pause(f"fingerprint enrollment - finger {finger_index}")
                    was_entrance_running = self.entrance_running
                    was_locker_running = self.locker_running
                    if was_entrance_running:
                        self.stop_entrance_listener()
                    if was_locker_running:
                        self.stop_locker_listener()
                    try:
                        result = self.enroll_client_single_finger(
                            client_id,
                            finger_index,
                            use_entrance_esp=use_entrance_esp,
                            manage_pause=False
                        )
                    finally:
                        if was_entrance_running:
                            self.start_entrance_listener()
                        if was_locker_running:
                            self.start_locker_listener()
                        if pause_acquired:
                            self._release_pause()
            finally:
                self.enrollment_active.clear()
        return result
    
    def stop_entrance_listener(self):
        """Stop entrance listener"""
        self.entrance_running = False
        if self.entrance_thread:
            self.entrance_thread.join(timeout=2.0)
        print("🛑 Entrance listener stopped")
    
    def stop_locker_listener(self):
        """Stop locker listener"""
        self.locker_running = False
        if self.locker_thread:
            self.locker_thread.join(timeout=2.0)
        print("🛑 Locker listener stopped")
    
    def stop_auto_identification(self):
        """Stop both listeners"""
        self.stop_entrance_listener()
        self.stop_locker_listener()
    
    def reload_fingerprints(self):
        """Reload all registered fingerprints from database"""
        try:
            self.registered_fingerprints = self.db.get_all_fingerprint_features()
            total_samples = sum(len(p.get('fingerprints', [])) for p in self.registered_fingerprints)
            print(f"🔄 Loaded {len(self.registered_fingerprints)} fingerprint profiles ({total_samples} total samples)")
        except Exception as e:
            print(f"❌ Error reloading fingerprints: {e}")
            import traceback
            traceback.print_exc()
            self.registered_fingerprints = []
    
    def _entrance_listener_loop(self):
        """Background loop for entrance ESP32 (entry & enrollment)"""
        print(f"👂 Entrance ESP32 listening on port {UDP_PORT_ENTRANCE}...")
        
        while self.entrance_running:
            try:
                # Wait for fingerprint
                image = self.entrance_receiver.receive_image(timeout=5.0)
                
                if image is None:
                    continue
                
                print(f"\n🚪 ENTRANCE - Fingerprint received")
                
                # Extract and identify
                result = self._identify_fingerprint(image, "ENTRANCE", self.entrance_receiver.last_press)
                
                # Call entrance callback
                if self.entrance_callback:
                    try:
                        if result:
                            client_id, name, confidence = result
                            self.entrance_callback(client_id, name, confidence)
                        else:
                            self.entrance_callback(None, None, 0.0)
                    except Exception as e:
                        print(f"❌ Entrance callback error: {e}")
                
            except Exception as e:
                print(f"❌ Entrance listener error: {e}")
                time.sleep(1)
    
    def _locker_listener_loop(self):
        """Background loop for locker ESP32 (unlock only)"""
        print(f"👂 Locker ESP32 listening on port {UDP_PORT_LOCKER}...")
        
        while self.locker_running:
            try:
                # Wait for fingerprint
                image = self.locker_receiver.receive_image(timeout=5.0)
                
                if image is None:
                    continue
                
                print(f"\n🔐 LOCKER - Fingerprint received")
                
                # Extract and identify
                result = self._identify_fingerprint(image, "LOCKER", self.locker_receiver.last_press)
                
                # Call locker callback
                if self.locker_callback:
                    try:
                        if result:
                            client_id, name, confidence = result
                            self.locker_callback(client_id, name, confidence)
                        else:
                            self.locker_callback(None, None, 0.0)
                    except Exception as e:
                        print(f"❌ Locker callback error: {e}")
                
            except Exception as e:
                print(f"❌ Locker listener error: {e}")
                time.sleep(1)
    
    def _identify_fingerprint(self, image, source="UNKNOWN", press=None):
        """Identify a fingerprint image. press: (ESP32 IP, image timestamp), for the sensor double-check.

        NBIS (MINDTCT + BOZORTH3) is the primary matcher; the SIFT/FAISS engine
        is kept as a fallback for members whose enrollment predates the NBIS
        migration (FINGERPRINT_ENGINE / NBIS_FALLBACK_TO_SIFT control this).
        """
        # Refresh fingerprints from cache if available
        if self.fingerprint_cache:
            self.registered_fingerprints = self.fingerprint_cache.get_fingerprints()

        if not self.registered_fingerprints:
            print(f"⚠️ [{source}] No fingerprints enrolled")
            self._save_query_image(image, source, "nomatch")
            return None

        # --- grayscale engine (production) ---------------------------------
        if self.gray is not None:
            return self._identify_with_gray(image, source, press)

        # --- NBIS engine ---------------------------------------------------
        if self.nbis is not None:
            result = self._identify_with_nbis(image, source)
            if result:
                return result
            if FINGERPRINT_ENGINE in ("nbis", "local") or not NBIS_FALLBACK_TO_SIFT:
                self._save_query_image(image, source, "nomatch")
                return None
            print(f"↩️  [{source}] Falling back to the SIFT engine")
        elif FINGERPRINT_ENGINE in ("nbis", "local"):
            print(f"❌ [{source}] minutiae engine unavailable and fallback disabled")
            return None

        # --- SIFT/FAISS engine (original) ----------------------------------
        result = self._identify_with_sift(image, source)
        if not result:
            self._save_query_image(image, source, "nomatch")
        return result

    def _identify_with_gray(self, image, source="UNKNOWN", press=None):
        """1:N with the grayscale engine. Returns (client_id, name, confidence) or None."""
        if not fingerprint_sift.is_grayscale(image):
            print(f"❌ [{source}] the sensor sent a two-level skeleton image - flash the current "
                  f"finger_UDP.ino (it uploads the grayscale image)")
            return None
        t0 = time.time()
        query = fingerprint_sift.extract(image)
        if query is None or query.n < 10:
            print(f"❌ [{source}] no usable ridges in this press")
            self._save_query_image(image, source, "nomatch")
            return None
        try:
            self.gray.sync(self.registered_fingerprints)
            best, ranking = self.gray.identify(query)
        except Exception as e:  # noqa: BLE001 - a bad scan must not kill the listener
            print(f"❌ [{source}] grayscale identification error: {e}")
            return None
        elapsed = time.time() - t0
        threshold = fingerprint_sift.SIFT_MATCH_INLIERS
        if best:
            inl, client_id, finger, name = best
            runner_up = next((r[0] for r in ranking if r[1] != client_id), 0)
            note = ""
            if inl < SIFT_CONFIRM_BELOW:
                verdict = self._sensor_confirm(press, client_id, finger, source)
                if verdict is False:
                    print(f"❌ [{source}] NO MATCH - {name} scored {inl} matches (borderline) and "
                          f"the sensor did not confirm ({time.time() - t0:.2f}s)")
                    self._save_query_image(image, source, f"unconfirmed_{client_id}")
                    return None
                note = ", sensor confirmed" if verdict else ", not double-checked (no sensor template)"
            print(f"✅ [{source}] MATCH: {name} (ID: {client_id}, finger {finger}) - {inl} matches "
                  f"(next member {runner_up}, need {threshold}{note}, {time.time() - t0:.2f}s)")
            self._save_query_image(image, source, f"match_{client_id}")
            return (client_id, name, fingerprint_sift.confidence(inl))
        top = (f"best {ranking[0][3] or ranking[0][1]} with {ranking[0][0]} matches" if ranking
               else "no candidate")
        print(f"❌ [{source}] NO MATCH - {top} (need {threshold}, {elapsed:.2f}s)")
        self._save_query_image(image, source, "nomatch")
        return None

    def _sensor_confirm(self, press, client_id, finger, source):
        """Ask the R503's own matcher whether this press is that member's finger.

        True - the sensor matched it to one of the finger's stored templates.
        False - it did not, or the check could not run (the ESP32 already holds a
                newer press, does not answer, ...): the member presses again.
        None - the member has no stored sensor template (enrolled before the
               double-check existed, or the sensor was unreachable then); the
               match stands, unchecked.
        """
        templates = [s['sensor_template']
                     for p in self.registered_fingerprints if p.get('client_id') == client_id
                     for f in p.get('fingers', []) if f.get('finger_index') == finger
                     for s in f.get('samples', []) if s.get('sensor_template')]
        if not templates:
            print(f"   [{source}] member {client_id} finger {finger} has no sensor template - "
                  f"not double-checked (enroll the finger again to enable it)")
            return None
        if not R503_LINK_AVAILABLE or press is None:
            print(f"   [{source}] cannot reach the sensor for the double-check")
            return False
        ip, stamp = press
        if source == "ENTRANCE":
            self._status("Checking fingerprint...", console=False)
        t0 = time.time()
        try:
            with r503_link.SensorLink(ip, ENTRANCE_ESP_PORT) as link:
                ok, reply = link.prepare_confirm(stamp)
                if not ok:
                    print(f"   [{source}] sensor double-check unavailable ({reply}) - press again")
                    return False
                for k, template in enumerate(templates[:SENSOR_TEMPLATES_PER_FINGER], start=1):
                    res = link.match(template)
                    if res and res[0]:
                        print(f"   [{source}] sensor confirms member {client_id} "
                              f"(template {k}, score {res[1]}, {time.time() - t0:.1f}s)")
                        return True
                    print(f"   [{source}] sensor: template {k} "
                          f"{'no match (score %d)' % res[1] if res else 'no answer'}")
        except OSError as e:
            print(f"   [{source}] sensor double-check failed: {e}")
        return False

    def _identify_with_nbis(self, image, source="UNKNOWN"):
        """1:N minutiae matching. Returns (client_id, name, confidence) or None."""
        engine_name = getattr(self.nbis_module, "__name__", "minutiae")
        try:
            result = self.nbis_module.identify(
                self.nbis,
                image,
                self.registered_fingerprints,
                self.nbis_gallery,
            )
        except Exception as e:  # noqa: BLE001 - a bad scan must not kill the listener
            print(f"❌ [{source}] {engine_name} identification error: {e}")
            return None

        if result:
            client_id, name, score, details = result
            runner_up = details['ranking'][1][2] if len(details['ranking']) > 1 else 0
            print(f"✅ [{source}] MATCH: {name} (ID: {client_id}) - score {score} "
                  f"(runner-up {runner_up}, {details['compared']} templates in {details['elapsed']:.2f}s)")
            self._save_query_image(image, source, f"match_{client_id}")
            # Report a 0-1 confidence to callers that expect a percentage.
            confidence = min(1.0, score / max(self.nbis.threshold * 2.0, 1.0))
            return (client_id, name, confidence)

        print(f"❌ [{source}] NO MATCH ({engine_name}, threshold {self.nbis.threshold})")
        return None

    def _identify_with_sift(self, image, source="UNKNOWN"):
        """Original SIFT/ORB + FAISS matcher."""
        # Without FAISS there is no index to search, and identify_fingerprint_fast
        # would return None for everyone - scan the profiles directly instead.
        if not FAISS_AVAILABLE:
            return self._identify_sift_bruteforce(image, source)

        # Build/rebuild FAISS index if needed
        if self.processor.faiss_index is None:
            print(f"🔨 Building FAISS index for {len(self.registered_fingerprints)} clients...")
            self.processor.build_faiss_index(self.registered_fingerprints)

        # The FAISS pre-filter ranks on mean-pooled descriptors, which barely
        # separates fingerprints, so keep a wide candidate list - with only 5 the
        # right member is usually thrown away before verification.
        result = self.processor.identify_fingerprint_fast(
            image,
            self.registered_fingerprints,
            threshold=MATCH_THRESHOLD,
            top_k_candidates=50
        )

        if result:
            client_id, name, confidence = result
            print(f"✅ [{source}] MATCH (SIFT): {name} (ID: {client_id}) - {confidence*100:.1f}%")
            return (client_id, name, confidence)

        print(f"❌ [{source}] NO MATCH (SIFT, threshold {MATCH_THRESHOLD})")
        return None

    def _identify_sift_bruteforce(self, image, source="UNKNOWN"):
        """SIFT verification against every enrolled sample (no FAISS pre-filter)."""
        features = self.processor.extract_features(image)
        if features is None:
            print(f"✗ [{source}] Failed to extract features")
            return None

        best_match = None
        best_score = 0.0

        for person in self.registered_fingerprints:
            if 'fingers' in person:
                all_samples = []
                for finger in person['fingers']:
                    all_samples.extend(finger.get('samples', []))
            else:
                all_samples = person.get('fingerprints', [])

            scores = []
            for enrolled in all_samples:
                if not isinstance(enrolled, dict):
                    continue
                descriptors = enrolled.get('descriptors')
                if descriptors is None or len(descriptors) == 0:
                    continue
                scores.append(self.processor.match_fingerprints_detailed(features, enrolled))

            if not scores:
                continue
            scores.sort(reverse=True)
            score = float(np.mean(scores[:5]))
            if score > best_score:
                best_score = score
                best_match = (person['client_id'], person['name'])

        if best_match and best_score >= MATCH_THRESHOLD:
            client_id, name = best_match
            print(f"✅ [{source}] MATCH (SIFT brute force): {name} (ID: {client_id}) - {best_score*100:.1f}%")
            return (client_id, name, best_score)

        print(f"❌ [{source}] NO MATCH (SIFT brute force, best {best_score*100:.1f}%, "
              f"threshold {MATCH_THRESHOLD*100:.1f}%)")
        return None

    @staticmethod
    def _save_query_image(image, source, outcome):
        """Keep the query image when FINGERPRINT_SAVE_QUERY_IMAGES=1 (threshold tuning)."""
        if IMAGE_STORE_AVAILABLE:
            fingerprint_image_store.save_query_image(image, source, outcome)
    
    def enroll_client_fingerprint(self, client_id, use_entrance_esp=True):
        """
        Backward-compatible entry point: Enroll both fingers (1 and 2), 5 samples each
        
        Args:
            client_id: Client ID to enroll
            use_entrance_esp: If True, use entrance ESP32, else use locker ESP32
        
        Returns:
            True if successful
        """
        with self.enrollment_lock:
            owns_event = False
            if not self.enrollment_active.is_set():
                self.enrollment_active.set()
                owns_event = True
            else:
                print("ℹ️  Reusing existing enrollment session context")

            pause_acquired = self._request_pause("fingerprint enrollment")

            # Pause listeners during enrollment
            was_entrance_running = self.entrance_running
            was_locker_running = self.locker_running
            
            if was_entrance_running:
                self.stop_entrance_listener()
            if was_locker_running:
                self.stop_locker_listener()
            
            try:
                success = self.enroll_client_two_fingers(client_id, use_entrance_esp)
            finally:
                # Resume listeners
                if was_entrance_running:
                    self.start_entrance_listener()
                if was_locker_running:
                    self.start_locker_listener()
                if pause_acquired:
                    self._release_pause()
                if owns_event:
                    self.enrollment_active.clear()
            
            return success
    
    def enroll_client_single_finger(self, client_id, finger_index=1, use_entrance_esp=True, manage_pause=True):
        """
        Enroll 5 fingerprint samples for a specific finger (1 or 2)
        
        Args:
            client_id: Client ID to enroll
            finger_index: 1 or 2 (which finger)
            use_entrance_esp: If True, use entrance ESP32, else use locker ESP32
            manage_pause: When True, pause/resume the main pipeline within this call.
        
        Returns:
            True if successful
        """
        single_pause = False
        if manage_pause:
            single_pause = self._request_pause(f"fingerprint enrollment - finger {finger_index}")
        try:
            print(f"\n{'='*60}")
            print(f"🖐️ FINGERPRINT ENROLLMENT - Client ID: {client_id} - Finger {finger_index}")
            print(f"{'='*60}")
            self._status("FINGERPRINT ENROLLMENT", console=False)
            self._status(f"Client {client_id} - Finger {finger_index}", console=False)

            # Get client info
            client_info = self.db.get_client_info(client_id)
            if not client_info:
                print(f"❌ Client {client_id} not found")
                return False
            
            name = f"{client_info['fname']} {client_info['lname']}"
            print(f"👤 Enrolling: {name}")
            print(f"Using: {'ENTRANCE' if use_entrance_esp else 'LOCKER'} ESP32")
            print(f"Please scan {SAMPLES_PER_FINGER} fingerprint samples for finger {finger_index}...")
            self._status(f"Enrolling: {name}", console=False)
            self._status(f"Finger {finger_index}: {SAMPLES_PER_FINGER} presses", console=False)

            receiver = self.entrance_receiver if use_entrance_esp else self.locker_receiver

            esp_ip = ENTRANCE_ESP_IP if use_entrance_esp else LOCKER_ESP_IP
            esp_port = ENTRANCE_ESP_PORT if use_entrance_esp else LOCKER_ESP_PORT

            if self.gray is not None:
                return self._enroll_finger_gray(client_id, finger_index, receiver, esp_ip, esp_port)

            samples = []
            retries_left = ENROLL_RETRY_BUDGET
            # Whole-finger template stitched so far (pure-Python engine only: the
            # stitching reads its .xyt layout). Shown on the LCD as it grows.
            fusing = self.nbis_module is fingerprint_local if LOCAL_MODULE_AVAILABLE else False
            whole = None
            while len(samples) < SAMPLES_PER_FINGER:
                i = len(samples)
                prompt = ENROLL_PROMPTS[i % len(ENROLL_PROMPTS)]
                print(f"\n📌 Finger {finger_index} - Press {i+1}/{SAMPLES_PER_FINGER}: {prompt}")
                # The prompt gets its own line: together they do not fit the LCD
                self._status(f"Finger {finger_index} - press {i+1}/{SAMPLES_PER_FINGER}", console=False)
                self._status(prompt, console=False)

                # Send status to ESP32
                if self.sock:
                    try:
                        sendstr = f"Finger {finger_index} - Sample {i+1}"
                        self.sock.sendto(sendstr.encode("utf-8"), (esp_ip, esp_port))
                        self.sock.sendto(sendstr.encode("utf-8"), (pc_ip, pc_port))
                    except Exception:
                        pass
                
                # Receive image
                image = receiver.receive_image(timeout=60.0)
                if image is None:
                    print(f"✗ Failed to receive sample {i+1}")
                    return False
                
                # Save the raw image and run both engines on it
                print("🔍 Extracting features...")
                features = build_sample_record(
                    image, self.processor,
                    client_id=client_id, finger_index=finger_index, sample_number=i + 1,
                    nbis_engine=self.nbis
                )
                preview_title = (f"Client {client_id} - Finger {finger_index} - "
                                 f"Press {i+1}/{SAMPLES_PER_FINGER}: {prompt}")
                whole_xyt = fingerprint_local.minutiae_to_xyt(whole) if whole else None

                if features is None:
                    print(f"✗ Failed to extract features from sample {i+1}")
                    publish_fingerprint_preview(image, preview_title, None,
                                                "REJECTED: no usable ridges - press again", ok=False,
                                                whole_xyt=whole_xyt)
                    self._status("Rejected: no usable ridges", console=False)
                    self._status("Clean the finger, press again", console=False)
                    retries_left -= 1
                    if retries_left < 0:
                        print("✗ Too many unusable presses - aborting this finger")
                        return False
                    continue

                # Quality gate: an impression this weak cannot be matched later,
                # so spend a retry on it rather than storing it forever.
                minutiae = features.get('minutiae_count')
                if minutiae is not None and minutiae < MIN_ENROLL_MINUTIAE:
                    retries_left -= 1
                    warn = (f"Rejected: only {minutiae} minutiae (need "
                            f"{MIN_ENROLL_MINUTIAE}) - press firmer and cover the whole window")
                    print(warn)
                    publish_fingerprint_preview(image, preview_title, features.get('xyt'),
                                                f"REJECTED: {minutiae} minutiae (need {MIN_ENROLL_MINUTIAE})",
                                                ok=False, whole_xyt=whole_xyt)
                    self._status(f"Rejected: {minutiae} minutiae (need {MIN_ENROLL_MINUTIAE})", console=False)
                    self._status("Press firmer, cover the whole sensor", console=False)
                    if retries_left < 0:
                        print("✗ Could not get enough good presses - clean the sensor and retry")
                        return False
                    continue

                samples.append(features)
                quality = f", {minutiae} minutiae" if minutiae else ""
                print(f"✓ Sample {i+1} captured ({features['num_features']} features{quality})")

                # Stitch this press onto the whole-finger template
                stitch, note = "", None
                if fusing and features.get('xyt'):
                    press = fingerprint_local.parse_xyt(features['xyt'])
                    if whole is None:
                        whole, note = press, "first press"
                    else:
                        fused, info = fingerprint_local.fuse_minutiae(whole, press)
                        if fused is not None:
                            whole, note = fused, f"+{info['added']} new minutiae"
                        else:
                            note = "not stitched - move less next time"
                    stitch = f", {note}, whole finger {len(whole)}"
                    print(f"   🧩 Whole-finger template: {len(whole)} minutiae ({note})")
                publish_fingerprint_preview(image, preview_title, features.get('xyt'),
                                            (f"OK: {minutiae} minutiae" if minutiae
                                             else "OK (no minutiae template, SIFT only)") + stitch,
                                            whole_xyt=fingerprint_local.minutiae_to_xyt(whole) if whole else None)
                # One short line each, so the LCD shows all of it
                self._status(f"Press {i+1} OK: {minutiae or 0} minutiae", console=False)
                if note:
                    self._status(note[0].upper() + note[1:], console=False)
                if whole:
                    self._status(f"Whole finger: {len(whole)} minutiae", console=False)

                if i < SAMPLES_PER_FINGER - 1:
                    next_prompt = ENROLL_PROMPTS[(i + 1) % len(ENROLL_PROMPTS)]
                    print(f"   Lift the finger. Next: {next_prompt}")
                    self._status("Lift the finger", console=False)
                    self._status(f"Next: {next_prompt}", console=False)
                    time.sleep(1)
            
            if len(samples) < SAMPLES_PER_FINGER:
                print("✗ Not enough valid samples for this finger")
                return False
            
            # Save to database with finger_index (the whole-finger template below
            # is stitched from ALL presses even if only the strongest are stored)
            success, stored = self._save_finger(client_id, finger_index, samples,
                                                strength=lambda s: s.get('minutiae_count') or 0)

            if success and fusing:
                # Re-stitch all presses in the best order (a press that did not fit
                # early can fit once its neighbours are in) and store the result
                # as one more template for this finger.
                fused, used = fingerprint_local.fuse_presses(
                    [fingerprint_local.parse_xyt(s['xyt']) for s in samples if s.get('xyt')])
                if fused is not None:
                    if self.db.save_fused_template(client_id, finger_index,
                                                   fingerprint_local.minutiae_to_xyt(fused), len(fused)):
                        print(f"🧩 Whole finger: {len(fused)} minutiae from {len(used)} presses")
                        self._status(f"Whole finger: {len(fused)} minutiae from {len(used)} presses")
                    else:
                        # The finger is still enrolled with its separate presses
                        self._status(f"Whole-finger template not saved: {self._db_error()}")
                else:
                    print("⚠️  Presses did not overlap enough to build a whole-finger template "
                          "- the finger is still enrolled with its separate presses")

            if success:
                self._after_finger_saved(client_id, finger_index, stored, esp_ip, esp_port)

            return success
        finally:
            if manage_pause and single_pause:
                self._release_pause()

    def _after_finger_saved(self, client_id, finger_index, stored, esp_ip, esp_port):
        """Make every matcher and cache pick up a newly saved finger, and tell the ESP32."""
        # Both matchers cache enrolled data - make them pick up the new finger
        nbis_invalidate_gallery()
        self.processor.faiss_index = None
        self.processor.cache_valid = False
        print(f"✅ Saved {stored} samples for finger {finger_index}")
        self._status(f"Saved {stored} samples for finger {finger_index}", console=False)
        # Send success to ESP32
        if self.sock:
            try:
                sendstr = f"Success Finger {finger_index}"
                self.sock.sendto(sendstr.encode("utf-8"), (esp_ip, esp_port))
            except Exception:
                pass

        # Update cache if available
        if self.fingerprint_cache:
            success_update = self.fingerprint_cache.update_client_fingerprints(client_id)
            if success_update:
                # Update registered fingerprints from cache
                self.registered_fingerprints = self.fingerprint_cache.get_fingerprints()
                print(f"✅ Fingerprint cache updated in fingerprint system")
                self._status("Fingerprint cache updated", console=False)

        # Update local cache without reloading all clients when cache is not available
        if not self.fingerprint_cache:
            profile = self.db.get_fingerprint_profile(client_id)
            if profile:
                self.registered_fingerprints = [
                    p for p in self.registered_fingerprints if p.get('client_id') != client_id
                ]
                self.registered_fingerprints.append(profile)
            else:
                # Fallback to full reload if targeted refresh failed
                self.reload_fingerprints()

        if self.gray is not None:
            self.gray.sync(self.registered_fingerprints)

    def _save_finger(self, client_id, finger_index, samples, strength):
        """Save a finger's presses. Returns (ok, presses stored).

        The members database may allow only 5 samples per finger (a sample_number
        limit from the admin app - see migrate_fingerprint_presses.sql). If it
        refuses them all, keep the strongest FALLBACK_STORED_PRESSES so the member
        is enrolled now.
        """
        success = self.db.save_fingerprint_features(client_id, finger_index, samples)
        stored = len(samples)
        if not success:
            self._status(f"Save failed: {self._db_error()}")
            if len(samples) > FALLBACK_STORED_PRESSES:
                best = sorted(samples, key=strength, reverse=True)[:FALLBACK_STORED_PRESSES]
                success = self.db.save_fingerprint_features(client_id, finger_index, best)
                stored = len(best)
                if success:
                    self._status(f"Saved the best {FALLBACK_STORED_PRESSES} presses - run "
                                 f"migrate_fingerprint_presses.sql to keep all {len(samples)}")
                else:
                    self._status(f"Save failed again: {self._db_error()}")
        return success, stored

    def _sensor_templates(self, samples, ip):
        """Store the R503's own template on the strongest presses, for the sensor double-check.

        Each one is built by the sensor from the press's image (sent back to it),
        ~3.5 s apiece. A failure only means this finger is not double-checked."""
        if not (R503_LINK_AVAILABLE and SIFT_CONFIRM_BELOW > 0 and SENSOR_TEMPLATES_PER_FINGER > 0 and ip):
            return 0
        columns = getattr(self.db, 'get_fingerprint_columns', None)
        if columns is not None and 'sensor_template' not in columns():
            print("⚠️  No sensor_template column - run migrate_fingerprint_sift_gray.sql to enable "
                  "the sensor double-check")
            return 0
        best = sorted(samples, key=lambda s: s.get('num_features') or 0, reverse=True)[:SENSOR_TEMPLATES_PER_FINGER]
        self._status("Preparing the sensor check...", console=False)
        made = 0
        try:
            with r503_link.SensorLink(ip, ENTRANCE_ESP_PORT) as link:
                for s in best:
                    t = link.image_template(s['image'])
                    if t and len(t) >= 256 and any(t):
                        s['sensor_template'] = t
                        made += 1
        except OSError as e:
            print(f"⚠️  Sensor templates failed: {e}")
        print(f"   Sensor templates: {made}/{len(best)} presses")
        if made == 0:
            self._status("Sensor check not available for this finger", console=False)
        return made

    def _enroll_finger_gray(self, client_id, finger_index, receiver, esp_ip, esp_port):
        """Enroll one finger with the grayscale engine.

        Each press must be grayscale, carry enough keypoints, and match an earlier
        press of this finger; the finished finger must not match any OTHER member
        (a double enrollment, or two members the door would confuse).
        """
        threshold = fingerprint_sift.SIFT_MATCH_INLIERS
        samples = []
        retries_left = ENROLL_RETRY_BUDGET
        while len(samples) < SAMPLES_PER_FINGER:
            i = len(samples)
            prompt = ENROLL_PROMPTS[i % len(ENROLL_PROMPTS)]
            print(f"\n📌 Finger {finger_index} - Press {i+1}/{SAMPLES_PER_FINGER}: {prompt}")
            self._status(f"Finger {finger_index} - press {i+1}/{SAMPLES_PER_FINGER}", console=False)
            self._status(prompt, console=False)
            if self.sock:
                try:
                    sendstr = f"Finger {finger_index} - Sample {i+1}"
                    self.sock.sendto(sendstr.encode("utf-8"), (esp_ip, esp_port))
                    self.sock.sendto(sendstr.encode("utf-8"), (pc_ip, pc_port))
                except Exception:
                    pass

            image = receiver.receive_image(timeout=60.0)
            if image is None:
                print(f"✗ Failed to receive sample {i+1}")
                self._status("No press received - enrollment stopped")
                return False
            sender = receiver.last_press[0] if receiver.last_press else None
            title = f"Client {client_id} - Finger {finger_index} - Press {i+1}/{SAMPLES_PER_FINGER}: {prompt}"

            if not fingerprint_sift.is_grayscale(image):
                self._status("Sensor sends the old black-and-white image - flash finger_UDP.ino")
                publish_fingerprint_preview(image, title, None, "REJECTED: not a grayscale image", ok=False)
                return False

            record = build_sample_record(image, self.processor, client_id=client_id,
                                         finger_index=finger_index, sample_number=i + 1)
            f = record.get('gray') if record else None
            problem = None
            if f is None:
                problem = "no usable ridges - clean the finger"
            elif f.n < fingerprint_sift.SIFT_MIN_KEYPOINTS:
                problem = f"only {f.n} features (need {fingerprint_sift.SIFT_MIN_KEYPOINTS}) - press firmer"
            elif samples:
                own = max(fingerprint_sift.inliers(f, s['gray']) for s in samples)
                if own < threshold:
                    problem = (f"does not match the earlier presses ({own} matches, need {threshold}) "
                               f"- same finger, move less")
            if problem:
                retries_left -= 1
                print(f"✗ Rejected: {problem}")
                publish_fingerprint_preview(image, title, _keypoints_xyt(f), f"REJECTED: {problem}", ok=False)
                self._status(f"Rejected: {problem}", console=False)
                if retries_left < 0:
                    self._status("Too many rejected presses - enrollment stopped")
                    return False
                continue

            record['image'] = image                  # for the sensor template below; not stored
            samples.append(record)
            print(f"✓ Press {i+1} OK ({f.n} features)")
            publish_fingerprint_preview(image, title, _keypoints_xyt(f), f"OK: {f.n} features")
            self._status(f"Press {i+1} OK: {f.n} features", console=False)
            if i < SAMPLES_PER_FINGER - 1:
                next_prompt = ENROLL_PROMPTS[(i + 1) % len(ENROLL_PROMPTS)]
                print(f"   Lift the finger. Next: {next_prompt}")
                self._status("Lift the finger", console=False)
                self._status(f"Next: {next_prompt}", console=False)
                time.sleep(1)

        # The new finger against every other member
        self._status("Checking against other members...", console=False)
        self.gray.sync(self.registered_fingerprints)
        closest = (0, None, None, "")
        for s in samples:
            _, ranking = self.gray.identify(s['gray'], exclude_client=client_id)
            if ranking and ranking[0][0] > closest[0]:
                closest = ranking[0]
        if closest[0] >= threshold:
            msg = (f"This finger matches member {closest[1]} ({closest[3]}), finger {closest[2]}: "
                   f"{closest[0]} matches")
            print(f"❌ {msg}")
            self._status(msg, console=False)
            if SIFT_ENROLL_COLLISION == "reject":
                self._status("Not saved - check for a double enrollment", console=False)
                return False
        else:
            who = f"member {closest[1]}" if closest[1] is not None else "none"
            print(f"   Closest other member: {who} with {closest[0]} matches (threshold {threshold})")

        self._sensor_templates(samples, sender)

        success, stored = self._save_finger(client_id, finger_index, samples,
                                            strength=lambda s: s.get('num_features') or 0)
        if success:
            self._after_finger_saved(client_id, finger_index, stored, esp_ip, esp_port)
        return success
    
    def enroll_client_two_fingers(self, client_id, use_entrance_esp=True):
        """
        Enroll both fingers (1 and 2), 5 samples each
        
        Args:
            client_id: Client ID to enroll
            use_entrance_esp: If True, use entrance ESP32, else use locker ESP32
        
        Returns:
            True if both fingers enrolled successfully
        """
        all_ok = True
        for finger_index in range(1, FINGERS_PER_CLIENT + 1):
            ok = self.enroll_client_single_finger(client_id, finger_index, use_entrance_esp)
            all_ok = all_ok and ok
            if not ok:
                print(f"⚠️  Finger {finger_index} enrollment failed")
                send_str = f"Finger {finger_index} enrollment failed"
                self.sock.sendto(send_str.encode("utf-8"), (pc_ip, pc_port))
                self.sock.sendto(send_str.encode("utf-8"), (raspi_entrance_ip, raspi_entrance_port))
        return all_ok
    
    def close(self):
        """Cleanup resources"""
        self.stop_auto_identification()
        self.entrance_receiver.close()
        self.locker_receiver.close()


class FingerprintGymSystem:
    """Main fingerprint system for gym access control (Single ESP32 - Legacy)"""
    
    def __init__(self, db_helper, auto_identify=False, identification_callback=None, sock=None, port=UDP_PORT_ENTRANCE):
        self.db = db_helper
        self.receiver = UDPFingerprintReceiver(port)
        self.processor = ImprovedFingerprintProcessor(embedding_dim=128, use_pca=True)
        self.nbis_module, self.nbis, self.nbis_gallery = get_minutiae_engine()
        self.fingerprint_cache = None  # set via set_fingerprint_cache(); used by the auto loop
        self.sock = sock  # Socket for sending messages to ESP32
        
        # Load registered fingerprints
        self.registered_fingerprints = []
        self.reload_fingerprints()
        
        # Auto-identification setup
        self.auto_identify_enabled = auto_identify
        self.identification_callback = identification_callback
        self.auto_thread = None
        self.auto_running = False
        
        if auto_identify:
            self.start_auto_identification()
        
        print("✅ Fingerprint gym system initialized")
    
    def set_fingerprint_cache(self, fingerprint_cache):
        """Set or update the shared fingerprint cache reference"""
        self.fingerprint_cache = fingerprint_cache
        if fingerprint_cache:
            self.registered_fingerprints = fingerprint_cache.get_fingerprints()

    def reload_fingerprints(self):
        """Reload all registered fingerprints from database"""
        try:
            self.registered_fingerprints = self.db.get_all_fingerprint_features()
            print(f"🔄 Loaded {len(self.registered_fingerprints)} fingerprint profiles")
        except Exception as e:
            print(f"❌ Error reloading fingerprints: {e}")
            self.registered_fingerprints = []
    
    def enroll_client_fingerprint(self, client_id):
        """
        Enroll 5 fingerprint samples for a client
        
        Returns:
            True if successful, False otherwise
        """
        print(f"\n{'='*60}")
        print(f"🖐️ FINGERPRINT ENROLLMENT - Client ID: {client_id}")
        print(f"{'='*60}")
        send_str = f"FINGERPRINT ENROLLMENT - Client ID: {client_id}"
        self.sock.sendto(send_str.encode("utf-8"), (pc_ip, pc_port))
        self.sock.sendto(send_str.encode("utf-8"), (raspi_entrance_ip, raspi_entrance_port))
        
        # Get client info
        client_info = self.db.get_client_info(client_id)
        if not client_info:
            print(f"❌ Client {client_id} not found")
            return False
        
        name = f"{client_info['fname']} {client_info['lname']}"
        print(f"👤 Enrolling: {name}")
        print(f"⚠️  Legacy class - Please use DualESP32FingerprintSystem for 2-finger enrollment")
        print(f"Scanning {SAMPLES_PER_FINGER} samples for finger 1 only...")
        
        fingerprints = []
        
        for i in range(SAMPLES_PER_FINGER):
            print(f"\n📌 Finger 1 - Sample {i+1}/{SAMPLES_PER_FINGER}")
            if self.sock:
                try:
                    sendstr = f"Sample {i+1}"
                    self.sock.sendto(sendstr.encode("utf-8"), (raspi_entrance_ip, raspi_entrance_port))
                except Exception as e:
                    print(f"⚠️  Could not send status to ESP32: {e}")
            # Receive image
            image = self.receiver.receive_image(timeout=60.0)
            if image is None:
                print(f"✗ Failed to receive sample {i+1}")
                if i >= 2:  # At least 3 samples required
                    print("Using partial enrollment data...")
                    break
                else:
                    return False
            
            # Save the raw image and run both engines on it
            print("🔍 Extracting features...")
            features = build_sample_record(
                image, self.processor,
                client_id=client_id, finger_index=1, sample_number=i + 1,
                nbis_engine=self.nbis
            )
            
            if features is None:
                print(f"✗ Failed to extract features from sample {i+1}")
                continue
            
            fingerprints.append(features)
            print(f"✓ Sample {i+1} captured ({features['num_features']} features)")
            
            if i < SAMPLES_PER_FINGER - 1:
                print("   Remove and place finger again...")
                time.sleep(2)
        
        if len(fingerprints) < SAMPLES_PER_FINGER:
            print("✗ Not enough valid samples for enrollment")
            return False
        
        # Save to database (legacy: only finger 1)
        success = self.db.save_fingerprint_features(client_id, 1, fingerprints)
        
        if success:
            print(f"\n{'='*60}")
            print(f"✅ FINGERPRINT ENROLLMENT COMPLETE!")
            print(f"  Client ID: {client_id}")
            print(f"  Name: {name}")
            print(f"  Samples: {len(fingerprints)}")
            print(f"{'='*60}")
            
            if self.sock:
                try:
                    sendstr = f"Success {client_id}"
                    self.sock.sendto(sendstr.encode("utf-8"), (raspi_entrance_ip, raspi_entrance_port))
                except Exception as e:
                    print(f"⚠️  Could not send success to ESP32: {e}")
            
            # Refresh local cache for this client only
            profile = self.db.get_fingerprint_profile(client_id)
            if profile:
                self.registered_fingerprints = [
                    p for p in self.registered_fingerprints if p.get('client_id') != client_id
                ]
                self.registered_fingerprints.append(profile)
            else:
                # Fallback to full reload if targeted refresh failed
                self.reload_fingerprints()
            return True
        else:
            print(f"❌ Failed to save fingerprints to database")
            return False
    
    def identify_fingerprint_for_access(self):
        """
        Identify a fingerprint for gym access
        
        Returns:
            Tuple of (client_id, client_name, confidence) if match found
            None if no match
        """
        print(f"\n{'='*60}")
        print("🔍 FINGERPRINT IDENTIFICATION FOR GYM ACCESS")
        print(f"{'='*60}")
        
        if not self.registered_fingerprints:
            print("✗ No fingerprints enrolled in database")
            return None
        
        print(f"Database: {len(self.registered_fingerprints)} clients enrolled")
        print("Please scan fingerprint...")
        
        # Receive image
        image = self.receiver.receive_image(timeout=60.0)
        if image is None:
            print("✗ Failed to receive fingerprint")
            return None
        
        # Extract features
        print("🔍 Extracting features...")
        features = self.processor.extract_features(image)
        
        if features is None:
            print("✗ Failed to extract features")
            return None
        
        print(f"✓ Extracted {features['num_features']} features")
        
        # Match against database
        print("\n🔎 Searching database...")
        best_match = None
        best_score = 0.0
        
        for person in self.registered_fingerprints:
            client_id = person['client_id']
            name = person['name']
            
            # Get all fingerprints (could be 'fingerprints' for backward compat or 'fingers' for new structure)
            if 'fingers' in person:
                # New structure: grouped by fingers
                all_samples = []
                for finger in person['fingers']:
                    all_samples.extend(finger.get('samples', []))
            else:
                # Old structure: flat list
                all_samples = person.get('fingerprints', [])
            
            print(f"  Checking {name}...", end=' ')
            
            # Compare with all samples from all fingers
            scores = []
            for enrolled_fp in all_samples:
                try:
                    if not isinstance(enrolled_fp, dict):
                        continue
                    if 'descriptors' not in enrolled_fp:
                        continue
                    desc = enrolled_fp.get('descriptors')
                    if desc is None or (hasattr(desc, '__len__') and len(desc) == 0):
                        continue
                    score = self.processor.match_fingerprints(features, enrolled_fp)
                    scores.append(score)
                except Exception as e:
                    # Skip samples that cause errors
                    continue
            
            if not scores:
                continue
                
            # Use average of top 5 scores (we have 5 samples per finger, 2 fingers = 10 total)
            scores.sort(reverse=True)
            avg_score = np.mean(scores[:5]) if len(scores) >= 5 else np.mean(scores)
            
            print(f"scores: {[f'{s:.3f}' for s in scores[:5]]} → avg: {avg_score:.3f} ({avg_score*100:.1f}%)")
            
            if avg_score > best_score:
                best_score = avg_score
                best_match = (client_id, name)
        
        # Display result
        print(f"\n{'='*60}")
        if best_score >= MATCH_THRESHOLD:
            print("✅ FINGERPRINT MATCH!")
            print(f"  Client ID: {best_match[0]}")
            print(f"  Name: {best_match[1]}")
            print(f"  Confidence: {best_score*100:.1f}%")
            print(f"{'='*60}")
            
            return (best_match[0], best_match[1], best_score)
        else:
            print("❌ NO MATCH FOUND")
            print(f"  Best score: {best_score*100:.1f}%")
            print(f"  Threshold: {MATCH_THRESHOLD*100:.1f}%")
            print("  Fingerprint not in database")
            print(f"{'='*60}")
            return None
    
    def start_auto_identification(self):
        """Start automatic fingerprint identification in background thread"""
        if self.auto_running:
            print("⚠️  Auto-identification already running")
            return
        
        self.auto_running = True
        self.auto_thread = threading.Thread(target=self._auto_identification_loop, daemon=True)
        self.auto_thread.start()
        print("🔄 Auto-identification started - listening for fingerprints...")
    
    def stop_auto_identification(self):
        """Stop automatic fingerprint identification"""
        self.auto_running = False
        if self.auto_thread:
            self.auto_thread.join(timeout=2.0)
        print("🛑 Auto-identification stopped")
    
    def _auto_identification_loop(self):
        """Background loop that automatically identifies fingerprints when received"""
        print("👂 Listening for fingerprints on UDP...")
        
        while self.auto_running:
            try:
                # Refresh fingerprints from cache if available (once per loop iteration)
                if self.fingerprint_cache:
                    self.registered_fingerprints = self.fingerprint_cache.get_fingerprints()
                
                # Wait for fingerprint image (non-blocking with timeout)
                image = self.receiver.receive_image(timeout=5.0)
                
                if image is None:
                    # Timeout - no image received, continue listening
                    continue
                
                # Image received! Run identification
                print("\n🖐️ Fingerprint received - Running automatic identification...")
                
                # Extract features
                features = self.processor.extract_features(image)
                
                if features is None:
                    print("✗ Failed to extract features - waiting for next scan...")
                    continue
                
                print(f"✓ Extracted {features['num_features']} features")
                
                # Match against database
                best_match = None
                best_score = 0.0
                
                for person in self.registered_fingerprints:
                    client_id = person['client_id']
                    name = person['name']
                    
                    # Get all fingerprints (could be 'fingerprints' for backward compat or 'fingers' for new structure)
                    if 'fingers' in person:
                        # New structure: grouped by fingers
                        all_samples = []
                        for finger in person['fingers']:
                            all_samples.extend(finger.get('samples', []))
                    else:
                        # Old structure: flat list
                        all_samples = person.get('fingerprints', [])
                    
                    # Compare with all samples from all fingers
                    scores = []
                    for enrolled_fp in all_samples:
                        if not isinstance(enrolled_fp, dict) or 'descriptors' not in enrolled_fp or enrolled_fp.get('descriptors') is None:
                            continue
                        score = self.processor.match_fingerprints(features, enrolled_fp)
                        scores.append(score)
                    
                    # Use average of top 5 scores (we have 5 samples per finger, 2 fingers = 10 total)
                    scores.sort(reverse=True)
                    avg_score = np.mean(scores[:5]) if len(scores) >= 5 else np.mean(scores) if scores else 0.0
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_match = (client_id, name)
                
                # Process result
                if best_score >= MATCH_THRESHOLD:
                    client_id, name = best_match
                    print(f"✅ MATCH: {name} (ID: {client_id}) - Confidence: {best_score*100:.1f}%")
                    
                    # Call callback if provided
                    if self.identification_callback:
                        try:
                            self.identification_callback(client_id, name, best_score)
                        except Exception as e:
                            print(f"❌ Callback error: {e}")
                else:
                    print(f"❌ NO MATCH - Best score: {best_score*100:.1f}%")
                    
                    # Call callback with None to indicate no match
                    if self.identification_callback:
                        try:
                            self.identification_callback(None, None, best_score)
                        except Exception as e:
                            print(f"❌ Callback error: {e}")
                
            except Exception as e:
                print(f"❌ Auto-identification error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)  # Prevent rapid error loops
    
    def close(self):
        """Cleanup resources"""
        self.stop_auto_identification()
        self.receiver.close()


# Standalone functions for use in RetinaFaceL.py

def enroll_fingerprint_standalone(db_helper, client_id):
    """
    Standalone function for fingerprint enrollment
    Can be called from RetinaFaceL.py
    """
    system = FingerprintGymSystem(db_helper)
    result = system.enroll_client_fingerprint(client_id)
    system.close()
    return result


def identify_fingerprint_standalone(db_helper):
    """
    Standalone function for fingerprint identification
    Can be called from RetinaFaceL.py
    
    Returns:
        Tuple of (client_id, client_name, confidence) or None
    """
    system = FingerprintGymSystem(db_helper)
    result = system.identify_fingerprint_for_access()
    system.close()
    return result


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════╗
║     Fingerprint Gym System - Test Mode                     ║
╚════════════════════════════════════════════════════════════╝
    """)
    print("This module should be imported into RetinaFaceL.py")
    print("For standalone testing, use fingerprint_receiver.py")

