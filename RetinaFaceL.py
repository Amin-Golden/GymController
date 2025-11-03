# python RetinaFaceL.py --model_path model/retinaface.rknn --db_host 10.183.120.162
import os
import sys
import urllib
import urllib.request
import time
import numpy as np
import argparse
import cv2
from math import ceil
from itertools import product as product
from scipy.spatial.distance import cosine , euclidean
import math
# from rknn.api import RKNN
from rknnlite.api import RKNNLite
from db_helper import DatabaseHelper
import socket
import threading
from queue import Queue
import ctypes
from ctypes import *
import traceback
from functools import lru_cache
from fingerprint_gym import FingerprintGymSystem, enroll_fingerprint_standalone, identify_fingerprint_standalone
from fingerprint_gym import DualESP32FingerprintSystem

HEADLESS = os.environ.get('DISPLAY') is None

def check_db_pool_health():
    """Monitor database connection pool status"""
    try:
        # Get pool statistics
        pool = db.connection_pool._pool
        available = len([c for c in pool if c])
        in_use = len(pool) - available
        
        if in_use > 40:  # 80% capacity
            print(f"⚠️ Database pool high: {in_use}/50 connections in use")
        
        if in_use >= 48:  # 96% capacity
            print(f"🚨 DATABASE POOL CRITICAL: {in_use}/50 connections!")
            # Force connection cleanup
            db.connection_pool.closeall()
            # Recreate pool
            db.__init__(db.host, db.database, db.user, db.password, db.port)
    except Exception as e:
        print(f"Error checking pool: {e}")

# Add caching with expiration
class TimedCache:
    def __init__(self, max_age=30):
        self.cache = {}
        self.timestamps = {}
        self.max_age = max_age
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                age = time.time() - self.timestamps[key]
                if age < self.max_age:
                    return self.cache[key]
                else:
                    # Expired, remove
                    del self.cache[key]
                    del self.timestamps[key]
            return None
    
    def set(self, key, value):
        with self.lock:
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()

# Create caches
client_info_cache = TimedCache(max_age=30)  # Cache for 30 seconds
membership_cache = TimedCache(max_age=30)

# Global in-memory caches for face embeddings and fingerprints
registered_faces_cache = []  # Loaded at startup, updated on notifications
registered_fingerprints_cache = []  # Loaded at startup, updated on notifications
cache_lock = threading.Lock()  # Thread-safe access to caches

class BiometricCache:
    """Manages in-memory cache for face embeddings and fingerprint features"""
    
    def __init__(self, db_helper):
        self.db = db_helper
        self.faces = []
        self.fingerprints = []
        self.lock = threading.Lock()
    
    def load_all(self):
        """Load all biometric data from database at startup"""
        print("📥 Loading biometric data from database...")
        
        # Load face embeddings
        try:
            embedding_shape = (512,)
            self.faces = self.db.get_all_face_embeddings(embedding_shape)
            print(f"✅ Loaded {len(self.faces)} face embeddings")
        except Exception as e:
            print(f"❌ Error loading face embeddings: {e}")
            self.faces = []
        
        # Load fingerprints
        try:
            self.fingerprints = self.db.get_all_fingerprint_features()
            print(f"✅ Loaded {len(self.fingerprints)} fingerprint profiles")
        except Exception as e:
            print(f"❌ Error loading fingerprints: {e}")
            self.fingerprints = []
    
    def reload_faces(self):
        """Reload face embeddings from database"""
        with self.lock:
            try:
                embedding_shape = (512,)
                self.faces = self.db.get_all_face_embeddings(embedding_shape)
                print(f"🔄 Reloaded {len(self.faces)} face embeddings")
            except Exception as e:
                print(f"❌ Error reloading faces: {e}")
    
    def reload_fingerprints(self):
        """Reload fingerprints from database"""
        with self.lock:
            try:
                self.fingerprints = self.db.get_all_fingerprint_features()
                print(f"🔄 Reloaded {len(self.fingerprints)} fingerprint profiles")
            except Exception as e:
                print(f"❌ Error reloading fingerprints: {e}")
    
    def update_client_face(self, client_id):
        """Update face embedding for a specific client"""
        with self.lock:
            try:
                embedding_shape = (512,)
                embedding = self.db.get_face_embedding(client_id, embedding_shape)
                
                if embedding is None:
                    # Remove if exists
                    self.faces = [f for f in self.faces if f['client_id'] != client_id]
                    return False
                
                # Update or add
                client_info = self.db.get_client_info(client_id)
                if not client_info:
                    return False
                
                name = f"{client_info['fname']} {client_info['lname']}"
                
                # Remove existing
                self.faces = [f for f in self.faces if f['client_id'] != client_id]
                
                # Add new
                self.faces.append({
                    'client_id': client_id,
                    'embedding': embedding,
                    'name': name
                })
                print(f"✅ Updated face cache for client {client_id} ({name})")
                return True
            except Exception as e:
                print(f"❌ Error updating face cache: {e}")
                return False
    
    def update_client_fingerprints(self, client_id):
        """Update fingerprints for a specific client"""
        with self.lock:
            try:
                fingerprints = self.db.get_fingerprint_features(client_id)
                
                if not fingerprints:
                    # Remove if exists
                    self.fingerprints = [f for f in self.fingerprints if f['client_id'] != client_id]
                    return False
                
                # Get client info
                client_info = self.db.get_client_info(client_id)
                if not client_info:
                    return False
                
                name = f"{client_info['fname']} {client_info['lname']}"
                
                # Remove existing
                self.fingerprints = [f for f in self.fingerprints if f['client_id'] != client_id]
                
                # Add new
                self.fingerprints.append({
                    'client_id': client_id,
                    'name': name,
                    'fingerprints': fingerprints
                })
                print(f"✅ Updated fingerprint cache for client {client_id} ({name})")
                return True
            except Exception as e:
                print(f"❌ Error updating fingerprint cache: {e}")
                return False
    
    def remove_client(self, client_id):
        """Remove all biometric data for a client"""
        with self.lock:
            before_faces = len(self.faces)
            before_prints = len(self.fingerprints)
            
            self.faces = [f for f in self.faces if f['client_id'] != client_id]
            self.fingerprints = [f for f in self.fingerprints if f['client_id'] != client_id]
            
            removed_faces = before_faces - len(self.faces)
            removed_prints = before_prints - len(self.fingerprints)
            
            if removed_faces > 0 or removed_prints > 0:
                print(f"🗑️  Removed client {client_id} from cache (faces: {removed_faces}, prints: {removed_prints})")
    
    def get_faces(self):
        """Get copy of face embeddings"""
        with self.lock:
            return self.faces.copy()
    
    def get_fingerprints(self):
        """Get copy of fingerprints"""
        with self.lock:
            return self.fingerprints.copy()

# Cached wrapper for get_client_info
def get_client_info_cached(client_id):
    """Get client info with caching to reduce database load"""
    cached = client_info_cache.get(client_id)
    if cached is not None:
        return cached
    
    # Not in cache, query database
    info = db.get_client_info(client_id)
    if info:
        client_info_cache.set(client_id, info)
    return info

def letterbox_resize(image, size, bg_color):
    """
    letterbox_resize the image according to the specified size
    :param image: input image, which can be a NumPy array or file path
    :param size: target size (width, height)
    :param bg_color: background filling data 
    :return: processed image
    """
    try:
        if isinstance(image, str):
            image = cv2.imread(image)
        
        if image is None or image.size == 0:
            raise ValueError("Invalid or empty image")

        target_width, target_height = size
        image_height, image_width, _ = image.shape

        # Calculate the adjusted image size
        aspect_ratio = min(target_width / image_width, target_height / image_height)
        new_width = int(image_width * aspect_ratio)
        new_height = int(image_height * aspect_ratio)

        # Use cv2.resize() for proportional scaling
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Create a new canvas and fill it
        result_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * bg_color
        offset_x = (target_width - new_width) // 2
        offset_y = (target_height - new_height) // 2
        result_image[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = image
        return result_image, aspect_ratio, offset_x, offset_y
    except Exception as e:
        print(f"Error in letterbox_resize: {e}")
        # Return default values
        return None, 1.0, 0, 0
    
def PriorBox(image_size): #image_size Support (320,320) and (640,640)
    """Generate prior boxes - with error handling"""
    try:    
        anchors = []
        min_sizes = [[16, 32], [64, 128], [256, 512]]
        steps = [8, 16, 32]
        feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps]
        for k, f in enumerate(feature_maps):
            min_sizes_ = min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes_:
                    s_kx = min_size / image_size[1]
                    s_ky = min_size / image_size[0]
                    dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        output = np.array(anchors).reshape(-1, 4)
        # print("image_size:",image_size," num_priors=",output.shape[0])
        return output
    except Exception as e:
        print(f"Error in PriorBox: {e}")
        return np.array([])

def box_decode(loc, priors):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    """Decode boxes with validation"""
    try:
        if loc is None or priors is None or len(loc) == 0 or len(priors) == 0:
            return np.array([])
        
        variances = [0.1, 0.2]
        boxes = np.concatenate((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes
    except Exception as e:
        print(f"Error in box_decode: {e}")
        return np.array([])

def decode_landm(pre, priors):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    try:
        if pre is None or priors is None or len(pre) == 0 or len(priors) == 0:
            return np.array([])
        variances = [0.1, 0.2]
        landmarks = np.concatenate((
            priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:]
        ), axis=1)
        return landmarks
    except Exception as e:
        print(f"Error in decode_landm: {e}")
        return np.array([])

def nms(dets, thresh):
    """Pure Python NMS baseline."""
    try:
        if dets is None or len(dets) == 0:
            return []
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
    except Exception as e:
        print(f"Error in nms: {e}")
        return []
    
def Similarity(alpha_embedding,beta_embedding):
    """Calculate similarity with error handling"""
    try:
        if alpha_embedding is None or beta_embedding is None:
            return 0.0
        alpha = np.array(alpha_embedding)
        beta = np.array(beta_embedding)


        dot_product = np.dot(alpha, beta)
        norm_alpha = np.linalg.norm(alpha)
        norm_beta = np.linalg.norm(beta)
        if norm_alpha == 0 or norm_beta == 0:
            return 0
        cosine_similarity = dot_product / (norm_alpha * norm_beta)
        cosine_distance = 1 - cosine_similarity

        return cosine_similarity
    except Exception as e:
        print(f"Error in Similarity: {e}")
        return 0.0

def safe_extract_face(img, coords):
    """Safely extract face region with bounds checking"""
    try:
        if img is None or img.size == 0:
            return None
        
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        
        # Validate and clip coordinates
        h, w = img.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        
        # Ensure valid region
        if x2 <= x1 or y2 <= y1:
            return None
        
        face_img = img[y1:y2, x1:x2]
        
        if face_img.size == 0:
            return None
        
        return face_img
    except Exception as e:
        print(f"Error in safe_extract_face: {e}")
        return None
def safe_socket_send(sock, data, address, timeout=0.1):
    """Context manager for safe socket operations"""
    try:
        sock.settimeout(timeout)
        yield sock.sendto(data, address)
    except socket.timeout:
        print("Socket send timeout")
    except Exception as e:
        print(f"Socket error: {e}")
    finally:
        sock.settimeout(None)


class FaceEnrollmentProcessor:
    """Handles automatic face enrollment when new members are added"""
    
    def __init__(self, db_helper, rknn_face, face_model_size=(112, 112), biometric_cache=None, fingerprint_system=None):
        self.db = db_helper
        self.rknn_face = rknn_face
        self.face_model_size = face_model_size
        self.biometric_cache = biometric_cache
        self.fingerprint_system = fingerprint_system
        self.processing_queue = Queue()
        self.is_running = True
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.process_thread.start()
        
    def enqueue_client(self, client_id):
        """Add client to processing queue"""
        self.processing_queue.put(client_id)
        print(f"📋 Queued client {client_id} for embedding extraction")
        
    def _process_queue(self):
        """Background thread to process enrollment queue"""
        while self.is_running:
            try:
                if not self.processing_queue.empty():
                    client_id = self.processing_queue.get()
                    self._process_client(client_id)
                else:
                    time.sleep(0.5)
            except Exception as e:
                print(f"❌ Error in processing queue: {e}")
                
    def _process_client(self, client_id):
        """Process a single client's face image"""
        try:
            print(f"🔄 Processing client {client_id}...")
            
            # Attempt to load image from database
            img = self.db.get_client_image(client_id)
            if img is None:
                print(f"❌ No image data available for client {client_id}")
                return
            
            # Process image for face detection
            model_height, model_width = (320, 320)
            letterbox_img, aspect_ratio, offset_x, offset_y = letterbox_resize(img, (model_height, model_width), 114)
            if letterbox_img is None:
                print(f"❌ Failed to preprocess image for client {client_id}")
                return
            infer_img = np.expand_dims(letterbox_img, 0)
            
            # Face detection inference
            with rknn_lock:
                outputs = rknn.inference(inputs=[infer_img])
            if outputs is None or len(outputs) < 3:
                print(f"❌ Face detection failed for client {client_id}")
                return
            loc, conf, landmarks = outputs
            priors = PriorBox(image_size=(model_height, model_width))
            boxes = box_decode(loc.squeeze(0), priors)
            if boxes.size == 0:
                print(f"❌ No face detected for client {client_id}")
                return
            scale = np.array([model_width, model_height, model_width, model_height])
            boxes = boxes * scale // 1
            boxes[...,0::2] = np.clip((boxes[...,0::2] - offset_x) / aspect_ratio, 0, img.shape[1])
            boxes[...,1::2] = np.clip((boxes[...,1::2] - offset_y) / aspect_ratio, 0, img.shape[0])
            scores = conf.squeeze(0)[:, 1]
            landmarks = decode_landm(landmarks.squeeze(0), priors)
            scale_landmarks = np.array([model_width, model_height, model_width, model_height,
                                        model_width, model_height, model_width, model_height,
                                        model_width, model_height])
            landmarks = landmarks * scale_landmarks // 1
            landmarks[...,0::2] = np.clip((landmarks[...,0::2] - offset_x) / aspect_ratio, 0, img.shape[1])
            landmarks[...,1::2] = np.clip((landmarks[...,1::2] - offset_y) / aspect_ratio, 0, img.shape[0])
            
            inds = np.where(scores > 0.1)[0]
            if len(inds) == 0:
                print(f"❌ No faces with sufficient confidence for client {client_id}")
                return
            boxes = boxes[inds]
            landmarks = landmarks[inds]
            scores = scores[inds]
            order = scores.argsort()[::-1]
            boxes = boxes[order]
            landmarks = landmarks[order]
            scores = scores[order]
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = nms(dets, 0.5)
            dets = dets[keep, :]
            landmarks = landmarks[keep]
            dets = np.concatenate((dets, landmarks), axis=1)
            
            for data in dets:
                if data[4] < 0.9:
                    continue
                data = list(map(int, data))
                dx =  data[7] - data[5]
                dy = data[8] - data[6]
                angle = math.atan2(dy, dx) * 180. / math.pi
                eye_center = ((data[5] + data[7]) // 2,(data[6] + data[8]) // 2)
                rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
                (h, w) = img.shape[:2]
                aligned_img = cv2.warpAffine(img, rotation_matrix, (w, h))
                face_img = aligned_img[(data[1]): (data[3]), (data[0]):(data[2])]
                if face_img.size == 0:
                    continue
                letterbox_face, Faspect_ratio, Foffset_x, Foffset_y = letterbox_resize(face_img, (self.face_model_size[0], self.face_model_size[1]), 114)
                if letterbox_face is None:
                    continue
                Finfer_img = np.expand_dims(letterbox_face, 0)
                with rknnFace_lock:
                    outputs = rknnFace.inference(inputs=[Finfer_img])
                if len(outputs) > 0:
                    embedding = outputs[0][0].astype(np.float32)
                    success = self.db.save_face_embedding(client_id, embedding, confidence=1.0)
                    if success:
                        client_info = get_client_info_cached(client_id)
                        name = f"{client_info['fname']} {client_info['lname']}" if client_info else "Unknown"
                        print(f"✅ Successfully enrolled: {name} (ID: {client_id})")                        
                        # Update cache if available
                        if self.biometric_cache:
                            self.biometric_cache.update_client_face(client_id)
                            registered_faces_cache[:] = self.biometric_cache.get_faces()
                        else:
                            # Fallback to reload function if cache not available
                            if 'reload_embeddings' in globals():
                                reload_embeddings()
                        # Trigger fingerprint enrollment asynchronously after face embedding
                        try:
                            if self.fingerprint_system:
                                def _enroll_fingerprint_bg():
                                    try:
                                        # Pause listeners to avoid interference
                                        was_entrance_running = self.fingerprint_system.entrance_running
                                        was_locker_running = self.fingerprint_system.locker_running
                                        if was_entrance_running:
                                            self.fingerprint_system.stop_entrance_listener()
                                        if was_locker_running:
                                            self.fingerprint_system.stop_locker_listener()
                                        # Name for logs
                                        display_name = name if name else f"Client {client_id}"
                                        print(f"🖐️ Auto-enrolling fingerprint for {display_name} (after face update)")
                                        # Use entrance ESP32 for enrollment
                                        success_fp = self.fingerprint_system.enroll_client_fingerprint(client_id, use_entrance_esp=True)
                                        if success_fp:
                                            print(f"✅ Fingerprint enrolled for {display_name}")
                                        else:
                                            print(f"⚠️ Fingerprint enrollment skipped/failed for {display_name}")
                                    except Exception as _e:
                                        print(f"❌ Error during auto fingerprint enrollment: {_e}")
                                    finally:
                                        # Resume listeners
                                        try:
                                            if 'was_entrance_running' in locals() and was_entrance_running:
                                                self.fingerprint_system.start_entrance_listener()
                                            if 'was_locker_running' in locals() and was_locker_running:
                                                self.fingerprint_system.start_locker_listener()
                                        except Exception:
                                            pass
                                threading.Thread(target=_enroll_fingerprint_bg, daemon=True).start()
                        except Exception as _e:
                            print(f"❌ Could not trigger fingerprint enrollment: {_e}")
                    else:
                        print(f"❌ Failed to save embedding for client {client_id}")
                else:
                    print(f"❌ No face embedding extracted for client {client_id}")
                    
        except Exception as e:
            print(f"❌ Error processing client {client_id}: {e}")
            import traceback
            traceback.print_exc()
    
    def stop(self):
        """Stop the processing thread"""
        self.is_running = False

def ps3camLoad():
    # Load library
    try:
        lib = ctypes.CDLL('/usr/local/lib/libps3eye_wrapper.so')
    except Exception as e:
        print(f"❌ Failed to load library: {e}")
        print("Run: ./use_cpp_wrapper_fixed.sh")
        sys.exit(1)

    # Define function signatures
    lib.ps3eye_init.restype = c_int
    lib.ps3eye_start.argtypes = [c_int, c_int, c_int]
    lib.ps3eye_start.restype = c_int
    lib.ps3eye_get_frame.argtypes = [POINTER(c_ubyte)]
    lib.ps3eye_get_frame.restype = c_int
    lib.ps3eye_stop.restype = None
    lib.ps3eye_set_gain.argtypes = [c_int]
    lib.ps3eye_set_exposure.argtypes = [c_int]
    lib.ps3eye_set_autogain.argtypes = [c_int]

    # Initialize
    print("Initializing camera...")
    if lib.ps3eye_init() == 0:
        print("❌ No camera found!")
        sys.exit(1)
    print("✓ Camera detected")

    # Start camera
    width, height, fps = 640, 480, 30
    if lib.ps3eye_start(width, height, fps) == 0:
        print("❌ Failed to start camera!")
        sys.exit(1)
    lib.ps3eye_set_exposure(200)

    return lib 

def enroll_fingerprint_for_client(client_id, client_name, biometric_cache_ref=None):
    """
    Enroll fingerprint for a client during gym entry/setup
    
    Args:
        client_id: Client database ID
        client_name: Client's full name
        biometric_cache_ref: Reference to biometric cache for updates
    
    Returns:
        True if successful, False otherwise
    """
    global fingerprint_system
    
    try:
        was_entrance_running = fingerprint_system.entrance_running
        was_locker_running = fingerprint_system.locker_running
        
        if was_entrance_running:
            fingerprint_system.stop_entrance_listener()
        if was_locker_running:
            fingerprint_system.stop_locker_listener()
        
        if was_entrance_running or was_locker_running:
            print("⏸️  Paused auto-identification for enrollment")

        print(f"\n🖐️ Starting fingerprint enrollment for {client_name} (ID: {client_id})")
        sendstr = f"ENROLL {client_id}"
        sock.sendto(sendstr.encode("utf-8"), (espcam_ip, espcam_port))

        # Check if already enrolled
        if db.check_fingerprint_exists(client_id):
            print(f"⚠️  {client_name} already has fingerprints enrolled")
        
            db.delete_fingerprint_features(client_id)
            print("✓ Existing fingerprints deleted")
        
        
        # Perform enrollment
        success = fingerprint_system.enroll_client_fingerprint(client_id)
        
        # Update cache after enrollment
        if success and biometric_cache_ref:
            biometric_cache_ref.update_client_fingerprints(client_id)
            registered_fingerprints_cache[:] = biometric_cache_ref.get_fingerprints()
        
        # Resume auto-identification
        if was_entrance_running:
            fingerprint_system.start_entrance_listener()
        if was_locker_running:
            fingerprint_system.start_locker_listener()
        
        if was_entrance_running or was_locker_running:
            print("▶️  Resumed auto-identification")

        if success:
            print(f"✅ Fingerprint enrollment complete for {client_name}")
            return True
        else:
            print(f"❌ Fingerprint enrollment failed for {client_name}")
            return False
    except ValueError:
        print("❌ Invalid client ID")
        if was_entrance_running:
            fingerprint_system.start_entrance_listener()
        if was_locker_running:
            fingerprint_system.start_locker_listener()
        return False
    except Exception as e:
        print(f"❌ Enrollment error: {e}")
        if was_entrance_running:
            fingerprint_system.start_entrance_listener()
        if was_locker_running:
            fingerprint_system.start_locker_listener()
        return False

def on_fingerprint_identified(client_id, client_name, confidence):
    """
    Callback function called when fingerprint is automatically identified
    
    Args:
        client_id: Client database ID (None if no match)
        client_name: Client name (None if no match)
        confidence: Match confidence score (0-1)
    """
    global sock, raspi_ip, raspi_port, esp32_1_ip, esp32_1_port, esp32_2_ip, esp32_2_port
    global last_locker_unlock, locker_unlock_cooldown
    
    if client_id is None:
        # No match found
        print("❌ Unknown fingerprint - Access DENIED")
        return
    
    print(f"\n🖐️ Fingerprint Match: {client_name} (ID: {client_id})")
    
    # Get client info
    client_info = get_client_info_cached(client_id)
    
    if not client_info:
        print(f"❌ Client info not found for {client_id}")
        return
    
    # Check if client has a locker assigned
    if client_info['locker'] is None:
        print(f"⚠️  {client_name} has no locker assigned")
        return
    
    locker_number = client_info['locker'] + 1
    current_time = time.time()
    
    # Check cooldown to prevent rapid unlocks
    if locker_number in last_locker_unlock:
        time_since_last = current_time - last_locker_unlock[locker_number]
        if time_since_last < locker_unlock_cooldown:
            remaining = int(locker_unlock_cooldown - time_since_last)
            print(f"⏳ Locker {locker_number} cooldown: {remaining}s remaining")
            return
    
    # Send unlock command to appropriate ESP32
    try:
        if locker_number > 0 and locker_number <= 24:
            sock.sendto(str(locker_number).encode("utf-8"), (esp32_1_ip, esp32_1_port))
            print(f"📤 Unlock command sent to ESP32-1")
        elif locker_number > 24 and locker_number <= 60:
            sock.sendto(str(locker_number).encode("utf-8"), (esp32_2_ip, esp32_2_port))
            print(f"📤 Unlock command sent to ESP32-2")
        
        # Also send to Raspberry Pi
        sock.sendto(str(locker_number).encode("utf-8"), (raspi_ip, raspi_port))
        
        # Update last unlock time
        last_locker_unlock[locker_number] = current_time
        
        print(f"✅ Locker {locker_number} unlocked via FINGERPRINT")
        print(f"   Client: {client_name}")
        print(f"   Confidence: {confidence*100:.1f}%")
        
        # Log access
        db.log_access(client_id, True, float(confidence))
        
    except Exception as e:
        print(f"❌ Error unlocking locker: {e}")

def on_entrance_fingerprint(client_id, client_name, confidence):
    # Handle entry/exit
    if client_id is None: return
    client_info = get_client_info_cached(client_id)
    if client_info['locker'] is None:
        # ENTRY - assign locker
        locker = db.get_available_locker()
        db.assign_locker_to_client(client_id, locker)
        db.decrease_membership_session(client_id)
        db.record_entrance(client_id, locker)
        print(f"🎉 ENTRY: {client_name} → Locker #{locker + 1}")
    else:
        # EXIT - return locker
        db.record_exit(client_id)
        db.unassign_locker(client_id)
        print(f"👋 EXIT: {client_name}")

def on_locker_fingerprint(client_id, client_name, confidence):
    """
    Callback function called when fingerprint is automatically identified
    
    Args:
        client_id: Client database ID (None if no match)
        client_name: Client name (None if no match)
        confidence: Match confidence score (0-1)
    """
    global sock, raspi_ip, raspi_port, esp32_1_ip, esp32_1_port, esp32_2_ip, esp32_2_port
    global last_locker_unlock, locker_unlock_cooldown
    
    if client_id is None:
        # No match found
        print("❌ Unknown fingerprint - Access DENIED")
        return
    
    print(f"\n🖐️ Fingerprint Match: {client_name} (ID: {client_id})")
    
    # Get client info
    client_info = get_client_info_cached(client_id)
    
    if not client_info:
        print(f"❌ Client info not found for {client_id}")
        return
    
    # Check if client has a locker assigned
    if client_info['locker'] is None:
        print(f"⚠️  {client_name} has no locker assigned")
        return
    
    locker_number = client_info['locker'] + 1
    current_time = time.time()
    
    # Check cooldown to prevent rapid unlocks
    if locker_number in last_locker_unlock:
        time_since_last = current_time - last_locker_unlock[locker_number]
        if time_since_last < locker_unlock_cooldown:
            remaining = int(locker_unlock_cooldown - time_since_last)
            print(f"⏳ Locker {locker_number} cooldown: {remaining}s remaining")
            return
    
    # Send unlock command to appropriate ESP32
    try:
        if locker_number > 0 and locker_number <= 24:
            sock.sendto(str(locker_number).encode("utf-8"), (esp32_1_ip, esp32_1_port))
            print(f"📤 Unlock command sent to ESP32-1")
        elif locker_number > 24 and locker_number <= 60:
            sock.sendto(str(locker_number).encode("utf-8"), (esp32_2_ip, esp32_2_port))
            print(f"📤 Unlock command sent to ESP32-2")
        
        # Also send to Raspberry Pi
        sock.sendto(str(locker_number).encode("utf-8"), (raspi_ip, raspi_port))
        
        # Update last unlock time
        last_locker_unlock[locker_number] = current_time
        
        print(f"✅ Locker {locker_number} unlocked via FINGERPRINT")
        print(f"   Client: {client_name}")
        print(f"   Confidence: {confidence*100:.1f}%")
        
        # Log access
        db.log_access(client_id, True, float(confidence))
        
    except Exception as e:
        print(f"❌ Error unlocking locker: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RetinaFace Python Demo', add_help=True)
    # basic params
    parser.add_argument('--model_path', type=str, required=True,
                        help='model path, could be .rknn file')
    parser.add_argument('--target', type=str,
                        default='rk3566', help='target RKNPU platform')
    parser.add_argument('--device_id', type=str,
                        default=None, help='device id')
    parser.add_argument('--db_host', type=str, required=True,
                        help='Database host IP')
    parser.add_argument('--mount_point', type=str, default='/mnt/winshare',
                        help='SSHFS mount point for images')
    args = parser.parse_args()

        # Initialize database connection
    print("🔌 Connecting to database...")
    db = DatabaseHelper(
        host=args.db_host,  # Your Windows PC IP
        database='gym-db',
        user='postgres',
        password='123456',
        port=5432,
        mount_point=args.mount_point
        )
    print("🖐️ Initializing fingerprint system...")
    
    print("🖐️ Fingerprint system initialized")
    tm = cv2.TickMeter()
    sock = None
    cap = None
    rknn = None
    rknnFace = None
    esp32_1_ip = "192.168.1.201"
    esp32_1_port = 4210
    esp32_2_ip = "192.168.1.202"
    esp32_2_port = 4210
    raspi_ip = "192.168.1.110"
    raspi_port = 4210
    espcam_ip = "192.168.1.111"
    espcam_port = 4210

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        fingerprint_system = DualESP32FingerprintSystem(
            db, on_entrance_fingerprint, on_locker_fingerprint, sock
        )
        # Note: fingerprint_cache will be set after biometric_cache is initialized

        # fingerprint_system = FingerprintGymSystem(db)
        # fingerprint_system = FingerprintGymSystem(
        #     db_helper=db,
        #     auto_identify=True,  # Enable automatic identification
        #     identification_callback=on_fingerprint_identified,  # Callback when fingerprint identified
        #     sock=sock  # Pass socket for sending messages
        # )
        # Load all registered face embeddings from database
        # embedding_shape = (512,)  # ArcFace embedding size
        # registered_faces = db.get_all_face_embeddings(embedding_shape)
        
        # print(f"Loaded {len(registered_faces)} registered faces from database")
        
        # Create RKNN object
        # rknn = RKNN(verbose=True)
        rknn = RKNNLite()
        rknnFace = RKNNLite()

        # Load RKNN model
        ret = rknn.load_rknn(args.model_path)
        if ret != 0:
            print('❌❌❌Load RKNN model \"{}\" failed!'.format(args.model_path))
            exit(ret)
        print('done')

        ret1 = rknnFace.load_rknn("/home/orangepi/Projects/GymController/model/Arcfacefp.rknn")
        if ret1 != 0:
            print('❌❌❌Load RKNN face model failed!')
            exit(ret1)
        print('done')


        # Init runtime environment
        print('--> Init runtime environment')
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)

        ret1 = rknnFace.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret1 != 0:
            print('Init runtime environment failed!')
            exit(ret1)
        print('done')

        rknn_lock = threading.Lock()
        rknnFace_lock = threading.Lock()
        
        # Initialize biometric cache FIRST (before callbacks)
        biometric_cache = BiometricCache(db)
        biometric_cache.load_all()
        
        # Update global cache references (already global at module level, no need for global keyword here)
        registered_faces_cache[:] = biometric_cache.get_faces()
        registered_fingerprints_cache[:] = biometric_cache.get_fingerprints()
        
        # Initialize face enrollment processor (pass cache and fingerprint system)
        enrollment_processor = FaceEnrollmentProcessor(db, rknnFace, biometric_cache=biometric_cache, fingerprint_system=fingerprint_system)
        
        # Define callback for database notifications
        def on_client_change(notification):
            """Handle database change notifications"""
            action = notification.get('action')
            client_id = notification.get('client_id')
            image_changed = notification.get('image_changed', False)
            
            print(f"📢 Client {action}: ID={client_id}")
            
            if client_id is None:
                return
            
            if action == 'DELETE':
                # Remove from cache
                biometric_cache.remove_client(client_id)
                # Remove from database
                db.delete_face_embedding(client_id)
                db.delete_fingerprint_features(client_id)
                # Update global cache (inside callback function, global keyword is needed)
                global registered_faces_cache, registered_fingerprints_cache
                registered_faces_cache[:] = biometric_cache.get_faces()
                registered_fingerprints_cache[:] = biometric_cache.get_fingerprints()
                return
            
            if action == 'UPDATE':
                # Update fingerprint cache if fingerprints exist
                if db.check_fingerprint_exists(client_id):
                    biometric_cache.update_client_fingerprints(client_id)
                    registered_fingerprints_cache[:] = biometric_cache.get_fingerprints()
                
                # Handle face embedding update
                if image_changed:
                    # Remove old embedding so it can be regenerated
                    db.delete_face_embedding(client_id)
                    biometric_cache.remove_client(client_id)
                    registered_faces_cache[:] = biometric_cache.get_faces()
            
            # Queue face embedding regeneration if needed
            if action in ('INSERT', 'UPDATE'):
                if image_changed or not db.check_embedding_exists(client_id):
                    enrollment_processor.enqueue_client(client_id)
                else:
                    # Face embedding exists, update cache
                    biometric_cache.update_client_face(client_id)
                    registered_faces_cache[:] = biometric_cache.get_faces()
                    print(f"ℹ️  Face embedding updated in cache for client {client_id}")
        
        # Update fingerprint system to use cached data
        fingerprint_system.set_fingerprint_cache(biometric_cache)
        
        # Start fingerprint auto-identification
        fingerprint_system.start_auto_identification()

        # Start listening for database changes
        db.start_listening(on_client_change)

        # Function to reload embeddings (call periodically or on notification)
        def reload_embeddings():
            biometric_cache.reload_faces()
            registered_faces_cache[:] = biometric_cache.get_faces()
            print(f"🔄 Reloaded {len(registered_faces_cache)} registered faces")

        # ADD THIS: Locker assignment tracking
        global last_locker_open

        last_locker_assignment = {}  # Track last assignment time per client
        last_locker_open = {}  # Track last open locker time per client
        locker_open_cooldown = 15 
        locker_assignment_cooldown = 60  # seconds between assignments for same client

        def assign_locker_to_recognized_face(client_id, client_name, similarity):
            """
            Assign/unassign locker based on recognition in USB camera
            - Entry: Assign locker if client has active membership and no locker
            - Exit: Unassign locker if client already has one
            """
            global last_locker_assignment
            
            current_time = time.time()
            
            # Check cooldown to prevent duplicate operations
            if client_id in last_locker_assignment:
                time_since_last = current_time - last_locker_assignment[client_id]
                if time_since_last < locker_assignment_cooldown:
                    remaining = int(locker_assignment_cooldown - time_since_last)
                    print(f"⏳ Cooldown active for {client_name} ({remaining}s remaining)")
                    return False
            
            # 1. Check if client has active membership
            membership = db.check_active_membership(client_id)
            
            if not membership:
                print(f"❌ {client_name} does not have an active membership!")
                # print(f"   Please ensure membership is paid and not expired.")
                return False
            
            if not membership['is_paid']:
                print(f"❌ {client_name}'s membership is not paid!")
                return False
            
            print(f"✅ Active membership verified for {client_name}")
            # print(f"   Valid until: {membership['end_date']}")
            
            # 2. Get client info to check current locker status
            client_info = get_client_info_cached(client_id)
            
            if not client_info:
                print(f"❌ Client info not found for {client_name}")
                return False
            
            current_locker = client_info['locker']
            
            # 3. Determine if this is ENTRY or EXIT
            # Determine if this is ENTRY or EXIT
            if current_locker is None:
                # ===== ENTRY: Check sessions and assign locker =====
                print(f"🚪 ENTRY detected for {client_name}")
                
                # 1. Check membership and remaining sessions
                membership_info = db.check_membership_sessions(client_id)
                
                if not membership_info:
                    print(f"❌ {client_name} does not have an active membership!")
                    print(f"   Please ensure membership is active and paid.")
                    return False
                
                if not membership_info['is_paid']:
                    print(f"❌ {client_name}'s membership is not paid!")
                    return False
                
                if not membership_info['has_access']:
                    print(f"❌ {client_name} has NO remaining sessions!")
                    print(f"   Current sessions: {membership_info['remain_sessions']}")
                    print(f"   Please renew membership or add more sessions.")
                    return False
                
                print(f"✅ Access GRANTED for {client_name}")
                print(f"   Remaining sessions: {membership_info['remain_sessions']}")
                print(f"   Valid until: {membership_info['end_date']}")
                
                # 2. Find available locker
                available_locker = db.get_available_locker()
                
                if available_locker is None:
                    print(f"⚠️  No available lockers for {client_name}")
                    return False
                
                # 3. Assign locker
                success = db.assign_locker_to_client(client_id, available_locker)
                
                if not success:
                    print(f"❌ Failed to assign locker to {client_name}")
                    return False
                
                # 4. Decrease session count (CRITICAL!)
                session_decreased = db.decrease_membership_session(client_id)
                
                if not session_decreased:
                    print(f"⚠️  Locker assigned but session count NOT decreased!")
                    # Consider rolling back locker assignment
                else:
                    print(f"✅ Session count decreased successfully")
                
                # 5. Record entrance time
                entrance_recorded = db.record_entrance(client_id, available_locker)
                
                if not entrance_recorded:
                    print(f"⚠️  Locker assigned but entrance recording failed")
                
                print(f"🎉 ENTRY COMPLETE: {client_name} → Locker #{available_locker}")
                print(f"   Entrance time recorded")
                print(f"   Session consumed")
                
                # Send locker number to ESP32
                sock.sendto(str(available_locker+1).encode("utf-8"), (esp32_1_ip, esp32_1_port))
                
                last_locker_assignment[client_id] = current_time
                return True
                
            else:
                # ===== EXIT: Unassign locker (NO session change on exit) =====
                print(f"🚪 EXIT detected for {client_name}")
                print(f"   Current locker: #{current_locker+1}")
                
                # Record exit time
                exit_recorded = db.record_exit(client_id)
                
                if not exit_recorded:
                    print(f"⚠️  No open session found for exit recording")
                
                # Unassign locker
                success = db.unassign_locker(client_id)
                
                if not success:
                    print(f"❌ Failed to unassign locker from {client_name}")
                    return False
                
                print(f"👋 EXIT COMPLETE: {client_name} left gym")
                print(f"   Locker #{current_locker} is now available")
                print(f"   Exit time recorded")
                
                # Display remaining sessions for next visit (NEW!)
                summary = db.get_membership_summary(client_id)
                if summary:
                    print(f"   Sessions remaining: {summary['sessions']}")
                
                last_locker_assignment[client_id] = current_time
                return True               
        # Set inputs
        # GStreamer pipeline for low-latency RTSP
    # rtph264depay converts the stream, and queue drops old frames.
        deviceId = 8
        # camrsstp = "rtsp://192.168.1.110:8554/live"
        camrstp = "rtsp://192.168.1.111:554/"

        def open_stream():
            try:
                gst_pipeline = (
                f"rtspsrc location={camrstp} latency=150 ! "
                "rtph264depay ! h264parse ! mppvideodec ! "
                "videoflip method=rotate-180 ! "              # Rotation
                "videobalance brightness=0.2 ! "              # Brightness (change 0.2)
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink max-buffers=1 drop=true"
                )
                # cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    print("✅ SUCCESS! Pipeline 1 works!")
                    return cap
                else:
                    print("❌ Failed")
                #     cap = cv2.VideoCapture(camrstp)
                #     cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
                #   return cap
                    return None
            except Exception as e:
                print(f"Error opening stream: {e}")
                return None
                
        cap = open_stream()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        retry_count = 0
        max_retries = 15
        consecutive_errors = 0
        max_consecutive_errors = 5


        # ADD THIS: Set up USB camera for locker assignment
        print("📷 Initializing USB camera...")
        buffer_size = 640 * 480 * 3
        buffer = (c_ubyte * buffer_size)()

        ps3Cap = ps3camLoad()
        # usb_cam = cv2.VideoCapture(0)  # 0 is usually the first USB camera
        # if not usb_cam.isOpened():
        #     print("⚠️  USB camera not found, trying camera 1...")
        #     usb_cam = cv2.VideoCapture(1)
        #     if not usb_cam.isOpened():
        #         print("❌ USB camera failed to open!")
        #         usb_cam = None
        #     else:
        #         print("✅ USB camera opened on index 1")
        # else:
        #     print("✅ USB camera opened on index 0")

        # Set USB camera properties
        # if usb_cam:
        #     usb_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #     usb_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        #     usb_cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        # cap.set(cv2.CAP_PROP_FPS,10)
        img_width = 640 # int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height = 480 #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        model_height, model_width = (320, 320)
        Face_model_height, Face_model_width = (112, 112)
        # Reload embeddings every N seconds
        last_reload_time = time.time()
        reload_interval = 120  # seconds
        
        print("🎬 Starting main recognition loop...")
        print("=" * 60)
        
        def process_locker_camera_frame(frame):
            if frame is None:
                return None
            letterbox_img, aspect_ratio, offset_x, offset_y = letterbox_resize(frame, (model_height,model_width), 114)  # letterbox缩放
            if letterbox_img is None:
                print("Letterbox resize failed, skipping frame")
                return frame
            # infer_img = letterbox_img[..., ::-1]  # BGR2RGB
            # cv2.namedWindow("Face", cv2.WINDOW_AUTOSIZE)
            # cv2.imshow("Face", infer_img)
            infer_img = np.expand_dims(letterbox_img, 0)

            try:
                # Inference
                # outputs = rknn.inference(inputs=[infer_img])
                with rknn_lock:
                    outputs = rknn.inference(inputs=[infer_img])
                if outputs is None or len(outputs) < 3:
                    print("Invalid inference output, skipping frame")
                    return frame
                loc, conf, landmarks = outputs

            except Exception as e:
                print("Inference failed:", e)
                return frame

            priors = PriorBox(image_size=(model_height, model_width))
            if len(priors) == 0:
                return frame
            boxes = box_decode(loc.squeeze(0), priors)
            if len(boxes) == 0:
                # tm.stop()
                # tm.reset()
                return frame
            scale = np.array([model_width, model_height,
                            model_width, model_height])
            boxes = boxes * scale // 1  # face box
            boxes[...,0::2] =np.clip((boxes[...,0::2] - offset_x) / aspect_ratio, 0, img_width)  #letterbox
            boxes[...,1::2] =np.clip((boxes[...,1::2] - offset_y) / aspect_ratio, 0, img_height) #letterbox
            scores = conf.squeeze(0)[:, 1]  # face score
            landmarks = decode_landm(landmarks.squeeze(
                0), priors)  # face keypoint data
            if len(landmarks) == 0:
                # tm.stop()
                # tm.reset()
                return frame
            scale_landmarks = np.array([model_width, model_height, model_width, model_height,
                                        model_width, model_height, model_width, model_height,
                                        model_width, model_height])
            landmarks = landmarks * scale_landmarks // 1
            landmarks[...,0::2] = np.clip((landmarks[...,0::2] - offset_x) / aspect_ratio, 0, img_width) #letterbox
            landmarks[...,1::2] = np.clip((landmarks[...,1::2] - offset_y) / aspect_ratio, 0, img_height) #letterbox
            # ignore low scores
            inds = np.where(scores > 0.1)[0]
            if len(inds) == 0:
                # tm.stop()
                # tm.reset()
                return frame
            boxes = boxes[inds]
            landmarks = landmarks[inds]
            scores = scores[inds]

            order = scores.argsort()[::-1]
            boxes = boxes[order]
            landmarks = landmarks[order]
            scores = scores[order]

            # NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            keep = nms(dets, 0.5)
            if len(keep) == 0:
                # tm.stop()
                # tm.reset()
                return frame
            dets = dets[keep, :]
            landmarks = landmarks[keep]
            dets = np.concatenate((dets, landmarks), axis=1)
            detected_client_ids = []
            for data in dets:
                try:
                    if data[4] < 0.85:
                        continue
                    # print("face @ (%d %d %d %d) %f"%(data[0], data[1], data[2], data[3], data[4]))
                    text = "{:.4f}".format(data[4])
                    data = list(map(int, data))
                    dx =  data[7] - data[5]
                    dy = data[8] - data[6]
                    angle = math.atan2(dy, dx) * 180. / math.pi  # Convert radians to degrees

                    # Calculate the center point between the eyes, which will be the rotation center
                    eye_center = ((data[5] + data[7]) // 2,(data[6] + data[8]) // 2)
                    # We use a scale of 1.0 to ensure the face size remains the same.
                    rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)

                    # Get the dimensions of the image
                    (h, w) = frame.shape[:2]

                    # Apply the affine transformation (rotation) to the image
                    aligned_img = cv2.warpAffine(frame, rotation_matrix, (w, h))

                    # face_img = aligned_img[(data[1]): (data[3]), (data[0]):(data[2])]
                    face_img = safe_extract_face(aligned_img, data)

                    if face_img.size == 0:
                        continue
                        
                    letterbox_face, Faspect_ratio, Foffset_x, Foffset_y = letterbox_resize(face_img, (Face_model_height,Face_model_width), 114)  # letterbox缩放
                    if letterbox_face is None:
                        continue
                    # infer_img = letterbox_img[..., ::-1]  # BGR2RGB
                    # if not HEADLESS:    
                    #     cv2.namedWindow("Face", cv2.WINDOW_AUTOSIZE)
                    #     cv2.imshow("Face", letterbox_face)
                    Finfer_img = np.expand_dims(letterbox_face, 0)
                    try:
                        # outputs = rknnFace.inference(inputs=[Finfer_img])
                        with rknnFace_lock:
                            outputs = rknnFace.inference(inputs=[Finfer_img])
                        # Then replace your code with this:
                        if len(outputs) > 0:
                            current_embedding = outputs[0][0]
                            best_match = None
                            best_similarity = 0.0
                            
                            for registered_face in registered_faces_cache:
                                sim = Similarity(registered_face['embedding'], current_embedding)
                                if sim > best_similarity:
                                    best_similarity = sim
                                    best_match = registered_face
                            
                            if best_similarity > 0.6:  # Adjust threshold as needed
                                client_id = best_match['client_id']
                                client_name = best_match['name']
                                
                                print(f"Access GRANTED: {client_name} (ID: {client_id}), Confidence: {best_similarity:.2f}")
                                
                                # Get locker number
                                client_info = get_client_info_cached(client_id)
                                
                                if client_info and client_info['locker'] is not None:
                                    locker_number = client_info['locker'] + 1
                                    current_time = time.time()
                                    
                                    # Check cooldown for this locker
                                    if locker_number in last_locker_unlock:
                                        time_since_last_unlock = current_time - last_locker_unlock[locker_number]
                                        
                                        if time_since_last_unlock < locker_unlock_cooldown:
                                            remaining_time = int(locker_unlock_cooldown - time_since_last_unlock)
                                            print(f"⏳ Locker {locker_number} cooldown active: {remaining_time}s remaining")
                                            print(f"   Please wait before unlocking again")
                                            continue  # Skip this iteration
                                    
                                    # Cooldown passed or first unlock - proceed
                                    locker_numbers = locker_number 
                                    sock.sendto(str(locker_number).encode("utf-8"), (raspi_ip, raspi_port))
                                    safe_socket_send(sock, str(locker_number).encode("utf-8"), (raspi_ip, raspi_port))
                                    if locker_numbers > 0 and locker_numbers < 25:
                                        # safe_socket_send(sock, str(locker_number).encode("utf-8"), (esp32_1_ip, esp32_1_port))
                                        sock.sendto(str(locker_numbers).encode("utf-8"), (esp32_1_ip, esp32_1_port))
                                    elif locker_numbers > 25 and locker_numbers < 61:
                                        # safe_socket_send(sock, str(locker_numbers).encode("utf-8"), (esp32_2_ip, esp32_2_port))
                                        sock.sendto(str(locker_numbers).encode("utf-8"), (esp32_2_ip, esp32_2_port))


                                    # Update last unlock time
                                    last_locker_unlock[locker_number] = current_time
                                    
                                    print(f"✅ Locker {locker_number} unlocked at {time.strftime('%H:%M:%S', time.localtime(current_time))}")
                                else:
                                    print(f"⚠️  Client {client_name} has no locker assigned")
                                
                                # Log access
                                db.log_access(client_id, True, float(best_similarity))
                            else:
                                print(f"Access DENIED: Unknown person, Best match: {best_similarity:.2f}")
                                db.log_access(None, False, float(best_similarity))
                    except Exception as e:
                        print(f"Face recognition error: {e}")
                
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue
            return frame

        def process_usb_camera_frame(frame):
            """Process frame from USB camera for locker assignment"""
            if frame is None:
                return None
            
            # Process frame for face detection
            letterbox_img, aspect_ratio, offset_x, offset_y = letterbox_resize(frame, (model_height, model_width), 114)
            if letterbox_img is None:
                print("Letterbox resize failed, skipping frame")
                return frame
            infer_img = np.expand_dims(letterbox_img, 0)

            try:
            # Inference
                with rknn_lock:
                     outputs = rknn.inference(inputs=[infer_img])
                # outputs = rknn.inference(inputs=[infer_img])
                if outputs is None or len(outputs) < 3:
                    print("Invalid inference output, skipping frame")
                    return frame
                loc, conf, landmarks = outputs
                
            except Exception as e:
                    print(f"Inference error: {e}")
                    return frame
            
            priors = PriorBox(image_size=(model_height, model_width))
            if len(priors) == 0:
                return frame
            boxes = box_decode(loc.squeeze(0), priors)
            scale = np.array([model_width, model_height, model_width, model_height])
            boxes = boxes * scale // 1
            boxes[..., 0::2] = np.clip((boxes[..., 0::2] - offset_x) / aspect_ratio, 0, img_width)
            boxes[..., 1::2] = np.clip((boxes[..., 1::2] - offset_y) / aspect_ratio, 0, img_height)
            scores = conf.squeeze(0)[:, 1]
            landmarks = decode_landm(landmarks.squeeze(0), priors)
            if len(landmarks) == 0:
                # tm.stop()
                # tm.reset()
                return frame
                
            scale_landmarks = np.array([model_width, model_height, model_width, model_height,
                                        model_width, model_height, model_width, model_height,
                                        model_width, model_height])
            landmarks = landmarks * scale_landmarks // 1
            landmarks[..., 0::2] = np.clip((landmarks[..., 0::2] - offset_x) / aspect_ratio, 0, img_width)
            landmarks[..., 1::2] = np.clip((landmarks[..., 1::2] - offset_y) / aspect_ratio, 0, img_height)
            
            # Filter low scores
            inds = np.where(scores > 0.1)[0]
            if len(inds) == 0:
                # tm.stop()
                # tm.reset()
                return frame
            
            boxes = boxes[inds]
            landmarks = landmarks[inds]
            scores = scores[inds]

            order = scores.argsort()[::-1]
            boxes = boxes[order]
            landmarks = landmarks[order]
            scores = scores[order]

            # NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = nms(dets, 0.5)
            if len(keep) == 0:
                # tm.stop()
                # tm.reset()
                return frame
            
            dets = dets[keep, :]
            landmarks = landmarks[keep]
            dets = np.concatenate((dets, landmarks), axis=1)
            detected_client_ids = []
            for data in dets:
                try:    
                    if data[4] < 0.85:
                        continue
                        
                    data = list(map(int, data))
                    dx = data[7] - data[5]
                    dy = data[8] - data[6]
                    angle = math.atan2(dy, dx) * 180. / math.pi

                    eye_center = ((data[5] + data[7]) // 2, (data[6] + data[8]) // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
                    (h, w) = frame.shape[:2]
                    aligned_img = cv2.warpAffine(frame, rotation_matrix, (w, h))

                    # face_img = aligned_img[data[1]:data[3], data[0]:data[2]]
                    face_img = safe_extract_face(aligned_img, data)
                    if face_img.size == 0:
                        continue
                        
                    letterbox_face, _, _, _ = letterbox_resize(face_img, (Face_model_height, Face_model_width), 114)
                    if letterbox_face is None:
                        continue
                    Finfer_img = np.expand_dims(letterbox_face, 0)
                    try:
                        # outputs = rknnFace.inference(inputs=[Finfer_img])
                        with rknnFace_lock:
                            outputs = rknnFace.inference(inputs=[Finfer_img])
                        if len(outputs) > 0:
                            current_embedding = outputs[0][0]
                            # print(f"current_embedding:",current_embedding)
                            # Compare with all registered faces
                            best_match = None
                            best_similarity = 0.0
                            
                            for registered_face in registered_faces_cache:
                                sim = Similarity(registered_face['embedding'], current_embedding)
                                # print(f"sim", sim,registered_face['name'])
                                if sim > best_similarity:
                                    best_similarity = sim
                                    best_match = registered_face
                            print(f"sim", best_similarity,best_match['name'])
                            if best_similarity > 0.60:  # Slightly higher threshold for assignment
                                client_id = best_match['client_id']
                                client_name = best_match['name']
                                
                                # Check current locker status
                                client_info = get_client_info_cached(client_id)
                                if client_info:
                                    if client_info['locker'] is None:
                                        status = "ENTRY"
                                        color = (0, 255, 0)  # Green for entry
                                    else:
                                        status = f"EXIT (Locker {client_info['locker']})"
                                        color = (255, 165, 0)  # Orange for exit
                                else:
                                    status = "UNKNOWN"
                                    color = (128, 128, 128)
                                
                                # Draw rectangle and name with status
                                cv2.rectangle(frame, (data[0], data[1]), (data[2], data[3]), color, 2)
                                # cv2.putText(frame, f"{client_name}", 
                                #         (data[0], data[1] - 30),
                                #         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                cv2.putText(frame, f"{status} ({best_similarity:.2f})", 
                                        (data[0], data[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
                                # Assign or unassign locker
                                assign_locker_to_recognized_face(client_id, client_name, best_similarity)
                            else:
                                # Unknown face
                                cv2.rectangle(frame, (data[0], data[1]), (data[2], data[3]), (0, 0, 255), 2)
                                cv2.putText(frame, "Unknown", (data[0], data[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    except Exception as e:
                        print(f"Face recognition error: {e}")
                
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue
            cv2.putText(frame, "LOCKER ASSIGNMENT CAMERA", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            # Add label to frame
            
            return frame
        # img = cv2.imread('../model/result.jpg')
        # letterbox_face, Faspect_ratio, Foffset_x, Foffset_y = letterbox_resize(img, (Face_model_height,Face_model_width), 114)  # letterbox缩放

        # fame_imag = np.expand_dims(letterbox_face, 0)
        # amin = rknnFace.inference(inputs=[fame_imag])[0][0]
        # img = cv2.imread('../model/result.jpg')
        # letterbox_face, Faspect_ratio, Foffset_x, Foffset_y = letterbox_resize(img, (Face_model_height,Face_model_width), 114)  # letterbox缩放

        # fame_imag = np.expand_dims(letterbox_face, 0)
        # amin1 = rknnFace.inference(inputs=[fame_imag])[0][0]
        # simi = Similarity(amin,amin1)
        # print("amin ==============" , simi)
            # Locker unlock cooldown tracking (ADD THIS BEFORE WHILE LOOP)
        last_locker_unlock = {}  # Track last unlock time per locker
        locker_unlock_cooldown = 10  # seconds between unlocks for same locker
        frame_count = 0
        while cv2.waitKey(1) < 0:
            try:
                frame_count += 1
                # tm.start()
                # if frame_count % 300 == 0:  # Every 10 seconds
                #     check_db_pool_health()
                # # Periodically reload embeddings
                # if time.time() - last_reload_time > reload_interval:
                #     reload_embeddings()
                #     last_reload_time = time.time()
            
                if ps3Cap.ps3eye_get_frame(buffer) == 0:
                    print("Failed to get frame")
                    break
                # Convert to numpy array
                frame = np.frombuffer(buffer, dtype=np.uint8)
                frame = frame.reshape((480, 640, 3))

                usb_img = process_usb_camera_frame(frame)

                # ret, frame = usb_cam.read()
                # if not ret or frame is None:
                #     print("Failed to get frame")
                #     # ps3Cap = ps3camLoad()
                #     usb_cam = cv2.VideoCapture(0)
                #     retry_count += 1
                #     if retry_count > 15:
                #         print("Too many retries, resetting ps3camLoad...")
                #         break
                #     continue
                retry_count = 0
                ret, img = cap.read()
                if not ret or img is None:
                    print("⚠️ Stream broken. Reconnecting...")
                    if cap is not None:
                        cap.release()
                    time.sleep(0.5)
                    cap = open_stream()
                    retry_count += 1
                    if retry_count > 15:
                        print("Too many retries, resetting camera/system...")
                        # here you can reset camera service, or reboot camera if possible
                        break
                    continue
                retry_count = 0
                # cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
                locker_cam = process_locker_camera_frame(img)

                # if usb_img is None:
                #     continue
        
                
                
            
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break
            except Exception as e:
                print(f"Unexpected error in main loop: {e}")
                traceback.print_exc()
                consecutive_errors += 1
                if consecutive_errors > max_consecutive_errors:
                    print("Too many consecutive errors. Exiting.")
                    break
                time.sleep(0.1)
                continue
        
            try:
                # tm.stop()
                # print('FPS! ',tm.getFPS())
                if not HEADLESS:
                    cv2.namedWindow("locker_cam CAM", cv2.WINDOW_FULLSCREEN)
                    cv2.imshow("locker_cam CAM", locker_cam)
                    cv2.namedWindow("USB CAM", cv2.WINDOW_FULLSCREEN)
                    cv2.imshow("USB CAM", usb_img)
            
            except Exception as e:
                print(f"Display error: {e}")
    # cv2.imwrite(img_path, img)
    # print("save image in", img_path)
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()

    finally:
        # Cleanup
        print("\n🛑 Shutting down...")
        db.stop_listening()
        db.close_all_connections()
        if rknn is not None:
            try:
                rknn.release()
            except:
                pass
        
        if rknnFace is not None:
            try:
                rknnFace.release()
            except:
                pass
        if cap is not None:
            try:
                cap.release()
            except:
                pass
        if sock is not None:
            try:
                sock.close()
            except:
                pass
        try:
            cv2.destroyAllWindows()
        except:
            pass
        enrollment_processor.stop()        
        # Add fingerprint cleanup
        if fingerprint_system:
            try:
                fingerprint_system.close()
                print("✓ Fingerprint system closed")
            except:
                pass
        print("✅ Shutdown complete")
        