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

# Configuration
UDP_PORT_ENTRANCE = 8889  # Port for entrance ESP32 (enrollment & entry)
UDP_PORT_LOCKER = 8890    # Port for locker ESP32 (unlock only)
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 80
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
EXPECTED_MARKER = b"FPIMG\0\0\0"
END_MARKER = b"FPIMGEND"

# Fingerprint matching parameters
FINGERS_PER_CLIENT = 2
SAMPLES_PER_FINGER = 5
MATCH_THRESHOLD = 0.10  # Adjusted for realistic SIFT/ORB matching

# ESP32 endpoints for feedback
ENTRANCE_ESP_IP = "192.168.1.120"  # Entrance/enrollment ESP32
ENTRANCE_ESP_PORT = 4210
LOCKER_ESP_IP = "192.168.1.121"    # Locker ESP32
LOCKER_ESP_PORT = 4210
pc_ip = "192.168.1.3"            # PC
pc_port = 4211
raspi_entrance_ip = "192.168.1.111"
raspi_entrance_port = 4210

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
        
        while time.time() - start_time < timeout:
            try:
                data, addr = self.sock.recvfrom(2048)
                
                # Check for header
                if not header_received and data[:8] == EXPECTED_MARKER:
                    if len(data) >= 20:
                        width, height, size, timestamp = struct.unpack('<HHI I', data[8:20])
                        print(f"📥 Receiving fingerprint: {width}x{height}, {size} bytes")
                        header_received = True
                        expected_size = size
                    continue
                
                # Check for end marker
                if data == END_MARKER:
                    if len(image_data) == expected_size:
                        print(f"✓ Fingerprint received ({len(image_data)} bytes)")
                        # Convert to numpy array
                        img_array = np.frombuffer(image_data, dtype=np.uint8)
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
        self.processor = FingerprintProcessor()
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
        
        print("✅ Dual ESP32 fingerprint system initialized")
    
    def set_fingerprint_cache(self, fingerprint_cache):
        """Set or update the fingerprint cache reference"""
        self.fingerprint_cache = fingerprint_cache
        if fingerprint_cache:
            self.registered_fingerprints = fingerprint_cache.get_fingerprints()
    
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
                result = self._identify_fingerprint(image, "ENTRANCE")
                
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
                result = self._identify_fingerprint(image, "LOCKER")
                
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
    
    def _identify_fingerprint(self, image, source="UNKNOWN"):
        """
        Identify fingerprint image
        
        Returns:
            Tuple of (client_id, name, confidence) or None
        """
        # Refresh fingerprints from cache if available
        if self.fingerprint_cache:
            self.registered_fingerprints = self.fingerprint_cache.get_fingerprints()
        
        # Extract features
        features = self.processor.extract_features(image)
        
        if features is None:
            print(f"✗ [{source}] Failed to extract features")
            return None
        
        print(f"✓ [{source}] Extracted {features['num_features']} features")
        
        # Match against database
        best_match = None
        best_score = 0.0
        
        if not self.registered_fingerprints:
            print(f"⚠️ [{source}] No fingerprints loaded in database")
            return None
        
        print(f"🔍 [{source}] Checking against {len(self.registered_fingerprints)} enrolled clients...")
        
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
            
            if not all_samples:
                continue
            
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
            
            if avg_score > best_score:
                best_score = avg_score
                best_match = (client_id, name)
                print(f"  📊 [{source}] Best so far: {name} - {avg_score*100:.1f}% ({len(scores)} valid matches)")
        
        # Check threshold
        if best_match and best_score >= MATCH_THRESHOLD:
            client_id, name = best_match
            print(f"✅ [{source}] MATCH: {name} (ID: {client_id}) - {best_score*100:.1f}%")
            send_str = f"✅ [{source}] MATCH: {name} (ID: {client_id}) - {best_score*100:.1f}%"
            self.sock.sendto(send_str.encode("utf-8"), (pc_ip, pc_port))
            self.sock.sendto(send_str.encode("utf-8"), (raspi_entrance_ip, raspi_entrance_port))
            return (client_id, name, best_score)
        else:
            if best_match:
                print(f"❌ [{source}] NO MATCH - Best: {best_match[1]} (Score: {best_score*100:.1f}%, Threshold: {MATCH_THRESHOLD*100:.1f}%)")
            else:
                print(f"❌ [{source}] NO MATCH - No fingerprints found in database")
            return None
    
    def enroll_client_fingerprint(self, client_id, use_entrance_esp=True):
        """
        Backward-compatible entry point: Enroll both fingers (1 and 2), 5 samples each
        
        Args:
            client_id: Client ID to enroll
            use_entrance_esp: If True, use entrance ESP32, else use locker ESP32
        
        Returns:
            True if successful
        """
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
        
        return success
    
    def enroll_client_single_finger(self, client_id, finger_index=1, use_entrance_esp=True):
        """
        Enroll 5 fingerprint samples for a specific finger (1 or 2)
        
        Args:
            client_id: Client ID to enroll
            finger_index: 1 or 2 (which finger)
            use_entrance_esp: If True, use entrance ESP32, else use locker ESP32
        
        Returns:
            True if successful
        """
        single_pause = self._request_pause(f"fingerprint enrollment - finger {finger_index}")
        try:
            print(f"\n{'='*60}")
            print(f"🖐️ FINGERPRINT ENROLLMENT - Client ID: {client_id} - Finger {finger_index}")
            print(f"{'='*60}")
            send_str = f"🖐️ FINGERPRINT ENROLLMENT - Client ID: {client_id} - Finger {finger_index}"
            self.sock.sendto(send_str.encode("utf-8"), (pc_ip, pc_port))
            self.sock.sendto(send_str.encode("utf-8"), (raspi_entrance_ip, raspi_entrance_port))
            
            # Get client info
            client_info = self.db.get_client_info(client_id)
            if not client_info:
                print(f"❌ Client {client_id} not found")
                return False
            
            name = f"{client_info['fname']} {client_info['lname']}"
            print(f"👤 Enrolling: {name}")
            print(f"Using: {'ENTRANCE' if use_entrance_esp else 'LOCKER'} ESP32")
            print(f"Please scan {SAMPLES_PER_FINGER} fingerprint samples for finger {finger_index}...")
            send_str = f"👤 Enrolling: {name}"
            self.sock.sendto(send_str.encode("utf-8"), (pc_ip, pc_port))
            self.sock.sendto(send_str.encode("utf-8"), (raspi_entrance_ip, raspi_entrance_port))
            send_str = f"Please scan {SAMPLES_PER_FINGER} fingerprint samples for finger {finger_index}..."
            self.sock.sendto(send_str.encode("utf-8"), (pc_ip, pc_port))
            self.sock.sendto(send_str.encode("utf-8"), (raspi_entrance_ip, raspi_entrance_port))

            receiver = self.entrance_receiver if use_entrance_esp else self.locker_receiver
            
            esp_ip = ENTRANCE_ESP_IP if use_entrance_esp else LOCKER_ESP_IP
            esp_port = ENTRANCE_ESP_PORT if use_entrance_esp else LOCKER_ESP_PORT
            
            samples = []
            for i in range(SAMPLES_PER_FINGER):
                print(f"\n📌 Finger {finger_index} - Sample {i+1}/{SAMPLES_PER_FINGER}")
                send_str = f"\n📌 Finger {finger_index} - Sample {i+1}/{SAMPLES_PER_FINGER}"
                self.sock.sendto(send_str.encode("utf-8"), (pc_ip, pc_port))
                self.sock.sendto(send_str.encode("utf-8"), (raspi_entrance_ip, raspi_entrance_port))
                
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
                
                # Extract features
                print("🔍 Extracting features...")
                features = self.processor.extract_features(image)
                
                if features is None:
                    print(f"✗ Failed to extract features from sample {i+1}")
                    continue
                
                samples.append(features)
                print(f"✓ Sample {i+1} captured ({features['num_features']} features)")
                send_str = f"✓ Sample {i+1} captured ({features['num_features']} features)"
                self.sock.sendto(send_str.encode("utf-8"), (pc_ip, pc_port))
                self.sock.sendto(send_str.encode("utf-8"), (raspi_entrance_ip, raspi_entrance_port))
                
                if i < SAMPLES_PER_FINGER - 1:
                    print("   Remove and place finger again...")
                    send_str = " Remove and place finger again..."
                    self.sock.sendto(send_str.encode("utf-8"), (pc_ip, pc_port))
                    self.sock.sendto(send_str.encode("utf-8"), (raspi_entrance_ip, raspi_entrance_port))
                    time.sleep(1)
            
            if len(samples) < SAMPLES_PER_FINGER:
                print("✗ Not enough valid samples for this finger")
                return False
            
            # Save to database with finger_index
            success = self.db.save_fingerprint_features(client_id, finger_index, samples)
            
            if success:
                print(f"✅ Saved {SAMPLES_PER_FINGER} samples for finger {finger_index}")
                send_str = f"✅ Saved {SAMPLES_PER_FINGER} samples for finger {finger_index}"
                self.sock.sendto(send_str.encode("utf-8"), (pc_ip, pc_port))
                self.sock.sendto(send_str.encode("utf-8"), (raspi_entrance_ip, raspi_entrance_port))
                # Send success to ESP32
                if self.sock:
                    try:
                        sendstr = f"Success Finger {finger_index}"
                        self.sock.sendto(sendstr.encode("utf-8"), (esp_ip, esp_port))
                        self.sock.sendto(send_str.encode("utf-8"), (pc_ip, pc_port))
                    except Exception:
                        pass
                
                # Update cache if available
                if self.fingerprint_cache:
                    success_update = self.fingerprint_cache.update_client_fingerprints(client_id)
                    if success_update:
                        # Update registered fingerprints from cache
                        self.registered_fingerprints = self.fingerprint_cache.get_fingerprints()
                        print(f"✅ Fingerprint cache updated in fingerprint system")
                        send_str = f"✅ Fingerprint cache updated in fingerprint system"
                        self.sock.sendto(send_str.encode("utf-8"), (pc_ip, pc_port))
                        self.sock.sendto(send_str.encode("utf-8"), (raspi_entrance_ip, raspi_entrance_port))

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
            
            return success
        finally:
            if single_pause:
                self._release_pause()
    
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
                send_str = f"⚠️  Finger {finger_index} enrollment failed"
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
        self.processor = FingerprintProcessor()
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
        send_str = f"🖐️ FINGERPRINT ENROLLMENT - Client ID: {client_id}"
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
            
            # Extract features
            print("🔍 Extracting features...")
            features = self.processor.extract_features(image)
            
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

