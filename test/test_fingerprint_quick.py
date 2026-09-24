#!/usr/bin/env python3
"""
Fingerprint Test with UDP Receiver
Supports enrollment and identification with local storage
"""

import socket
import struct
import numpy as np
import cv2
import time
import sys
import os
import pickle
import json
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configuration
UDP_IP = "0.0.0.0"  # Listen on all interfaces
UDP_PORT = 8889
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 80
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT  # 5120 bytes

# Local storage
STORAGE_DIR = "fingerprint_storage"
DATABASE_FILE = os.path.join(STORAGE_DIR, "fingerprints.db")
IMAGES_DIR = os.path.join(STORAGE_DIR, "images")
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Matching threshold
MATCH_THRESHOLD = 0.7  # For improved processor (cosine similarity)
MATCH_THRESHOLD_OLD = 0.12  # For old processor

print("="*60)
print("🧪 FINGERPRINT TEST - UDP Receiver")
print("="*60)

# Try to import processors
try:
    from fingerprint_gym import FingerprintProcessor
    old_processor = FingerprintProcessor()
    print("✅ Old processor loaded")
except Exception as e:
    print(f"⚠️  Old processor: {e}")
    old_processor = None

try:
    from fingerprint_processor_improved import ImprovedFingerprintProcessor
    improved_processor = ImprovedFingerprintProcessor(embedding_dim=128, use_pca=True)
    print("✅ Improved processor loaded")
except Exception as e:
    print(f"⚠️  Improved processor: {e}")
    improved_processor = None

if not old_processor and not improved_processor:
    print("❌ No fingerprint processors available!")
    sys.exit(1)

# Use improved processor if available, otherwise old
processor = improved_processor if improved_processor else old_processor
use_improved = improved_processor is not None


def receive_fingerprint_image(sock, timeout=30.0):
    """
    Receive a complete fingerprint image via UDP
    Based on fingerprint_receiver.py
    
    Returns: numpy array of the image, or None if failed
    """
    print("\n" + "="*50)
    print("⏳ Waiting for fingerprint image...")
    
    image_buffer = bytearray()
    header_received = False
    image_width = IMAGE_WIDTH
    image_height = IMAGE_HEIGHT
    expected_size = IMAGE_SIZE
    start_time = time.time()
    
    # Set socket timeout
    sock.settimeout(1.0)
    
    try:
        while time.time() - start_time < timeout:
            try:
                data, addr = sock.recvfrom(2048)
            except socket.timeout:
                continue
            
            # Check for header packet
            if not header_received and data.startswith(b'FPIMG'):
                # Parse header: marker(8) + width(2) + height(2) + size(4) + timestamp(4)
                marker = data[0:8]
                image_width = struct.unpack('<H', data[8:10])[0]
                image_height = struct.unpack('<H', data[10:12])[0]
                expected_size = struct.unpack('<I', data[12:16])[0]
                timestamp = struct.unpack('<I', data[16:20])[0]
                
                print(f"📥 Header received from {addr[0]}:{addr[1]}")
                print(f"   Dimensions: {image_width}x{image_height}")
                print(f"   Expected size: {expected_size} bytes")
                
                header_received = True
                continue
            
            # Check for end marker
            if data.startswith(b'FPIMGEND'):
                print(f"\n✓ End marker received")
                print(f"  Total bytes: {len(image_buffer)}/{expected_size}")
                break
            
            # Collect image data
            if header_received:
                image_buffer.extend(data)
                progress = (len(image_buffer) / expected_size) * 100
                print(f"  Progress: {progress:.1f}%", end='\r')
        
        # Check if we received the complete image
        if len(image_buffer) < expected_size:
            print(f"\n⚠️  Incomplete image! Got {len(image_buffer)} bytes, expected {expected_size}")
            # Pad with zeros if needed
            image_buffer.extend(b'\x00' * (expected_size - len(image_buffer)))
        elif len(image_buffer) > expected_size:
            print(f"\n⚠️  Extra data! Got {len(image_buffer)} bytes, expected {expected_size}")
            # Truncate to expected size
            image_buffer = image_buffer[:expected_size]
        
        # Convert to numpy array and reshape
        image_array = np.frombuffer(image_buffer, dtype=np.uint8)
        image_array = image_array.reshape((image_height, image_width))
        
        print(f"✅ Image received successfully!")
        return image_array
        
    except socket.timeout:
        print("\n⏱️  Timeout: No data received")
        return None
    except Exception as e:
        print(f"\n❌ Error receiving image: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_image(image_array, filename=None):
    """Save fingerprint image as PNG"""
    if image_array is None:
        return None
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fingerprint_{timestamp}.png"
    
    filepath = os.path.join(IMAGES_DIR, filename)
    cv2.imwrite(filepath, image_array)
    return filepath


def load_database():
    """Load fingerprint database from file"""
    if not os.path.exists(DATABASE_FILE):
        return {}
    
    try:
        with open(DATABASE_FILE, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"⚠️  Error loading database: {e}")
        return {}


def save_database(database):
    """Save fingerprint database to file"""
    try:
        with open(DATABASE_FILE, 'wb') as f:
            pickle.dump(database, f)
        return True
    except Exception as e:
        print(f"❌ Error saving database: {e}")
        return False


def enroll_fingerprint(sock, person_name, num_samples=5):
    """
    Enroll a new fingerprint
    
    Args:
        sock: UDP socket
        person_name: Name/ID for the person
        num_samples: Number of samples to collect
    
    Returns:
        True if successful
    """
    print("\n" + "="*60)
    print(f"🖐️  ENROLLMENT: {person_name}")
    print("="*60)
    print(f"Please scan {num_samples} fingerprint samples...")
    
    samples = []
    
    for i in range(num_samples):
        print(f"\n📌 Sample {i+1}/{num_samples}")
        print("   Place finger on sensor...")
        
        # Receive image
        image = receive_fingerprint_image(sock, timeout=60.0)
        if image is None:
            print(f"❌ Failed to receive sample {i+1}")
            if i < 2:  # Need at least 3 samples
                return False
            break
        
        # Save image
        img_filename = f"{person_name}_sample_{i+1}.png"
        save_image(image, img_filename)
        
        # Extract features
        print("🔍 Extracting features...")
        features = processor.extract_features(image)
        
        if features is None:
            print(f"⚠️  Failed to extract features from sample {i+1}")
            continue
        
        samples.append(features)
        print(f"✅ Sample {i+1} captured ({features['num_features']} features)")
        
        if i < num_samples - 1:
            print("   Remove and place finger again...")
            time.sleep(1)
    
    if len(samples) < 3:
        print(f"❌ Not enough valid samples (got {len(samples)}, need at least 3)")
        return False
    
    # Load database
    database = load_database()
    
    # Store fingerprint
    if person_name not in database:
        database[person_name] = {
            'name': person_name,
            'samples': [],
            'enrolled_at': datetime.now().isoformat(),
            'num_samples': 0
        }
    
    database[person_name]['samples'] = samples
    database[person_name]['num_samples'] = len(samples)
    database[person_name]['updated_at'] = datetime.now().isoformat()
    
    # Save database
    if save_database(database):
        print(f"\n✅ Enrollment complete!")
        print(f"   Name: {person_name}")
        print(f"   Samples: {len(samples)}")
        return True
    else:
        print("❌ Failed to save to database")
        return False


def identify_fingerprint(sock):
    """
    Identify a fingerprint
    
    Args:
        sock: UDP socket
    
    Returns:
        (name, confidence) or (None, 0.0)
    """
    print("\n" + "="*60)
    print("🔍 IDENTIFICATION")
    print("="*60)
    print("Please scan fingerprint...")
    
    # Receive image
    image = receive_fingerprint_image(sock, timeout=60.0)
    if image is None:
        print("❌ Failed to receive fingerprint")
        return None, 0.0
    
    # Save query image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_image(image, f"query_{timestamp}.png")
    
    # Extract features
    print("🔍 Extracting features...")
    query_features = processor.extract_features(image)
    if query_features is None:
        print("❌ Failed to extract features")
        return None, 0.0
    
    print(f"✅ Extracted {query_features['num_features']} features")
    
    # Load database
    database = load_database()
    
    if len(database) == 0:
        print("⚠️  No fingerprints enrolled in database")
        return None, 0.0
    
    print(f"\n🔎 Checking against {len(database)} enrolled persons...")
    
    # Match against all enrolled fingerprints
    best_match = None
    best_score = 0.0
    
    start_time = time.time()
    
    for person_name, person_data in database.items():
        samples = person_data.get('samples', [])
        if not samples:
            continue
        
        # Compare with all samples
        scores = []
        for enrolled_fp in samples:
            if not isinstance(enrolled_fp, dict) or 'descriptors' not in enrolled_fp:
                continue
            
            if use_improved:
                score = processor.match_fingerprints_improved(query_features, enrolled_fp)
            else:
                score = processor.match_fingerprints(query_features, enrolled_fp)
            
            scores.append(score)
        
        if scores:
            # Use average of top scores
            scores.sort(reverse=True)
            avg_score = np.mean(scores[:3]) if len(scores) >= 3 else np.mean(scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_match = person_name
    
    elapsed = (time.time() - start_time) * 1000
    
    # Check threshold
    threshold = MATCH_THRESHOLD if use_improved else MATCH_THRESHOLD_OLD
    
    print(f"\n⏱️  Matching took {elapsed:.2f}ms")
    
    if best_match and best_score >= threshold:
        print(f"✅ MATCH FOUND!")
        print(f"   Name: {best_match}")
        print(f"   Confidence: {best_score*100:.1f}%")
        print(f"   Threshold: {threshold*100:.1f}%")
        return best_match, best_score
    else:
        if best_match:
            print(f"❌ NO MATCH")
            print(f"   Best candidate: {best_match}")
            print(f"   Score: {best_score*100:.1f}% (threshold: {threshold*100:.1f}%)")
        else:
            print(f"❌ NO MATCH - No fingerprints found")
        return None, best_score


def list_enrolled():
    """List all enrolled fingerprints"""
    database = load_database()
    
    if len(database) == 0:
        print("\n📋 No fingerprints enrolled")
        return
    
    print("\n" + "="*60)
    print("📋 ENROLLED FINGERPRINTS")
    print("="*60)
    
    for i, (name, data) in enumerate(database.items(), 1):
        num_samples = data.get('num_samples', 0)
        enrolled_at = data.get('enrolled_at', 'Unknown')
        print(f"{i}. {name}")
        print(f"   Samples: {num_samples}")
        print(f"   Enrolled: {enrolled_at}")
    
    print(f"\nTotal: {len(database)} persons")


def delete_fingerprint(person_name):
    """Delete a fingerprint from database"""
    database = load_database()
    
    if person_name not in database:
        print(f"❌ {person_name} not found in database")
        return False
    
    del database[person_name]
    
    if save_database(database):
        print(f"✅ Deleted {person_name} from database")
        return True
    else:
        print("❌ Failed to save database")
        return False


def main():
    """Main function"""
    print(f"\n📡 Listening on {UDP_IP}:{UDP_PORT}")
    print(f"💾 Storage: {STORAGE_DIR}/")
    print(f"🔧 Processor: {'Improved (FAISS)' if use_improved else 'Old (SIFT/ORB)'}")
    print(f"📊 Threshold: {MATCH_THRESHOLD if use_improved else MATCH_THRESHOLD_OLD}")
    
    # Load existing database
    database = load_database()
    if len(database) > 0:
        print(f"✅ Loaded {len(database)} enrolled fingerprints")
    
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    print("\n" + "="*60)
    print("COMMANDS:")
    print("  e - Enroll new fingerprint")
    print("  i - Identify fingerprint")
    print("  l - List enrolled fingerprints")
    print("  d - Delete fingerprint")
    print("  q - Quit")
    print("="*60)
    
    try:
        while True:
            print("\n" + "-"*60)
            command = input("Enter command (e/i/l/d/q): ").strip().lower()
            
            if command == 'q':
                print("\n👋 Goodbye!")
                break
            
            elif command == 'e':
                person_name = input("Enter person name/ID: ").strip()
                if not person_name:
                    print("❌ Name cannot be empty")
                    continue
                
                num_samples = input("Number of samples (default 5): ").strip()
                num_samples = int(num_samples) if num_samples.isdigit() else 5
                
                enroll_fingerprint(sock, person_name, num_samples)
            
            elif command == 'i':
                identify_fingerprint(sock)
            
            elif command == 'l':
                list_enrolled()
            
            elif command == 'd':
                list_enrolled()
                person_name = input("\nEnter name to delete: ").strip()
                if person_name:
                    confirm = input(f"Delete {person_name}? (y/n): ").strip().lower()
                    if confirm == 'y':
                        delete_fingerprint(person_name)
            
            else:
                print("❌ Unknown command")
    
    except KeyboardInterrupt:
        print("\n\n👋 Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sock.close()
        print(f"\n✅ Database saved to {DATABASE_FILE}")


if __name__ == "__main__":
    main()
