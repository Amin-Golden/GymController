#!/usr/bin/env python3
"""
PS3 Eye Camera - Simple Python + OpenCV Example
Uses C wrapper library
"""

import cv2
import numpy as np
import ctypes
from ctypes import *
import sys

print("=" * 60)
print("PS3 Eye Camera - Python + OpenCV")
print("=" * 60)
print()

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
    print("\nTroubleshooting:")
    print("1. Check USB connection: lsusb | grep 1415:2000")
    print("2. Unplug and replug the camera")
    print("3. Check permissions")
    sys.exit(1)

print("✓ Camera detected")

# Start camera
width, height, fps = 640, 480, 60
print(f"Starting camera: {width}x{height} @ {fps}fps...")

if lib.ps3eye_start(width, height, fps) == 0:
    print("❌ Failed to start camera!")
    sys.exit(1)

print("✓ Camera started")
print()
print("Controls:")
print("  q - Quit")
print("  s - Save snapshot")
print("  + - Increase exposure")
print("  - - Decrease exposure")
print()

# Create buffer
buffer_size = width * height * 3
buffer = (c_ubyte * buffer_size)()

snapshot_count = 0
exposure = 120

try:
    while True:
        # Get frame
        if lib.ps3eye_get_frame(buffer) == 0:
            print("Failed to get frame")
            break
        
        # Convert to numpy array
        frame = np.frombuffer(buffer, dtype=np.uint8)
        frame = frame.reshape((height, width, 3))
        
        # Add info overlay
        cv2.putText(frame, f"Exposure: {exposure}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Display
        cv2.imshow('PS3 Eye Camera', frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            snapshot_count += 1
            filename = f'snapshot_{snapshot_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"✓ Saved: {filename}")
        elif key == ord('+') or key == ord('='):
            exposure = min(255, exposure + 10)
            lib.ps3eye_set_exposure(exposure)
            print(f"Exposure: {exposure}")
        elif key == ord('-') or key == ord('_'):
            exposure = max(0, exposure - 10)
            lib.ps3eye_set_exposure(exposure)
            print(f"Exposure: {exposure}")

except KeyboardInterrupt:
    print("\nInterrupted")

finally:
    # Cleanup
    lib.ps3eye_stop()
    cv2.destroyAllWindows()
    print("\n✓ Camera closed")

