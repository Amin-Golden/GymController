#!/bin/bash
# Fixed PS3 Eye C++ Wrapper Setup
# Handles libusb include path correctly

echo "╔═══════════════════════════════════════════════════════╗"
echo "║   PS3 Eye - Fixed Python Solution (C++ Wrapper)      ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Install dependencies
echo "Step 1: Installing dependencies..."
sudo apt update
sudo apt install -y \
    git \
    cmake \
    build-essential \
    libusb-1.0-0-dev \
    pkg-config \
    libopencv-dev \
    python3-opencv

echo "✓ Dependencies installed"

# Check if demo exists
if [ ! -f "/tmp/PS3Eye-OpenCV-Demo/ps3eye.cpp" ]; then
    echo ""
    echo "Step 2: Setting up PS3Eye demo..."
    
    cd /tmp
    rm -rf PS3Eye-OpenCV-Demo
    git clone https://github.com/ThomasDebrunner/PS3Eye-OpenCV-Demo.git
    cd PS3Eye-OpenCV-Demo
    cmake .
    make
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to build PS3Eye demo"
        exit 1
    fi
    echo "✓ Demo built"
else
    echo ""
    echo "Step 2: PS3Eye demo already exists"
fi

cd /tmp/PS3Eye-OpenCV-Demo

echo ""
echo "Step 3: Creating C wrapper file..."

cat > ps3eye_wrapper.cpp << 'WRAPPER_EOF'
// ps3eye_wrapper.cpp - C wrapper for PS3Eye C++ library
#include "ps3eye.h"
#include <stdlib.h>
#include <string.h>

using namespace ps3eye;

// Global camera reference
static PS3EYECam::PS3EYERef camera = nullptr;
static std::vector<PS3EYECam::PS3EYERef> devices;

extern "C" {

// Initialize and find cameras
int ps3eye_init() {
    devices = PS3EYECam::getDevices();
    if (devices.empty()) {
        return 0;
    }
    camera = devices.at(0);
    return 1;
}

// Start camera with given parameters
int ps3eye_start(int width, int height, int fps) {
    if (camera == nullptr) {
        return 0;
    }
    
    bool success = camera->init(width, height, fps);
    if (!success) {
        return 0;
    }
    
    camera->start();
    return 1;
}

// Get frame data
int ps3eye_get_frame(unsigned char* buffer) {
    if (camera == nullptr || !camera->isStreaming()) {
        return 0;
    }
    
    camera->getFrame(buffer);
    return 1;
}

// Stop camera
void ps3eye_stop() {
    if (camera != nullptr) {
        camera->stop();
    }
}

// Get camera info
void ps3eye_get_size(int* width, int* height) {
    if (camera != nullptr) {
        *width = camera->getWidth();
        *height = camera->getHeight();
    }
}

// Set camera parameters
void ps3eye_set_gain(int value) {
    if (camera != nullptr) {
        camera->setGain(value);
    }
}

void ps3eye_set_exposure(int value) {
    if (camera != nullptr) {
        camera->setExposure(value);
    }
}

void ps3eye_set_autogain(int enable) {
    if (camera != nullptr) {
        camera->setAutogain(enable != 0);
    }
}

} // extern "C"
WRAPPER_EOF

echo "✓ Wrapper created"

echo ""
echo "Step 4: Building wrapper library with correct include paths..."

# Use pkg-config to get the correct libusb include path
LIBUSB_CFLAGS=$(pkg-config --cflags libusb-1.0)
LIBUSB_LIBS=$(pkg-config --libs libusb-1.0)

echo "Using libusb flags: $LIBUSB_CFLAGS"

g++ -shared -fPIC -o libps3eye_wrapper.so \
    ps3eye_wrapper.cpp ps3eye.cpp \
    -I. \
    $LIBUSB_CFLAGS \
    -std=c++11 \
    $LIBUSB_LIBS \
    -lpthread \
    -O3

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Build failed"
    echo ""
    echo "Checking libusb installation..."
    pkg-config --modversion libusb-1.0
    echo ""
    echo "Include paths:"
    pkg-config --cflags libusb-1.0
    exit 1
fi

echo "✓ Library built successfully"

echo ""
echo "Step 5: Installing library..."

sudo cp libps3eye_wrapper.so /usr/local/lib/
sudo ldconfig

echo "✓ Library installed to /usr/local/lib/"

echo ""
echo "Step 6: Creating Python example..."

cat > /tmp/ps3eye_simple.py << 'PYTHON_EOF'
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

PYTHON_EOF

chmod +x /tmp/ps3eye_simple.py

echo "✓ Python example created: /tmp/ps3eye_simple.py"

echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║              SETUP COMPLETE!                          ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""
echo "✅ C++ wrapper library built and installed"
echo "✅ Python example created"
echo ""
echo "Run it:"
echo "  python3 /tmp/ps3eye_simple.py"
echo ""
echo "Or copy to your home directory:"
echo "  cp /tmp/ps3eye_simple.py ~/camera.py"
echo "  python3 ~/camera.py"