# import cv2
# print("OpenCV version:", cv2.__version__)
# # print("GStreamer support:", cv2.getBuildInformation())
# print("OpenCV version:", cv2.__version__)
# print("OpenCV path:", cv2.__file__)
# try:
#     # This will show available backends
#     cap = cv2.VideoCapture()
#     print("VideoCapture created")
    
#     # Try to query backend name
#     backend_name = cap.getBackendName()
#     print(f"Default backend: {backend_name}")
    
# except Exception as e:
#     print(f"Error: {e}")

# # Check build info for Python module specifically
# print("\n" + "="*50)
# print("Build Information:")
# print("="*50)
# info = cv2.getBuildInformation()
# # Print only Video I/O section
# lines = info.split('\n')
# print_flag = False
# for line in lines:
#     if 'Video I/O:' in line:
#         print_flag = True
#     if print_flag:
#         print(line)
#         if line.strip() == '' and print_flag:
#             break

import cv2

print("Testing GStreamer...")
cap = cv2.VideoCapture("videotestsrc ! video/x-raw,width=640,height=480 ! videoconvert ! appsink", cv2.CAP_GSTREAMER)

if cap.isOpened():
    print("✓ GStreamer IS WORKING!")
    ret, frame = cap.read()
    if ret:
        print(f"✓ Frame captured: {frame.shape}")
    cap.release()
else:
    print("✗ GStreamer NOT working")