# import os
# from db_helper import DatabaseHelper

# # Initialize database
# db = DatabaseHelper(
#     host="192.168.1.3",
#     database="gym-db",
#     user="postgres",
#     password="123456",
#     mount_point="/mnt/winshare"
# )

# # Test the problematic path
# windows_path = r"E:\Benjamin\Occupation\Gym App\ASAgym\ClientImages\one one.jpg"

# print("="*60)
# print("PATH CONVERSION TEST")
# print("="*60)

# linux_path = db.convert_image_path(windows_path)

# print("\n" + "="*60)
# print("RESULT")
# print("="*60)
# print(f"Windows: {windows_path}")
# print(f"Linux:   {linux_path}")
# print(f"Exists:  {linux_path and os.path.exists(linux_path)}")

# # List mount point contents
# print("\n" + "="*60)
# print("MOUNT POINT CONTENTS")
# print("="*60)
# if os.path.exists("/mnt/winshare"):
#     files = os.listdir("/mnt/winshare")
#     print(f"Files in /mnt/winshare ({len(files)} total):")
#     for f in files[:10]:
#         print(f"  - {f}")
# else:
#     print("❌ /mnt/winshare does not exist!")

# # Test actual client from database
# print("\n" + "="*60)
# print("DATABASE CLIENT TEST")
# print("="*60)
# client = db.get_client_info(18)
# if client:
#     print(f"Client: {client['fname']} {client['lname']}")
#     print(f"Windows path: {client.get('image_path_windows')}")
#     print(f"Linux path: {client.get('image_path')}")
#     if client.get('image_path'):
#         print(f"File exists: {os.path.exists(client['image_path'])}")
# else:
#     print("❌ Client not found")

[Unit]
Description=RetinaFace Detection Service
After=network.target

[Service]
Type=simple
User=orangepi
WorkingDirectory=/home/orangepi/Projects/GymController
ExecStart=/usr/bin/python3 /home/orangepi/Projects/GymController/RetinaFaceL.py --model_path model/retinafacefp.rknn --db_host 192.168.1.3
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target