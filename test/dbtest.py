# Test script
from db_helper import DatabaseHelper

db = DatabaseHelper(
    host="192.168.1.3",
    database="gym-db",
    user="postgres",
    password="123456",
    mount_point="/mnt/winshare"
)

# Test get client info
client = db.get_client_info(18)
if client:
    print(f"✅ Client: {client['fname']} {client['lname']}")
    print(f"   Locker: {client['locker']}")
    print(f"   Windows path: {client.get('image_path_windows')}")
    print(f"   Linux path: {client.get('image_path')}")
else:
    print("❌ Client not found")

db.close_all_connections()