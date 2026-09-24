from db_helper import DatabaseHelper
import time

# Initialize database
db = DatabaseHelper(
    host="192.168.1.2",
    database="gym-db",
    user="postgres",
    password="123456",
    mount_point="/mnt/winshare"
)

print("="*60)
print("TESTING LOCKER OPERATIONS")
print("="*60)

# Diagnose initial state
db.diagnose_database_state()

# Test with a real client ID
test_client_id = 1  # Change to a real client ID from your database

print(f"\n📝 Testing with client ID: {test_client_id}")

# Get initial state
print("\n1️⃣ Getting initial client info...")
client_before = db.get_client_info(test_client_id)
if client_before:
    print(f"   Client: {client_before['fname']} {client_before['lname']}")
    print(f"   Current locker: {client_before['locker']}")
else:
    print(f"   ❌ Client {test_client_id} not found!")
    exit(1)

# Assign a locker
print("\n2️⃣ Assigning locker 50...")
result = db.assign_locker_to_client(test_client_id, 50)
print(f"   Result: {result}")

time.sleep(1)

# Verify assignment
print("\n3️⃣ Verifying assignment...")
client_after_assign = db.get_client_info(test_client_id)
if client_after_assign:
    print(f"   Locker after assignment: {client_after_assign['locker']}")
else:
    print("   ❌ Client disappeared!")

# Diagnose after assignment
db.diagnose_database_state(test_client_id)

# Unassign the locker
print("\n4️⃣ Unassigning locker...")
result = db.unassign_locker(test_client_id)
print(f"   Result: {result}")

time.sleep(1)

# Verify unassignment
print("\n5️⃣ Verifying unassignment...")
client_after_unassign = db.get_client_info(test_client_id)
if client_after_unassign:
    print(f"   Locker after unassignment: {client_after_unassign['locker']}")
else:
    print("   ❌ Client disappeared!")

# Final diagnostic
db.diagnose_database_state(test_client_id)

print("\n✅ Test complete!")
db.close_all_connections()