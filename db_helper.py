import psycopg2
import psycopg2.extensions
import numpy as np
import json
import select
from psycopg2 import pool
from datetime import datetime
import threading
import time
import os

class DatabaseHelper:
    def __init__(self, host, database, user, password, port=5432, mount_point="/mnt/winshare"):
        """Initialize database connection pool"""
        try:
            self.host = host
            self.database = database
            self.user = user
            self.password = password
            self.port = port
            self.mount_point = mount_point
            self.windows_image_base = r"C:\ASAgym\ClientImages"

            
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                5, 50,
                host=host,
                database=database,
                user=user,
                password=password,
                port=port,
                connect_timeout=5,  # ← Add this
                options='-c statement_timeout=3000'  # ← Add this
            )
            if self.connection_pool:
                print("✅ Database connection pool created successfully")
                
            # Separate connection for listening to notifications
            self.listen_connection = None
            self.listening = False
            self.notification_callback = None
            
        except Exception as e:
            print(f"❌ Error creating connection pool: {e}")
            raise

    def convert_image_path(self, windows_path):
        """
        Convert Windows image path to Linux mounted path
        
        Example:
        Windows: E:\Benjamin\Occupation\Gym App\ASAgym\Gym.Desktop\bin\Debug\net8.0-windows\ClientImages\amin abaz.jpg
        Linux:   /mnt/winshare/amin abaz.jpg
        """
        if not windows_path:
            return None
        
        try:
            # Remove any quotes from path
            windows_path = windows_path.strip("'\"")
            
            # Normalize path separators to backslash
            windows_path = windows_path.replace('/', '\\')
            
            # print(f"🔄 Converting path:")
            # print(f"   Input: {windows_path}")
            # print(f"   Base: {self.windows_image_base}")
            
            # Method 1: Try to remove base path
            if self.windows_image_base in windows_path:
                relative_path = windows_path.replace(self.windows_image_base, "")
                # Remove leading backslashes
                relative_path = relative_path.lstrip('\\')
                # print(f"   Relative: {relative_path}")
            else:
                # Method 2: Just extract filename
                relative_path = os.path.basename(windows_path)
                # print(f"   Using filename only: {relative_path}")
            
            # Convert backslashes to forward slashes for Linux
            relative_path = relative_path.replace('\\', '/')
            
            # Combine with mount point
            linux_path = os.path.join(self.mount_point, relative_path)
            
            # print(f"   Output: {linux_path}")
            
            # Verify file exists
            if os.path.exists(linux_path):
                # print(f"   ✅ File exists!")
                return linux_path
            else:
                print(f"   ❌ File not found!")
                
                # Try listing directory to debug
                mount_dir = os.path.dirname(linux_path)
                if os.path.exists(mount_dir):
                    files = os.listdir(mount_dir)
                    # print(f"   📁 Files in {mount_dir}:")
                    for f in files[:5]:  # Show first 5 files
                        print(f"      - {f}")
                
                return None
            
        except Exception as e:
            print(f"❌ Error converting path: {e}")
            import traceback
            traceback.print_exc()
            return None
    

    def get_connection(self):
        """Get connection from pool"""
        return self.connection_pool.getconn()

    def return_connection(self, connection):
        """Return connection to pool"""
        self.connection_pool.putconn(connection)

    def close_all_connections(self):
        """Close all connections in pool"""
        if self.listen_connection:
            self.stop_listening()
        self.connection_pool.closeall()

    def save_face_embedding(self, client_id, embedding, confidence=None):
        """Save face embedding for a client"""
        connection = None
        try:
            connection = self.get_connection_with_timeout()
            cursor = connection.cursor()
            confidence = self.convert_numpy_types(confidence)

            # Convert numpy array to binary
            embedding_binary = embedding.tobytes()
            
            # Insert or update embedding
            query = """
                INSERT INTO face_embeddings (client_id, embedding, confidence, updated_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (client_id) 
                DO UPDATE SET 
                    embedding = EXCLUDED.embedding,
                    confidence = EXCLUDED.confidence,
                    updated_at = EXCLUDED.updated_at
            """
            
            cursor.execute(query, (
                client_id, 
                psycopg2.Binary(embedding_binary),
                confidence,
                datetime.now()
            ))
            
            connection.commit()
            cursor.close()
            print(f"✅ Face embedding saved for client_id: {client_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving face embedding: {e}")
            if connection:
                connection.rollback()
            return False
        finally:
            if connection:
                self.return_connection(connection)

    def get_face_embedding(self, client_id, embedding_shape):
        """Retrieve face embedding for a specific client"""
        connection = None
        try:
            connection = self.get_connection_with_timeout()
            cursor = connection.cursor()
            
            query = "SELECT embedding FROM face_embeddings WHERE client_id = %s"
            cursor.execute(query, (client_id,))
            
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                embedding = np.frombuffer(result[0], dtype=np.float32)
                embedding = embedding.reshape(embedding_shape)
                return embedding
            return None
            
        except Exception as e:
            print(f"❌ Error retrieving face embedding: {e}")
            return None
        finally:
            if connection:
                self.return_connection(connection)

    def get_all_face_embeddings(self, embedding_shape):
        """Retrieve all face embeddings with client IDs"""
        connection = None
        try:
            connection = self.get_connection_with_timeout()
            cursor = connection.cursor()
            
            query = """
                SELECT fe.client_id, fe.embedding, c.fname, c.lname
                FROM face_embeddings fe
                JOIN clients c ON fe.client_id = c.id
            """
            cursor.execute(query)
            
            results = cursor.fetchall()
            cursor.close()
            
            embeddings = []
            for row in results:
                client_id, embedding_binary, fname, lname = row
                embedding = np.frombuffer(embedding_binary, dtype=np.float32)
                embedding = embedding.reshape(embedding_shape)
                embeddings.append({
                    'client_id': client_id,
                    'embedding': embedding,
                    'name': f"{fname} {lname}"
                })
            
            return embeddings
            
        except Exception as e:
            print(f"❌ Error retrieving all face embeddings: {e}")
            return []
        finally:
            if connection:
                self.return_connection(connection)

    def get_client_info(self, client_id):
        """Get client information including image path"""
        connection = None
        try:
            connection = self.get_connection_with_timeout()
            cursor = connection.cursor()
            
            query = """
                SELECT id, fname, lname, email, phone_number, 
                       locker , image_path
                FROM clients WHERE id = %s
            """
            cursor.execute(query, (client_id,))
            
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                windows_path = result[6] if len(result) > 6 else None
                # linux_path = self.convert_image_path(windows_path)
                try:
                    linux_path = self.convert_image_path(windows_path) if windows_path else None
                except Exception as path_error:
                    print(f"⚠️  Path conversion failed for client {client_id}: {path_error}")
                    linux_path = None
            
                return {
                    'id': result[0],
                    'fname': result[1],
                    'lname': result[2],
                    'email': result[3],
                    'phone_number': result[4],
                    'locker': result[5],
                    'image_path': linux_path,  # Converted path
                    'image_path_original': windows_path  # Keep original
                }
            return None
            
        except Exception as e:
            print(f"❌ Error getting client: {e}")
            return None
        finally:
            if connection:
                self.return_connection(connection)

    def check_embedding_exists(self, client_id):
        """Check if face embedding exists for a client"""
        connection = None
        try:
            connection = self.get_connection_with_timeout()
            cursor = connection.cursor()
            
            query = "SELECT COUNT(*) FROM face_embeddings WHERE client_id = %s"
            cursor.execute(query, (client_id,))
            
            result = cursor.fetchone()
            cursor.close()
            
            return result[0] > 0
            
        except Exception as e:
            print(f"❌ Error checking embedding: {e}")
            return False
        finally:
            if connection:
                self.return_connection(connection)

    def start_listening(self, callback):
        """Start listening for database notifications"""
        self.notification_callback = callback
        self.listening = True
        
        # Create dedicated connection for listening
        self.listen_connection = psycopg2.connect(
            host=self.host,
            database=self.database,
            user=self.user,
            password=self.password,
            port=self.port
        )
        self.listen_connection.set_isolation_level(
            psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT
        )
        
        cursor = self.listen_connection.cursor()
        cursor.execute("LISTEN client_changes;")
        cursor.close()
        
        print("🎧 Started listening for database changes...")
        
        # Start listening thread
        listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        listen_thread.start()

        
    def get_all_clients_for_enrollment(self):
        """Get all clients that need face embedding enrollment"""
        connection = None
        try:
            connection = self.get_connection_with_timeout()
            cursor = connection.cursor()
            
            query = """
                SELECT c.id, c.fname, c.lname, c.image_path
                FROM clients c
                LEFT JOIN face_embeddings fe ON c.id = fe.client_id
                WHERE fe.id IS NULL AND c.image_path IS NOT NULL
            """
            cursor.execute(query)
            
            results = cursor.fetchall()
            cursor.close()
            
            clients_to_enroll = []
            for row in results:
                client_id, fname, lname, windows_path = row
                linux_path = self.convert_image_path(windows_path)
                
                if linux_path:  # Only add if path conversion succeeded
                    clients_to_enroll.append({
                        'client_id': client_id,
                        'name': f"{fname} {lname}",
                        'image_path': linux_path
                    })
            
            return clients_to_enroll
            
        except Exception as e:
            print(f"❌ Error getting clients for enrollment: {e}")
            return []
        finally:
            if connection:
                self.return_connection(connection)

    def _listen_loop(self):
        """Background loop to listen for notifications"""
        while self.listening and self.listen_connection:
            try:
                if select.select([self.listen_connection], [], [], 1.0) != ([], [], []):
                    self.listen_connection.poll()
                    while self.listen_connection.notifies:
                        notify = self.listen_connection.notifies.pop(0)
                        
                        # Parse notification payload
                        try:
                            payload = json.loads(notify.payload)
                            print(f"📬 Database notification received: {payload}")
                            
                            if self.notification_callback:
                                self.notification_callback(payload)
                        except json.JSONDecodeError as e:
                            print(f"❌ Error parsing notification: {e}")
                            
            except Exception as e:
                print(f"❌ Error in listen loop: {e}")
                time.sleep(1)

    def stop_listening(self):
        """Stop listening for notifications"""
        self.listening = False
        if self.listen_connection:
            try:
                cursor = self.listen_connection.cursor()
                cursor.execute("UNLISTEN client_changes;")
                cursor.close()
                self.listen_connection.close()
                print("🔇 Stopped listening for database changes")
            except Exception as e:
                print(f"❌ Error stopping listener: {e}")

    def log_access(self, client_id, access_granted, confidence=None):
        """Log gym access attempts"""
        connection = None
        try:
            connection = self.get_connection_with_timeout()
            cursor = connection.cursor()
            confidence = self.convert_numpy_types(confidence)

            # Create access log table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS access_logs (
                    id SERIAL PRIMARY KEY,
                    client_id BIGINT,
                    access_granted BOOLEAN,
                    confidence FLOAT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            query = """
                INSERT INTO access_logs (client_id, access_granted, confidence)
                VALUES (%s, %s, %s)
            """
            cursor.execute(query, (client_id, access_granted, confidence))
            
            connection.commit()
            cursor.close()
            
        except Exception as e:
            print(f"❌ Error logging access: {e}")
            if connection:
                connection.rollback()
        finally:
            if connection:
                self.return_connection(connection)

    def get_available_locker(self, total_lockers=200):
        """Find the first available locker number"""
        connection = None
        try:
            connection = self.get_connection_with_timeout()
            cursor = connection.cursor()
            
            # Get all assigned lockers
            query = """
                SELECT locker FROM clients 
                WHERE locker IS NOT NULL AND locker >= 0
            """
            cursor.execute(query)
            
            assigned_lockers = set(row[0] for row in cursor.fetchall())
            cursor.close()
            
            # Find first available locker
            for i in range(total_lockers):
                if i not in assigned_lockers:
                    return i
            
            return None  # No available lockers
            
        except Exception as e:
            print(f"❌ Error getting available locker: {e}")
            return None
        finally:
            if connection:
                self.return_connection(connection)

    def assign_locker_to_client(self, client_id, locker_number):
        """Assign a locker to a client"""
        connection = None
        cursor = None
        try:
            # Validate inputs
            if client_id is None or client_id <= 0:
                print(f"❌ Invalid client_id: {client_id}")
                return False
            
            if locker_number is None or locker_number < 0:
                print(f"❌ Invalid locker_number: {locker_number}")
                return False
            
            client_id = int(client_id)
            locker_number = int(locker_number)
            
            connection = self.get_connection_with_timeout()
            cursor = connection.cursor()
            print(f"🔍 Assigning locker {locker_number} to client {client_id}...")
            cursor.execute("SELECT id, fname, lname FROM clients WHERE id = %s", (client_id,))
            client = cursor.fetchone()

            if not client:
                print(f"❌ Client {client_id} not found")
                return False
            
            print(f"📋 Found client: {client[1]} {client[2]} (ID: {client[0]})")
            
            # Check if locker is already assigned to another client
            cursor.execute("SELECT id, fname, lname FROM clients WHERE locker = %s AND id != %s", 
                        (locker_number, client_id))
            existing = cursor.fetchone()
            
            if existing:
                print(f"⚠️  Locker {locker_number} is already assigned to client {existing[0]} ({existing[1]} {existing[2]})")
                # Optionally, you could unassign it first
                # cursor.execute("UPDATE clients SET locker = NULL WHERE id = %s", (existing[0],))
            
            # Assign the locker
            query = """
                UPDATE clients 
                SET locker = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """
            
            cursor.execute(query, (locker_number, client_id))
            rows_affected = cursor.rowcount
            
            # Verify
            cursor.execute("SELECT id, locker FROM clients WHERE id = %s", (client_id,))
            verify = cursor.fetchone()
            print(f"✓ Verification: Client {verify[0]} now has locker = {verify[1]}")
            
            connection.commit()
            
            if rows_affected > 0:
                print(f"✅ Locker {locker_number} assigned to client {client_id}")
                return True
            else:
                print(f"⚠️  No rows updated")
                return False
            
        except Exception as e:
            print(f"❌ Error assigning locker: {e}")
            import traceback
            traceback.print_exc()
            if connection:
                try:
                    connection.rollback()
                except:
                    pass
            return False
        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if connection:
                self.return_connection(connection)

            
    def get_membership_summary(self, client_id):
        """Get detailed membership summary for display"""
        connection = None
        try:
            connection = self.get_connection_with_timeout()
            cursor = connection.cursor()
            
            query = """
                SELECT 
                    c.fname,
                    c.lname,
                    m.remainsessions,
                    m.status,
                    m.start_date,
                    m.end_date,
                    p.packagename,
                    m.is_paid
                FROM clients c
                LEFT JOIN memberships m ON c.id = m.client_id  
                    AND m.status = 'Active'
                    AND m.end_date >= CURRENT_DATE
                LEFT JOIN packages p ON m.package_id = p.id
                WHERE c.id = %s
                ORDER BY m.id DESC
                LIMIT 1
            """
            
            cursor.execute(query, (client_id,))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                fname, lname, sessions, status, start, end, package, paid = result
                return {
                    'name': f"{fname} {lname}",
                    'sessions': sessions if sessions is not None else 0,
                    'status': status if status else 'No Active Membership',
                    'start_date': start,
                    'end_date': end,
                    'package': package if package else 'N/A',
                    'is_paid': paid if paid is not None else False
                }
            return None
            
        except Exception as e:
            print(f"❌ Error getting membership summary: {e}")
            return None
        finally:
            if connection:
                self.return_connection(connection)
    def decrease_membership_session(self, client_id):
        """
        Decrease remaining sessions by 1 when client enters gym
        Returns: True if successful, False otherwise
        """
        connection = None
        try:
            connection = self.get_connection_with_timeout()
            cursor = connection.cursor()
            
            # First check current sessions
            check_query = """
                SELECT id, remainsessions, fname, lname
                FROM (
                    SELECT m.id, m.remainsessions, c.fname, c.lname
                    FROM memberships m
                    JOIN clients c ON m.client_id  = c.id
                    WHERE m.client_id  = %s 
                        AND m.status = 'Active'
                        AND m.end_date >= CURRENT_DATE
                        AND m.is_paid = true
                    ORDER BY m.id DESC
                    LIMIT 1
                ) AS active_membership
            """
            
            cursor.execute(check_query, (client_id,))
            result = cursor.fetchone()
            
            if not result:
                print(f"❌ No active membership to decrease sessions")
                cursor.close()
                return False
            
            membership_id, current_sessions, fname, lname = result
            
            if current_sessions <= 0:
                print(f"❌ {fname} {lname} has no remaining sessions!")
                cursor.close()
                return False
            
            # Decrease session count
            update_query = """
                UPDATE memberships 
                SET remainsessions = remainsessions - 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                RETURNING remainsessions
            """
            
            cursor.execute(update_query, (membership_id,))
            new_session_count = cursor.fetchone()[0]
            
            connection.commit()
            cursor.close()
            
            print(f"✅ Session decreased for {fname} {lname}")
            print(f"   Previous: {current_sessions} → New: {new_session_count}")
            
            # Check if sessions exhausted
            if new_session_count == 0:
                print(f"⚠️  WARNING: {fname} {lname} has used all sessions!")
                print(f"   Please renew membership")
            
            return True
            
        except Exception as e:
            print(f"❌ Error decreasing membership session: {e}")
            import traceback
            traceback.print_exc()
            if connection:
                connection.rollback()
            return False
        finally:
            if connection:
                self.return_connection(connection)
    def check_membership_sessions(self, client_id):
        """
        Check if client has remaining sessions in their active membership
        Returns: dict with session info or None
        """
        connection = None
        try:
            connection = self.get_connection_with_timeout()
            cursor = connection.cursor()
            
            query = """
                SELECT 
                    m.id as membership_id,
                    m.remainsessions,
                    m.status,
                    m.start_date,
                    m.end_date,
                    m.is_paid,
                    c.fname,
                    c.lname
                FROM memberships m
                JOIN clients c ON m.client_id  = c.id
                WHERE m.client_id  = %s 
                    AND m.status = 'Active'
                    AND m.end_date >= CURRENT_DATE
                    AND m.is_paid = true
                ORDER BY m.id DESC
                LIMIT 1
            """
            
            cursor.execute(query, (client_id,))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                membership_id, remainsessions , status, start_date, end_date, is_paid, fname, lname = result
                
                print(f"📋 Membership check for {fname} {lname}:")
                print(f"   Remaining sessions: {remainsessions }")
                print(f"   Status: {status}")
                print(f"   Valid until: {end_date}")
                
                return {
                    'membership_id': membership_id,
                    'remain_sessions': remainsessions ,
                    'status': status,
                    'start_date': start_date,
                    'end_date': end_date,
                    'is_paid': is_paid,
                    'has_access': remainsessions  > 0
                }
            else:
                print(f"❌ No active membership found for client {client_id}")
                return None
            
        except Exception as e:
            print(f"❌ Error checking membership sessions: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if connection:
                self.return_connection(connection)

    def check_active_membership(self, client_id):
        """Check if client has an active, paid membership"""
        connection = None
        try:
            connection = self.get_connection_with_timeout()
            cursor = connection.cursor()
            
            query = """
                SELECT m.id, m.start_date, m.end_date, m.is_paid, m.status,
                    c.fname, c.lname
                FROM memberships m
                JOIN clients c ON m.client_id = c.id
                WHERE m.client_id = %s 
                    AND m.is_paid = TRUE 
                    AND m.end_date >= CURRENT_DATE
                ORDER BY m.end_date DESC
                LIMIT 1
            """
            cursor.execute(query, (client_id,))
            
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return {
                    'membership_id': result[0],
                    'start_date': result[1],
                    'end_date': result[2],
                    'is_paid': result[3],
                    'status': result[4],
                    'fname': result[5],
                    'lname': result[6],
                    'is_active': True
                }
            return None
            
        except Exception as e:
            print(f"❌ Error checking membership: {e}")
            return None
        finally:
            if connection:
                self.return_connection(connection)

    def record_entrance(self, client_id, locker_number):
        """Record gym entrance time and locker assignment"""
        connection = None
        try:
            connection = self.get_connection_with_timeout()
            cursor = connection.cursor()
            
            # Check if there's an open session (no exit_time)
            cursor.execute("""
                SELECT id FROM gym_sessions 
                WHERE client_id = %s AND exit_time IS NULL
                ORDER BY entrance_time DESC
                LIMIT 1
            """, (client_id,))
            
            existing_session = cursor.fetchone()
            
            if existing_session:
                print(f"⚠️  Client {client_id} already has an open session")
                cursor.close()
                return False
            
            # Insert new entrance record
            query = """
                INSERT INTO gym_sessions (client_id, entrance_time, locker_number)
                VALUES (%s, CURRENT_TIMESTAMP, %s)
                RETURNING id, entrance_time
            """
            cursor.execute(query, (client_id, locker_number))
            result = cursor.fetchone()
            
            connection.commit()
            cursor.close()
            
            if result:
                print(f"✅ Entrance recorded for client {client_id} at {result[1]}")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error recording entrance: {e}")
            if connection:
                connection.rollback()
            return False
        finally:
            if connection:
                self.return_connection(connection)

    def record_exit(self, client_id):
        """Record gym exit time"""
        connection = None
        try:
            connection = self.get_connection_with_timeout()
            cursor = connection.cursor()
            
            # Find the open session
            query = """
                UPDATE gym_sessions 
                SET exit_time = CURRENT_TIMESTAMP
                WHERE client_id = %s 
                    AND exit_time IS NULL
                RETURNING id, entrance_time, exit_time, locker_number
            """
            cursor.execute(query, (client_id,))
            result = cursor.fetchone()
            
            connection.commit()
            cursor.close()
            
            if result:
                session_id, entrance, exit_time, locker = result
                duration = exit_time - entrance
                print(f"✅ Exit recorded for client {client_id}. Duration: {duration}")
                return True
            else:
                print(f"⚠️  No open session found for client {client_id}")
                return False
            
        except Exception as e:
            print(f"❌ Error recording exit: {e}")
            if connection:
                connection.rollback()
            return False
        finally:
            if connection:
                self.return_connection(connection)

    def unassign_locker(self, client_id):
        """Unassign locker from a client"""
        connection = None
        cursor = None
        try:
            # Validate client_id
            if client_id is None or client_id <= 0:
                print(f"❌ Invalid client_id: {client_id}")
                return False
            
            client_id = int(client_id)

            connection = self.get_connection_with_timeout()
            cursor = connection.cursor()
            print(f"🔍 Unassigning locker for client {client_id}...")
            # First, check if client exists and has a locker assigned
            check_query = "SELECT id, fname, lname, locker FROM clients WHERE id = %s"
           
            cursor.execute(check_query, (client_id,))
            current = cursor.fetchone()
            
            if not current:
                print(f"❌ Client {client_id} not found")
                cursor.close()
                return False
            
            client_id_db, fname, lname, previous_locker = current
            
            if previous_locker is None:
                print(f"ℹ️  Client {client_id} doesn't have a locker assigned")
                cursor.close()
                return True  # Already unassigned, return success
            
            # Now unassign the locker
            update_query = """
                UPDATE clients 
                SET locker = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """
            print(f"🔄 Executing UPDATE for client {client_id}...")

            cursor.execute(update_query, (client_id,))
            rows_affected = cursor.rowcount
            print(f"📊 Rows affected: {rows_affected}")
            # Verify the update
            cursor.execute("SELECT id, locker FROM clients WHERE id = %s", (client_id,))
            verify = cursor.fetchone()
            print(f"✓ Verification: Client {verify[0]} now has locker = {verify[1]}")
            
            # Commit the transaction
            connection.commit()
            
            if rows_affected > 0:
                print(f"✅ Locker {previous_locker} unassigned from client {client_id}")
                return True
            else:
                print(f"⚠️  No rows updated for client {client_id}")
                return False
            
        except Exception as e:
            print(f"❌ Error unassigning locker from client {client_id}: {e}")
            import traceback
            traceback.print_exc()
            if connection:
                try:
                    connection.rollback()
                    print("↩️  Transaction rolled back")
                except:
                    pass
            return False
        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if connection:
                self.return_connection(connection)

    def get_current_gym_session(self, client_id):
        """Get current open gym session for a client"""
        connection = None
        try:
            connection = self.get_connection_with_timeout()
            cursor = connection.cursor()
            
            query = """
                SELECT id, entrance_time, locker_number
                FROM gym_sessions 
                WHERE client_id = %s AND exit_time IS NULL
                ORDER BY entrance_time DESC
                LIMIT 1
            """
            cursor.execute(query, (client_id,))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return {
                    'session_id': result[0],
                    'entrance_time': result[1],
                    'locker_number': result[2]
                }
            return None
            
        except Exception as e:
            print(f"❌ Error getting current session: {e}")
            return None
        finally:
            if connection:
                self.return_connection(connection)    

    def diagnose_database_state(self, client_id=None):
        """Diagnose database state - useful for debugging"""
        connection = None
        try:
            connection = self.get_connection_with_timeout()
            cursor = connection.cursor()
            
            print("\n" + "="*60)
            print("🔍 DATABASE DIAGNOSTIC")
            print("="*60)
            
            # Total clients
            cursor.execute("SELECT COUNT(*) FROM clients")
            total = cursor.fetchone()[0]
            print(f"Total clients in database: {total}")
            
            # Clients with lockers
            cursor.execute("SELECT COUNT(*) FROM clients WHERE locker IS NOT NULL")
            with_lockers = cursor.fetchone()[0]
            print(f"Clients with lockers: {with_lockers}")
            
            # Clients without lockers
            print(f"Clients without lockers: {total - with_lockers}")
            
            if client_id:
                print(f"\nSpecific client {client_id}:")
                cursor.execute("""
                    SELECT id, fname, lname, locker, image_path
                    FROM clients WHERE id = %s
                """, (client_id,))
                client = cursor.fetchone()
                if client:
                    print(f"  ID: {client[0]}")
                    print(f"  Name: {client[1]} {client[2]}")
                    print(f"  Locker: {client[3]}")
                    print(f"  Image: {client[4]}")
                else:
                    print(f"  ❌ Client not found!")
            
            # Show first 5 clients
            print("\nFirst 5 clients:")
            cursor.execute("""
                SELECT id, fname, lname, locker 
                FROM clients 
                ORDER BY id 
                LIMIT 5
            """)
            for row in cursor.fetchall():
                print(f"  ID={row[0]}, Name={row[1]} {row[2]}, Locker={row[3]}")
            
            cursor.close()
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"❌ Diagnostic error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if connection:
                self.return_connection(connection)

    def get_today_sessions(self):

        """Get all gym sessions for today"""
        connection = None
        try:
            connection = self.get_connection_with_timeout()
            cursor = connection.cursor()
            
            query = """
                SELECT gs.id, c.fname, c.lname, 
                    gs.entrance_time, gs.exit_time, gs.locker_number,
                    CASE 
                        WHEN gs.exit_time IS NULL THEN 'ACTIVE'
                        ELSE 'COMPLETED'
                    END as status
                FROM gym_sessions gs
                JOIN clients c ON gs.client_id = c.id
                WHERE DATE(gs.entrance_time) = CURRENT_DATE
                ORDER BY gs.entrance_time DESC
            """
            cursor.execute(query)
            
            results = cursor.fetchall()
            cursor.close()
            
            sessions = []
            for row in results:
                sessions.append({
                    'session_id': row[0],
                    'name': f"{row[1]} {row[2]}",
                    'entrance': row[3],
                    'exit': row[4],
                    'locker': row[5],
                    'status': row[6]
                })
            
            return sessions
            
        except Exception as e:
            print(f"❌ Error getting today's sessions: {e}")
            return []
        finally:
            if connection:
                self.return_connection(connection)

    def delete_face_embedding(self, client_id):
        """Delete face embedding for a client"""
        connection = None
        try:
            connection = self.get_connection_with_timeout()
            cursor = connection.cursor()
            
            query = "DELETE FROM face_embeddings WHERE client_id = %s"
            cursor.execute(query, (client_id,))
            
            connection.commit()
            cursor.close()
            print(f"✅ Face embedding deleted for client_id: {client_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting face embedding: {e}")
            if connection:
                connection.rollback()
            return False
        finally:
            if connection:
                self.return_connection(connection)


    def regenerate_face_embedding(self, client_id, face_recognition_model):
        """Regenerate face embedding for a client after image update"""
        print(f"🔄 Regenerating face embedding for client {client_id}...")
        
        # Get client info
        client_info = self.get_client_info(client_id)
        if not client_info:
            print(f"❌ Client {client_id} not found")
            return False
        
        image_path = client_info.get('image_path')
        if not image_path:
            print(f"⚠️  No image path for client {client_id}")
            return False
        
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return False
        
        try:
            # Import required libraries (assuming you have face recognition code)
            import cv2
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Failed to read image: {image_path}")
                return False
            
            # Generate embedding using your face recognition model
            # This is a placeholder - replace with your actual face recognition code
            embedding = face_recognition_model.generate_embedding(image)
            
            if embedding is None:
                print(f"❌ Failed to generate embedding for client {client_id}")
                return False
            
            # Save new embedding
            success = self.save_face_embedding(client_id, embedding, confidence=0.95)
            
            if success:
                print(f"✅ Face embedding regenerated for client {client_id}")
            
            return success
            
        except Exception as e:
            print(f"❌ Error regenerating face embedding: {e}")
            import traceback
            traceback.print_exc()
            return False


    def handle_client_update_notification(self, payload, face_recognition_model):
        """
        Handle client update notifications and regenerate embeddings if needed
        
        Args:
            payload: Notification payload from database
            face_recognition_model: Your face recognition model instance
        """
        try:
            action = payload.get('action')
            client_id = payload.get('client_id')
            
            print(f"📬 Handling {action} notification for client {client_id}")
            
            if action == 'INSERT':
                # New client - generate embedding
                image_path = payload.get('image_path')
                if image_path:
                    print(f"🆕 New client {client_id} - will generate embedding")
                    self.regenerate_face_embedding(client_id, face_recognition_model)
            
            elif action == 'UPDATE':
                # Check if image changed
                image_changed = payload.get('image_changed', False)
                
                if image_changed:
                    print(f"🔄 Client {client_id} image changed - regenerating embedding")
                    # Delete old embedding first
                    self.delete_face_embedding(client_id)
                    # Generate new embedding
                    self.regenerate_face_embedding(client_id, face_recognition_model)
                else:
                    print(f"ℹ️  Client {client_id} updated but image unchanged - skipping embedding update")
                    print(f"🔄 Client {client_id} image changed - regenerating embedding")
                    # Delete old embedding first
                    self.delete_face_embedding(client_id)
                    # Generate new embedding
                    self.regenerate_face_embedding(client_id, face_recognition_model)
            
            elif action == 'DELETE':
                # Client deleted - remove embedding
                print(f"🗑️  Client {client_id} deleted - removing embedding")
                self.delete_face_embedding(client_id)
        
        except Exception as e:
            print(f"❌ Error handling notification: {e}")
            import traceback
            traceback.print_exc()


    def convert_numpy_types(self, value):
        """Convert numpy types to native Python types"""
        import numpy as np
        
        if isinstance(value, (np.float32, np.float64)):
            return float(value)
        elif isinstance(value, (np.int32, np.int64, np.int8, np.int16)):
            return int(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, np.bool_):
            return bool(value)
        else:
            return value
        
    def get_connection_with_timeout(self, timeout=1.0):
        """Get connection with timeout to prevent infinite blocking"""
        import threading
        
        result = [None]
        exception = [None]
        
        def get_conn():
            try:
                result[0] = self.connection_pool.getconn()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=get_conn)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            print(f"⚠️ Connection pool timeout after {timeout}s!")
            return None
        
        if exception[0]:
            raise exception[0]
        
        return result[0]

    # Then update all get_connection() calls:
    # OLD: connection = self.get_connection()
    # NEW: connection = self.get_connection_with_timeout(timeout=2.0)
    #      if connection is None:
    #          print("Failed to get database connection!")
    #          return None  # or handle appropriately