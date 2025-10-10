import psycopg2
import psycopg2.extensions
import numpy as np
import json
import select
from psycopg2 import pool
from datetime import datetime
import threading
import time

class DatabaseHelper:
    def __init__(self, host, database, user, password, port=5432):
        """Initialize database connection pool"""
        try:
            self.host = host
            self.database = database
            self.user = user
            self.password = password
            self.port = port
            
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, 10,
                host=host,
                database=database,
                user=user,
                password=password,
                port=port
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
            connection = self.get_connection()
            cursor = connection.cursor()
            
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
            connection = self.get_connection()
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
            connection = self.get_connection()
            cursor = connection.cursor()
            
            query = """
                SELECT fe.client_id, fe.embedding, c.first_name, c.last_name
                FROM face_embeddings fe
                JOIN clients c ON fe.client_id = c.id
            """
            cursor.execute(query)
            
            results = cursor.fetchall()
            cursor.close()
            
            embeddings = []
            for row in results:
                client_id, embedding_binary, first_name, last_name = row
                embedding = np.frombuffer(embedding_binary, dtype=np.float32)
                embedding = embedding.reshape(embedding_shape)
                embeddings.append({
                    'client_id': client_id,
                    'embedding': embedding,
                    'name': f"{first_name} {last_name}"
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
            connection = self.get_connection()
            cursor = connection.cursor()
            
            query = """
                SELECT id, first_name, last_name, email, phone_number, 
                       locker, image_path
                FROM clients WHERE id = %s
            """
            cursor.execute(query, (client_id,))
            
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return {
                    'id': result[0],
                    'first_name': result[1],
                    'last_name': result[2],
                    'email': result[3],
                    'phone_number': result[4],
                    'locker': result[5],
                    'image_path': result[6]
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
            connection = self.get_connection()
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
            connection = self.get_connection()
            cursor = connection.cursor()
            
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
            connection = self.get_connection()
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
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            query = """
                UPDATE clients 
                SET locker = %s 
                WHERE id = %s
            """
            cursor.execute(query, (locker_number, client_id))
            
            connection.commit()
            cursor.close()
            print(f"✅ Locker {locker_number} assigned to client {client_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error assigning locker: {e}")
            if connection:
                connection.rollback()
            return False
        finally:
            if connection:
                self.return_connection(connection)

    def check_active_membership(self, client_id):
        """Check if client has an active, paid membership"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            query = """
                SELECT m.id, m.start_date, m.end_date, m.is_paid, m.status,
                    c.first_name, c.last_name
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
                    'first_name': result[5],
                    'last_name': result[6],
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
            connection = self.get_connection()
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
            connection = self.get_connection()
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
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            query = """
                UPDATE clients 
                SET locker = NULL 
                WHERE id = %s
                RETURNING locker
            """
            cursor.execute(query, (client_id,))
            result = cursor.fetchone()
            
            connection.commit()
            cursor.close()
            
            if result:
                print(f"✅ Locker unassigned from client {client_id}")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error unassigning locker: {e}")
            if connection:
                connection.rollback()
            return False
        finally:
            if connection:
                self.return_connection(connection)

    def get_current_gym_session(self, client_id):
        """Get current open gym session for a client"""
        connection = None
        try:
            connection = self.get_connection()
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

    def get_today_sessions(self):
        """Get all gym sessions for today"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            query = """
                SELECT gs.id, c.first_name, c.last_name, 
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