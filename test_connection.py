import psycopg2
import numpy as np
from psycopg2 import pool
from datetime import datetime
import io

class DatabaseHelper:
    def __init__(self, host, database, user, password, port=5432):
        """Initialize database connection pool"""
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, 10,  # min and max connections
                host=host,
                database=database,
                user=user,
                password=password,
                port=port
            )
            if self.connection_pool:
                print("Database connection pool created successfully")
        except Exception as e:
            print(f"Error creating connection pool: {e}")
            raise

    def get_connection(self):
        """Get connection from pool"""
        return self.connection_pool.getconn()

    def return_connection(self, connection):
        """Return connection to pool"""
        self.connection_pool.putconn(connection)

    def close_all_connections(self):
        """Close all connections in pool"""
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
            print(f"Face embedding saved for client_id: {client_id}")
            return True
            
        except Exception as e:
            print(f"Error saving face embedding: {e}")
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
                # Convert binary back to numpy array
                embedding = np.frombuffer(result[0], dtype=np.float32)
                embedding = embedding.reshape(embedding_shape)
                return embedding
            return None
            
        except Exception as e:
            print(f"Error retrieving face embedding: {e}")
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
            print(f"Error retrieving all face embeddings: {e}")
            return []
        finally:
            if connection:
                self.return_connection(connection)

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
            print(f"Error logging access: {e}")
            if connection:
                connection.rollback()
        finally:
            if connection:
                self.return_connection(connection)

    def get_client_by_id(self, client_id):
        """Get client information by ID"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            query = """
                SELECT id, first_name, last_name, email, phone_number, locker
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
                    'locker': result[5]
                }
            return None
            
        except Exception as e:
            print(f"Error getting client: {e}")
            return None
        finally:
            if connection:
                self.return_connection(connection)