import cv2
import numpy as np
from db_helper import DatabaseHelper
from rknnlite.api import RKNNLite
import argparse

def enroll_member(client_id, image_path, db, rknn_face_model):
    """Enroll a member by extracting and saving their face embedding"""
    
    # Load and process image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return False
    
    # Resize to 112x112 for ArcFace
    face_img = cv2.resize(img, (112, 112))
    
    # Get embedding
    face_input = np.expand_dims(face_img, 0)
    outputs = rknn_face_model.inference(inputs=[face_input])
    
    if len(outputs) > 0:
        embedding = outputs[0][0].astype(np.float32)
        
        # Save to database
        success = db.save_face_embedding(client_id, embedding)
        
        if success:
            print(f"Successfully enrolled member ID: {client_id}")
            return True
    
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', type=int, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    args = parser.parse_args()
    
    # Initialize database
    db = DatabaseHelper(
        host='192.168.1.X',
        database='gym-db',
        user='postgres',
        password='123456'
    )
    
    # Load face recognition model
    rknn_face = RKNNLite()
    rknn_face.load_rknn("../model/Arcfacefp.rknn")
    rknn_face.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    
    # Enroll member
    enroll_member(args.client_id, args.image_path, db, rknn_face)
    
    # Cleanup
    db.close_all_connections()
    rknn_face.release()