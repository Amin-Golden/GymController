import os
import sys
import time
import numpy as np
import argparse
import cv2
from math import ceil
from itertools import product as product
import math
from rknnlite.api import RKNNLite
import socket
import threading
from queue import Queue

# Import database helper
from db_helper import DatabaseHelper

# Keep all your existing functions (letterbox_resize, PriorBox, box_decode, etc.)
def letterbox_resize(image, size, bg_color):
    """letterbox_resize the image according to the specified size"""
    if isinstance(image, str):
        image = cv2.imread(image)

    target_width, target_height = size
    image_height, image_width, _ = image.shape

    aspect_ratio = min(target_width / image_width, target_height / image_height)
    new_width = int(image_width * aspect_ratio)
    new_height = int(image_height * aspect_ratio)

    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    result_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * bg_color
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    result_image[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = image
    return result_image, aspect_ratio, offset_x, offset_y

def PriorBox(image_size):
    anchors = []
    min_sizes = [[16, 32], [64, 128], [256, 512]]
    steps = [8, 16, 32]
    feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps]
    for k, f in enumerate(feature_maps):
        min_sizes_ = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes_:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]
    output = np.array(anchors).reshape(-1, 4)
    return output

def box_decode(loc, priors):
    variances = [0.1, 0.2]
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors):
    variances = [0.1, 0.2]
    landmarks = np.concatenate((
        priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:]
    ), axis=1)
    return landmarks

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def Similarity(alpha_embedding, beta_embedding):
    alpha = np.array(alpha_embedding)
    beta = np.array(beta_embedding)
    
    dot_product = np.dot(alpha, beta)
    norm_alpha = np.linalg.norm(alpha)
    norm_beta = np.linalg.norm(beta)
    cosine_similarity = dot_product / (norm_alpha * norm_beta)
    
    return cosine_similarity


class FaceEnrollmentProcessor:
    """Handles automatic face enrollment when new members are added"""
    
    def __init__(self, db_helper, rknn_face, face_model_size=(112, 112)):
        self.db = db_helper
        self.rknn_face = rknn_face
        self.face_model_size = face_model_size
        self.processing_queue = Queue()
        self.is_running = True
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.process_thread.start()
        
    def enqueue_client(self, client_id, image_path):
        """Add client to processing queue"""
        self.processing_queue.put((client_id, image_path))
        print(f"📋 Queued client {client_id} for embedding extraction")
        
    def _process_queue(self):
        """Background thread to process enrollment queue"""
        while self.is_running:
            try:
                if not self.processing_queue.empty():
                    client_id, image_path = self.processing_queue.get()
                    self._process_client(client_id, image_path)
                else:
                    time.sleep(0.5)
            except Exception as e:
                print(f"❌ Error in processing queue: {e}")
                
    def _process_client(self, client_id, image_path):
        """Process a single client's face image"""
        try:
            print(f"🔄 Processing client {client_id}...")
            
            # Check if image path exists
            if not os.path.exists(image_path):
                print(f"⚠️  Image not found: {image_path}")
                return
            
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Failed to read image: {image_path}")
                return
            
            # Process image for face detection
            model_height, model_width = (320, 320)
            letterbox_img, aspect_ratio, offset_x, offset_y = letterbox_resize(
                img, (model_height, model_width), 114
            )
            infer_img = np.expand_dims(letterbox_img, 0)
            
            # Detect face (you need to pass rknn for detection)
            # For now, we'll assume the image contains a face and process the whole face
            # You may want to add face detection here using your RetinaFace model
            
            # Resize for ArcFace
            face_img = cv2.resize(img, self.face_model_size)
            letterbox_face, _, _, _ = letterbox_resize(
                face_img, self.face_model_size, 114
            )
            
            # Extract embedding
            face_input = np.expand_dims(letterbox_face, 0)
            outputs = self.rknn_face.inference(inputs=[face_input])
            
            if len(outputs) > 0:
                embedding = outputs[0][0].astype(np.float32)
                
                # Save to database
                success = self.db.save_face_embedding(client_id, embedding, confidence=1.0)
                
                if success:
                    client_info = self.db.get_client_info(client_id)
                    name = f"{client_info['first_name']} {client_info['last_name']}" if client_info else "Unknown"
                    print(f"✅ Successfully enrolled: {name} (ID: {client_id})")
                else:
                    print(f"❌ Failed to save embedding for client {client_id}")
            else:
                print(f"❌ No face embedding extracted for client {client_id}")
                
        except Exception as e:
            print(f"❌ Error processing client {client_id}: {e}")
            import traceback
            traceback.print_exc()
    
    def stop(self):
        """Stop the processing thread"""
        self.is_running = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RetinaFace Python Demo with Auto-Enrollment', add_help=True)
    parser.add_argument('--model_path', type=str, required=True,
                        help='model path, could be .rknn file')
    parser.add_argument('--target', type=str,
                        default='rk3566', help='target RKNPU platform')
    parser.add_argument('--device_id', type=str,
                        default=None, help='device id')
    parser.add_argument('--db_host', type=str, required=True,
                        help='Database host IP')
    parser.add_argument('--db_name', type=str, default='gym_db',
                        help='Database name')
    parser.add_argument('--db_user', type=str, default='gym_user',
                        help='Database user')
    parser.add_argument('--db_password', type=str, required=True,
                        help='Database password')
    args = parser.parse_args()
    
    # Initialize database connection
    print("🔌 Connecting to database...")
    db = DatabaseHelper(
        host=args.db_host,
        database=args.db_name,
        user=args.db_user,
        password=args.db_password,
        port=5432
    )
    
    tm = cv2.TickMeter()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    esp32_ip = "192.168.1.110"
    esp32_port = 4210
    
    # Create RKNN objects
    print("🔧 Loading models...")
    rknn = RKNNLite()
    rknnFace = RKNNLite()

    # Load RKNN model
    ret = rknn.load_rknn(args.model_path)
    if ret != 0:
        print(f'❌ Load RKNN model "{args.model_path}" failed!')
        exit(ret)

    ret1 = rknnFace.load_rknn("../model/Arcfacefp.rknn")
    if ret1 != 0:
        print('❌ Load RKNN face model failed!')
        exit(ret1)

    # Init runtime environment
    print('🚀 Initializing runtime environment...')
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    if ret != 0:
        print('❌ Init runtime environment failed!')
        exit(ret)

    ret1 = rknnFace.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    if ret1 != 0:
        print('❌ Init runtime environment failed!')
        exit(ret1)
    
    print('✅ Models loaded successfully')
    
    # Initialize face enrollment processor
    enrollment_processor = FaceEnrollmentProcessor(db, rknnFace)
    
    # Define callback for database notifications
    def on_client_change(notification):
        """Handle database change notifications"""
        action = notification.get('action')
        client_id = notification.get('client_id')
        image_path = notification.get('image_path')
        
        print(f"📢 Client {action}: ID={client_id}")
        
        if image_path:
            # Check if embedding already exists
            if not db.check_embedding_exists(client_id):
                enrollment_processor.enqueue_client(client_id, image_path)
            else:
                print(f"ℹ️  Embedding already exists for client {client_id}")
    
    # Start listening for database changes
    db.start_listening(on_client_change)
    
    # Load all registered face embeddings from database
    embedding_shape = (512,)
    print("📥 Loading registered faces from database...")
    registered_faces = db.get_all_face_embeddings(embedding_shape)
    print(f"✅ Loaded {len(registered_faces)} registered faces")
    
    # Function to reload embeddings (call periodically or on notification)
    def reload_embeddings():
        global registered_faces
        registered_faces = db.get_all_face_embeddings(embedding_shape)
        print(f"🔄 Reloaded {len(registered_faces)} registered faces")
    
    # Set up video capture
    camrstp = "rtsp://10.183.120.18:8554/live"
    
    def open_stream():
        cap = cv2.VideoCapture(camrstp)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        return cap
    
    cap = open_stream()
    retry_count = 0

    img_width = 640
    img_height = 480
    model_height, model_width = (320, 320)
    Face_model_height, Face_model_width = (112, 112)
    
    # Reload embeddings every N seconds
    last_reload_time = time.time()
    reload_interval = 10  # seconds
    
    print("🎬 Starting main recognition loop...")
    print("=" * 60)
    
    while cv2.waitKey(1) < 0:
        tm.start()
        
        # Periodically reload embeddings
        if time.time() - last_reload_time > reload_interval:
            reload_embeddings()
            last_reload_time = time.time()
        
        ret, img = cap.read()
        if not ret or img is None:
            print("⚠️  Stream broken. Reconnecting...")
            cap.release()
            time.sleep(1)
            cap = open_stream()
            retry_count += 1
            if retry_count > 15:
                print("❌ Too many retries, resetting camera/system...")
            continue

        retry_count = 0
        
        # Process frame for face detection
        letterbox_img, aspect_ratio, offset_x, offset_y = letterbox_resize(
            img, (model_height, model_width), 114
        )
        infer_img = np.expand_dims(letterbox_img, 0)

        # Inference
        outputs = rknn.inference(inputs=[infer_img])
        loc, conf, landmarks = outputs
        priors = PriorBox(image_size=(model_height, model_width))
        boxes = box_decode(loc.squeeze(0), priors)
        scale = np.array([model_width, model_height, model_width, model_height])
        boxes = boxes * scale // 1
        boxes[..., 0::2] = np.clip((boxes[..., 0::2] - offset_x) / aspect_ratio, 0, img_width)
        boxes[..., 1::2] = np.clip((boxes[..., 1::2] - offset_y) / aspect_ratio, 0, img_height)
        scores = conf.squeeze(0)[:, 1]
        landmarks = decode_landm(landmarks.squeeze(0), priors)
        scale_landmarks = np.array([model_width, model_height, model_width, model_height,
                                    model_width, model_height, model_width, model_height,
                                    model_width, model_height])
        landmarks = landmarks * scale_landmarks // 1
        landmarks[..., 0::2] = np.clip((landmarks[..., 0::2] - offset_x) / aspect_ratio, 0, img_width)
        landmarks[..., 1::2] = np.clip((landmarks[..., 1::2] - offset_y) / aspect_ratio, 0, img_height)
        
        # Filter low scores
        inds = np.where(scores > 0.1)[0]
        boxes = boxes[inds]
        landmarks = landmarks[inds]
        scores = scores[inds]

        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landmarks = landmarks[order]
        scores = scores[order]

        # NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, 0.5)
        dets = dets[keep, :]
        landmarks = landmarks[keep]
        dets = np.concatenate((dets, landmarks), axis=1)

        for data in dets:
            if data[4] < 0.7:
                continue
                
            text = "{:.4f}".format(data[4])
            data = list(map(int, data))
            dx = data[7] - data[5]
            dy = data[8] - data[6]
            angle = math.atan2(dy, dx) * 180. / math.pi

            eye_center = ((data[5] + data[7]) // 2, (data[6] + data[8]) // 2)
            rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
            (h, w) = img.shape[:2]
            aligned_img = cv2.warpAffine(img, rotation_matrix, (w, h))

            face_img = aligned_img[data[1]:data[3], data[0]:data[2]]
            
            if face_img.size == 0:
                continue
                
            letterbox_face, Faspect_ratio, Foffset_x, Foffset_y = letterbox_resize(
                face_img, (Face_model_height, Face_model_width), 114
            )
            Finfer_img = np.expand_dims(letterbox_face, 0)
            outputs = rknnFace.inference(inputs=[Finfer_img])
            
            if len(outputs) > 0:
                current_embedding = outputs[0][0]
                
                # Compare with all registered faces
                best_match = None
                best_similarity = 0.0
                
                for registered_face in registered_faces:
                    sim = Similarity(registered_face['embedding'], current_embedding)
                    
                    if sim > best_similarity:
                        best_similarity = sim
                        best_match = registered_face
                
                # Access threshold
                if best_similarity > 0.6:
                    client_id = best_match['client_id']
                    client_name = best_match['name']
                    
                    print(f"✅ Access GRANTED: {client_name} (ID: {client_id}), Confidence: {best_similarity:.2f}")
                    
                    # Get locker number
                    client_info = db.get_client_info(client_id)
                    if client_info and client_info['locker'] is not None:
                        locker_number = client_info['locker']
                        sock.sendto(str(locker_number).encode("utf-8"), (esp32_ip, esp32_port))
                        print(f"🔓 Locker {locker_number} unlocked")
                    
                    # Log access
                    db.log_access(client_id, True, best_similarity)
                    
                    # Draw rectangle and name
                    cv2.rectangle(img, (data[0], data[1]), (data[2], data[3]), (0, 255, 0), 2)
                    cv2.putText(img, client_name, (data[0], data[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    print(f"❌ Access DENIED: Unknown person, Best match: {best_similarity:.2f}")
                    db.log_access(None, False, best_similarity)
                    
                    # Draw rectangle in red
                    cv2.rectangle(img, (data[0], data[1]), (data[2], data[3]), (0, 0, 255), 2)
        
        tm.stop()
        fps = tm.getFPS()
        cv2.putText(img, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.namedWindow("Face Recognition", cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Face Recognition", img)
        tm.reset()
    
    # Cleanup
    print("\n🛑 Shutting down...")
    enrollment_processor.stop()
    db.stop_listening()
    db.close_all_connections()
    rknn.release()
    rknnFace.release()
    cv2.destroyAllWindows()
    print("✅ Shutdown complete")



    # Replace this part in process_usb_camera_frame():
            # Recognition threshold for locker assignment
            if best_similarity > 0.65:  # Slightly higher threshold for assignment
                client_id = best_match['client_id']
                client_name = best_match['name']
                
                # Check current locker status
                client_info = db.get_client_info(client_id)
                if client_info:
                    if client_info['locker'] is None:
                        status = "ENTRY"
                        color = (0, 255, 0)  # Green for entry
                    else:
                        status = f"EXIT (Locker {client_info['locker']})"
                        color = (255, 165, 0)  # Orange for exit
                else:
                    status = "UNKNOWN"
                    color = (128, 128, 128)
                
                # Draw rectangle and name with status
                cv2.rectangle(frame, (data[0], data[1]), (data[2], data[3]), color, 2)
                cv2.putText(frame, f"{client_name}", 
                           (data[0], data[1] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"{status} ({best_similarity:.2f})", 
                           (data[0], data[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Assign or unassign locker
                assign_locker_to_recognized_face(client_id, client_name, best_similarity)