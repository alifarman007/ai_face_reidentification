import gradio as gr
import cv2
import os
import shutil
import numpy as np
import random
import time
import requests
import threading
import queue
from PIL import Image
from database import FaceDatabase
from models import SCRFD, ArcFace
from utils.logging import setup_logging
from utils.helpers import compute_similarity, draw_bbox_info, draw_bbox

# Setup logging
setup_logging(log_to_file=True)

# Global models and database
DETECTION_MODEL_PATH = "./weights/det_10g.onnx"
RECOGNITION_MODEL_PATH = "./weights/w600k_r50.onnx"
FACES_DIR = "./assets/faces"
DB_PATH = "./database/face_database"

# Load models and database
try:
    detector = SCRFD(DETECTION_MODEL_PATH, input_size=(640, 640), conf_thres=0.5)
    recognizer = ArcFace(RECOGNITION_MODEL_PATH)
    face_db = FaceDatabase(db_path=DB_PATH)
    face_db.load()
except Exception as e:
    detector = None
    recognizer = None
    face_db = None
    print(f"Error loading models: {e}")

# Settings
similarity_thresh = 0.4
confidence_thresh = 0.5

# FAISS API configuration
FAISS_API_URL = "http://localhost:8000"

def sync_with_faiss_api():
    """Trigger database reload in FAISS API to sync changes"""
    try:
        response = requests.post(f"{FAISS_API_URL}/database/reload", timeout=5)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print(f"FAISS API synced: {result.get('message', 'Database reloaded')}")
            else:
                print(f"FAISS API sync failed: {result.get('message', 'Unknown error')}")
        else:
            print(f"FAISS API sync failed: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Could not sync with FAISS API: {e}")
    except Exception as e:
        print(f"Error syncing with FAISS API: {e}")

# --- Person Management Functions ---
def add_person_gradio(name, image):
    if not name or image is None:
        return "Please provide a name and an image.", gr.update(), gr.update()
    os.makedirs(FACES_DIR, exist_ok=True)
    ext = ".jpg"
    dest_path = os.path.join(FACES_DIR, f"{name}{ext}")
    image.save(dest_path)
    return f"Person '{name}' added. Click 'Update Database' to process.", gr.update(), gr.update()

def update_database_gradio():
    global face_db
    if detector is None or recognizer is None:
        return "Models not loaded."
    face_db = FaceDatabase(db_path=DB_PATH)
    for filename in os.listdir(FACES_DIR):
        if not (filename.endswith('.jpg') or filename.endswith('.png')):
            continue
        name = filename.rsplit('.', 1)[0]
        image_path = os.path.join(FACES_DIR, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue
        bboxes, kpss = detector.detect(image, max_num=1)
        if len(kpss) == 0:
            continue
        embedding = recognizer.get_embedding(image, kpss[0])
        face_db.add_face(embedding, name)
    face_db.save()
    
    # Sync with FAISS API
    sync_with_faiss_api()
    
    return "Database updated and synced with FAISS API."

def list_persons_gradio():
    if not os.path.exists(FACES_DIR):
        return []
    return [f.rsplit('.', 1)[0] for f in os.listdir(FACES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]

def delete_person_gradio(name):
    if not name:
        return "Please select a person to delete."
    for filename in os.listdir(FACES_DIR):
        if filename.startswith(name + '.') and filename.endswith(('.jpg', '.png', '.jpeg')):
            os.remove(os.path.join(FACES_DIR, filename))
            break
    update_database_gradio()
    return f"Person '{name}' deleted and synced with FAISS API."

# --- Video Processing Functions ---
def process_video_simple(video_file, similarity, confidence, progress=gr.Progress()):
    """Simple video processing: upload -> process -> save -> display result"""
    if detector is None or recognizer is None or face_db is None:
        return None, "❌ Error: Models not loaded"
    
    if video_file is None:
        return None, "❌ Error: No video file provided"
    
    # Create output directory
    output_dir = "./processed_videos"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = int(time.time())
    output_filename = f"processed_video_{timestamp}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        # Open input video
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            return None, "❌ Error: Could not open video file"
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0:
            fps = 25.0
        
        # Create video writer with H.264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # If H264 fails, try with mp4v
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        # If still fails, try with XVID
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_path = output_path.replace('.mp4', '.avi')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Final check if video writer is working
        if not out.isOpened():
            cap.release()
            return None, "❌ Error: Could not initialize video writer with any codec"
        
        print(f"Video writer initialized: {output_path}")
        print(f"Codec: {fourcc}, FPS: {fps}, Size: {width}x{height}")
        
        # Initialize processing variables
        frame_count = 0
        colors = {}
        start_time = time.time()
        
        # Process video frame by frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect and recognize faces
            try:
                bboxes, kpss = detector.detect(frame, max_num=0)
                for bbox, kps in zip(bboxes, kpss):
                    *bbox, conf_score = bbox.astype(np.int32)
                    embedding = recognizer.get_embedding(frame, kps)
                    name, sim = face_db.search(embedding, similarity)
                    
                    if name != "Unknown":
                        if name not in colors:
                            colors[name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        draw_bbox_info(frame, bbox, similarity=sim, name=name, color=colors[name])
                    else:
                        draw_bbox(frame, bbox, (0, 0, 255))  # Red for unknown faces
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                # Continue with unprocessed frame
            
            # Write processed frame
            out.write(frame)
            frame_count += 1
            
            # Update progress
            progress_percent = frame_count / total_frames
            elapsed = time.time() - start_time
            fps_current = frame_count / elapsed if elapsed > 0 else 0
            
            progress(progress_percent, desc=f"Processing: {progress_percent*100:.1f}% ({frame_count}/{total_frames}) - {fps_current:.1f} FPS")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Small delay to ensure file is written to disk
        time.sleep(0.5)
        
        # Calculate final stats
        elapsed = time.time() - start_time
        
        # Verify the output file exists and has size > 0
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:  # At least 1KB
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            status_msg = f"✅ Processing Complete!\n📊 {frame_count} frames processed in {elapsed:.2f}s\n⚡ Average speed: {frame_count/elapsed:.1f} FPS\n💾 Saved: {os.path.basename(output_path)}\n📁 Size: {file_size_mb:.2f} MB\n🎬 Video ready to play!"
            print(f"Video processing completed successfully: {output_path}")
            return output_path, status_msg
        else:
            return None, f"❌ Error: Output video file was not created properly or is corrupted"
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# --- Webcam Functions ---
def webcam_recognition_gradio(similarity, confidence):
    if detector is None or recognizer is None or face_db is None:
        return None
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None
    bboxes, kpss = detector.detect(frame, max_num=0)
    colors = {}
    for bbox, kps in zip(bboxes, kpss):
        *bbox, conf_score = bbox.astype(np.int32)
        embedding = recognizer.get_embedding(frame, kps)
        name, sim = face_db.search(embedding, similarity)
        if name != "Unknown":
            if name not in colors:
                colors[name] = (0, 255, 0)
            draw_bbox_info(frame, bbox, similarity=sim, name=name, color=colors[name])
        else:
            draw_bbox(frame, bbox, (0, 0, 255))
    cap.release()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

def image_processing_gradio(image, similarity, confidence):
    if detector is None or recognizer is None or face_db is None or image is None:
        return None, ""
    if isinstance(image, Image.Image):
        image = np.array(image)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    bboxes, kpss = detector.detect(frame_bgr, max_num=0)
    colors = {}
    recognized_names = []
    for bbox, kps in zip(bboxes, kpss):
        try:
            *bbox, conf_score = bbox.astype(np.int32)
            embedding = recognizer.get_embedding(frame_bgr, kps)
            name, sim = face_db.search(embedding, similarity)
            if name != "Unknown":
                if name not in colors:
                    colors[name] = (0, 255, 0)
                draw_bbox_info(frame_bgr, bbox, similarity=sim, name=name, color=colors[name])
                recognized_names.append(name)
            else:
                draw_bbox(frame_bgr, bbox, (0, 0, 255))
        except Exception as e:
            print(f"Error in bbox/kps loop: {e}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    recognized_names = sorted(set(recognized_names))
    names_text = "\n".join(recognized_names)
    return Image.fromarray(frame_rgb), names_text

# --- IP Camera Section ---
import threading
ipcam_thread = None
ipcam_stop_flag = False
ipcam_frame = None
ipcam_names = []

# --- IP Camera 2 Section ---
ipcam2_thread = None
ipcam2_stop_flag = False
ipcam2_frame = None
ipcam2_names = []

def ipcam_streamer(url, similarity, confidence):
    global ipcam_thread, ipcam_stop_flag, ipcam_frame, ipcam_names
    ipcam_stop_flag = False
    def grab_frames():
        global ipcam_frame, ipcam_names
        cap = cv2.VideoCapture(url)
        while not ipcam_stop_flag and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            bboxes, kpss = detector.detect(frame, max_num=0)
            colors = {}
            recognized_names = []
            for bbox, kps in zip(bboxes, kpss):
                try:
                    *bbox, conf_score = bbox.astype(np.int32)
                    embedding = recognizer.get_embedding(frame, kps)
                    name, sim = face_db.search(embedding, similarity)
                    if name != "Unknown":
                        if name not in colors:
                            colors[name] = (0, 255, 0)
                        draw_bbox_info(frame, bbox, similarity=sim, name=name, color=colors[name])
                        recognized_names.append(name)
                    else:
                        draw_bbox(frame, bbox, (0, 0, 255))
                except Exception as e:
                    print(f"Error in bbox/kps loop: {e}")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ipcam_frame = Image.fromarray(frame_rgb)
            ipcam_names = sorted(set(recognized_names))
            time.sleep(0.03)
        cap.release()
    if ipcam_thread is None or not ipcam_thread.is_alive():
        ipcam_thread = threading.Thread(target=grab_frames, daemon=True)
        ipcam_thread.start()
    return None, ""

def ipcam_get_frame():
    global ipcam_frame, ipcam_names
    if ipcam_frame is None:
        return None, ""
    return ipcam_frame, "\n".join(ipcam_names)

def ipcam_stop():
    global ipcam_stop_flag
    ipcam_stop_flag = True
    return None, ""

# --- IP Camera 2 Functions ---
def ipcam2_streamer(url, similarity, confidence):
    global ipcam2_thread, ipcam2_stop_flag, ipcam2_frame, ipcam2_names
    ipcam2_stop_flag = False
    def grab_frames():
        global ipcam2_frame, ipcam2_names
        cap = cv2.VideoCapture(url)
        while not ipcam2_stop_flag and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            bboxes, kpss = detector.detect(frame, max_num=0)
            colors = {}
            recognized_names = []
            for bbox, kps in zip(bboxes, kpss):
                try:
                    *bbox, conf_score = bbox.astype(np.int32)
                    embedding = recognizer.get_embedding(frame, kps)
                    name, sim = face_db.search(embedding, similarity)
                    if name != "Unknown":
                        if name not in colors:
                            colors[name] = (0, 255, 0)
                        draw_bbox_info(frame, bbox, similarity=sim, name=name, color=colors[name])
                        recognized_names.append(name)
                    else:
                        draw_bbox(frame, bbox, (0, 0, 255))
                except Exception as e:
                    print(f"Error in bbox/kps loop: {e}")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ipcam2_frame = Image.fromarray(frame_rgb)
            ipcam2_names = sorted(set(recognized_names))
            time.sleep(0.03)
        cap.release()
    if ipcam2_thread is None or not ipcam2_thread.is_alive():
        ipcam2_thread = threading.Thread(target=grab_frames, daemon=True)
        ipcam2_thread.start()
    return None, ""

def ipcam2_get_frame():
    global ipcam2_frame, ipcam2_names
    if ipcam2_frame is None:
        return None, ""
    return ipcam2_frame, "\n".join(ipcam2_names)

def ipcam2_stop():
    global ipcam2_stop_flag
    ipcam2_stop_flag = True
    return None, ""

# --- RTSP URL Generation Function ---
def generate_rtsp_url(camera_ip, camera_port, username, password, rtsp_path="/stream"):
    """Generate RTSP URL from camera configuration"""
    if not camera_ip:
        return ""
    
    # Default port if not specified
    if not camera_port:
        camera_port = "554"
    
    # Build RTSP URL
    if username and password:
        rtsp_url = f"rtsp://{username}:{password}@{camera_ip}:{camera_port}{rtsp_path}"
    else:
        rtsp_url = f"rtsp://{camera_ip}:{camera_port}{rtsp_path}"
    
    return rtsp_url

# --- Gradio Interface Layout ---
with gr.Blocks(title="AI Face Recognition") as demo:
    gr.Markdown("""
    <div style='display: flex; align-items: center; justify-content: center; margin-bottom: 20px;'>
        <h1>AI Face Reidentification</h1>
    </div>
    """)
    with gr.Tab("Person Management"):
        with gr.Row():
            name_input = gr.Text(label="Person Name")
            image_input = gr.Image(type="pil", label="Select Image")
        add_btn = gr.Button("Add Person")
        update_db_btn = gr.Button("Update Database")
        person_list = gr.Dropdown(choices=list_persons_gradio(), label="Registered Persons", interactive=True)
        delete_btn = gr.Button("Delete Selected")
        person_status = gr.Textbox(label="Status", interactive=False)
        add_btn.click(add_person_gradio, inputs=[name_input, image_input], outputs=[person_status, name_input, image_input])
        update_db_btn.click(update_database_gradio, outputs=person_status)
        delete_btn.click(delete_person_gradio, inputs=person_list, outputs=person_status)
        update_db_btn.click(lambda: gr.update(choices=list_persons_gradio()), None, person_list)
        delete_btn.click(lambda: gr.update(choices=list_persons_gradio()), None, person_list)
    with gr.Tab("Video Processing"):
        gr.Markdown("### Upload and Process Video")
        gr.Markdown("Upload a video file, adjust settings, and click process to analyze faces. The processed video will be saved and displayed below.")
        
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="📁 Upload Video File")
                similarity_slider = gr.Slider(0.1, 1.0, value=similarity_thresh, label="🎯 Similarity Threshold")
                confidence_slider = gr.Slider(0.1, 1.0, value=confidence_thresh, label="🔍 Confidence Threshold")
                process_btn = gr.Button("🚀 Process Video", variant="primary", size="lg")
            
            with gr.Column():
                status_display = gr.Textbox(
                    label="📊 Processing Status", 
                    interactive=False, 
                    lines=5,
                    value="Ready to process video...\n\n1. Upload a video file\n2. Adjust thresholds if needed\n3. Click 'Process Video'"
                )
        
        gr.Markdown("---")
        
        with gr.Row():
            video_output = gr.Video(label="🎬 Processed Video (Click to play)", height=400)
        
        # Process video event handler
        process_btn.click(
            process_video_simple, 
            inputs=[video_input, similarity_slider, confidence_slider], 
            outputs=[video_output, status_display]
        )
    with gr.Tab("Image Processing"):
        image_input = gr.Image(type="pil", label="Upload Image")
        image_similarity = gr.Slider(0.1, 1.0, value=similarity_thresh, label="Similarity Threshold")
        image_confidence = gr.Slider(0.1, 1.0, value=confidence_thresh, label="Confidence Threshold")
        process_image_btn = gr.Button("Process Image")
        image_output = gr.Image(label="Processed Image")
        image_names = gr.Textbox(label="Recognized Persons (Live)", interactive=False, lines=5)
        process_image_btn.click(image_processing_gradio, inputs=[image_input, image_similarity, image_confidence], outputs=[image_output, image_names])
    with gr.Tab("Webcam"):
        webcam_similarity = gr.Slider(0.1, 1.0, value=similarity_thresh, label="Similarity Threshold")
        webcam_confidence = gr.Slider(0.1, 1.0, value=confidence_thresh, label="Confidence Threshold")
        webcam_live = gr.Image(sources=["webcam"], streaming=True, label="Webcam Live Feed")
        webcam_names = gr.Textbox(label="Recognized Persons (Live)", interactive=False, lines=5)

        def webcam_realtime_gradio(frame, similarity, confidence):
            print("Processing frame for webcam detection...")
            if detector is None or recognizer is None or face_db is None or frame is None:
                print("Model or frame not available.")
                return None, ""
            print(f"Frame type: {type(frame)}")
            if isinstance(frame, Image.Image):
                frame = np.array(frame)
            print(f"Frame shape after np.array: {frame.shape}")
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]  # Remove alpha if present
                print("Alpha channel removed.")
            try:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"cv2.cvtColor error: {e}")
                return Image.fromarray(frame), ""
            try:
                bboxes, kpss = detector.detect(frame_bgr, max_num=0)
                print(f"Detected {len(bboxes)} faces.")
            except Exception as e:
                print(f"Detector error: {e}")
                return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)), ""
            colors = {}
            recognized_names = []
            for bbox, kps in zip(bboxes, kpss):
                try:
                    *bbox, conf_score = bbox.astype(np.int32)
                    embedding = recognizer.get_embedding(frame_bgr, kps)
                    name, sim = face_db.search(embedding, similarity)
                    if name != "Unknown":
                        if name not in colors:
                            colors[name] = (0, 255, 0)
                        draw_bbox_info(frame_bgr, bbox, similarity=sim, name=name, color=colors[name])
                        recognized_names.append(name)
                    else:
                        draw_bbox(frame_bgr, bbox, (0, 0, 255))
                except Exception as e:
                    print(f"Error in bbox/kps loop: {e}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            recognized_names = sorted(set(recognized_names))
            names_text = "\n".join(recognized_names)
            return Image.fromarray(frame_rgb), names_text

        webcam_live.stream(
            fn=webcam_realtime_gradio,
            inputs=[webcam_live, webcam_similarity, webcam_confidence],
            outputs=[webcam_live, webcam_names]
        )
    with gr.Tab("IP Camera"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Camera Configuration")
                ipcam_name = gr.Textbox(label="Camera Name", value="Camera 1")
                ipcam_ip = gr.Textbox(label="Camera IP Address", placeholder="192.168.1.100")
                ipcam_port = gr.Textbox(label="Camera Port", value="554", placeholder="554")
                ipcam_username = gr.Textbox(label="Username", placeholder="admin")
                ipcam_password = gr.Textbox(label="Password", type="password", placeholder="password")
                ipcam_rtsp_path = gr.Textbox(label="RTSP Path", value="/stream", placeholder="/stream")
                generate_url_btn = gr.Button("Generate RTSP URL")
                ipcam_url = gr.Textbox(label="Generated RTSP URL", interactive=False)
                
                def update_rtsp_url(ip, port, username, password, rtsp_path):
                    return generate_rtsp_url(ip, port, username, password, rtsp_path)
                
                generate_url_btn.click(
                    update_rtsp_url, 
                    inputs=[ipcam_ip, ipcam_port, ipcam_username, ipcam_password, ipcam_rtsp_path], 
                    outputs=ipcam_url
                )
                
            with gr.Column():
                gr.Markdown("### Manual URL Input (Optional)")
                ipcam_manual_url = gr.Textbox(label="Manual RTSP/HTTP URL", placeholder="rtsp://192.168.1.100:554/stream")
                gr.Markdown("*Use either generated URL or manual URL*")
        
        ipcam_similarity = gr.Slider(0.1, 1.0, value=similarity_thresh, label="Similarity Threshold")
        ipcam_confidence = gr.Slider(0.1, 1.0, value=confidence_thresh, label="Confidence Threshold")
        
        with gr.Row():
            start_ipcam_btn = gr.Button("Start IP Camera")
            stop_ipcam_btn = gr.Button("Stop IP Camera")
        
        ipcam_image = gr.Image(label="IP Camera Live Feed")
        ipcam_names_box = gr.Textbox(label="Recognized Persons (Live)", interactive=False, lines=5)
        
        def start_camera_with_url_selection(generated_url, manual_url, similarity, confidence):
            # Use manual URL if provided, otherwise use generated URL
            final_url = manual_url if manual_url.strip() else generated_url
            if not final_url:
                return None, "Please provide either generated or manual URL"
            return ipcam_streamer(final_url, similarity, confidence)
        
        start_ipcam_btn.click(
            start_camera_with_url_selection, 
            inputs=[ipcam_url, ipcam_manual_url, ipcam_similarity, ipcam_confidence], 
            outputs=[ipcam_image, ipcam_names_box]
        )
        stop_ipcam_btn.click(ipcam_stop, outputs=[ipcam_image, ipcam_names_box])
        # Poll for new frames every 100ms
        ipcam_image.stream(ipcam_get_frame, outputs=[ipcam_image, ipcam_names_box])
    with gr.Tab("IP Camera 2"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Camera 2 Configuration")
                ipcam2_name = gr.Textbox(label="Camera Name", value="Camera 2")
                ipcam2_ip = gr.Textbox(label="Camera IP Address", placeholder="192.168.1.101")
                ipcam2_port = gr.Textbox(label="Camera Port", value="554", placeholder="554")
                ipcam2_username = gr.Textbox(label="Username", placeholder="admin")
                ipcam2_password = gr.Textbox(label="Password", type="password", placeholder="password")
                ipcam2_rtsp_path = gr.Textbox(label="RTSP Path", value="/stream", placeholder="/stream")
                generate_url2_btn = gr.Button("Generate RTSP URL")
                ipcam2_url = gr.Textbox(label="Generated RTSP URL", interactive=False)
                
                def update_rtsp_url2(ip, port, username, password, rtsp_path):
                    return generate_rtsp_url(ip, port, username, password, rtsp_path)
                
                generate_url2_btn.click(
                    update_rtsp_url2, 
                    inputs=[ipcam2_ip, ipcam2_port, ipcam2_username, ipcam2_password, ipcam2_rtsp_path], 
                    outputs=ipcam2_url
                )
                
            with gr.Column():
                gr.Markdown("### Manual URL Input (Optional)")
                ipcam2_manual_url = gr.Textbox(label="Manual RTSP/HTTP URL", placeholder="rtsp://192.168.1.101:554/stream")
                gr.Markdown("*Use either generated URL or manual URL*")
        
        ipcam2_similarity = gr.Slider(0.1, 1.0, value=similarity_thresh, label="Similarity Threshold")
        ipcam2_confidence = gr.Slider(0.1, 1.0, value=confidence_thresh, label="Confidence Threshold")
        
        with gr.Row():
            start_ipcam2_btn = gr.Button("Start IP Camera 2")
            stop_ipcam2_btn = gr.Button("Stop IP Camera 2")
        
        ipcam2_image = gr.Image(label="IP Camera 2 Live Feed")
        ipcam2_names_box = gr.Textbox(label="Recognized Persons (Live)", interactive=False, lines=5)
        
        def start_camera2_with_url_selection(generated_url, manual_url, similarity, confidence):
            # Use manual URL if provided, otherwise use generated URL
            final_url = manual_url if manual_url.strip() else generated_url
            if not final_url:
                return None, "Please provide either generated or manual URL"
            return ipcam2_streamer(final_url, similarity, confidence)
        
        start_ipcam2_btn.click(
            start_camera2_with_url_selection, 
            inputs=[ipcam2_url, ipcam2_manual_url, ipcam2_similarity, ipcam2_confidence], 
            outputs=[ipcam2_image, ipcam2_names_box]
        )
        stop_ipcam2_btn.click(ipcam2_stop, outputs=[ipcam2_image, ipcam2_names_box])
        # Poll for new frames every 100ms
        ipcam2_image.stream(ipcam2_get_frame, outputs=[ipcam2_image, ipcam2_names_box])
    with gr.Tab("Settings"):
        gr.Markdown("**Detection Model:** ./weights/det_10g.onnx")
        gr.Markdown("**Recognition Model:** ./weights/w600k_r50.onnx")
        gr.Markdown("Adjust thresholds in the Video, Image, Webcam, and IP Camera tabs.")

demo.launch(server_name="0.0.0.0", server_port=7861)  # Use a different port than the FastAPI server