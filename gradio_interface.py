import gradio as gr
import cv2
import os
import shutil
import numpy as np
import random
import time
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
    return "Database updated."

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
    return f"Person '{name}' deleted."

# --- Video Processing Functions ---
def process_video_gradio(video, similarity, confidence):
    if detector is None or recognizer is None or face_db is None:
        return None, "Models not loaded."
    temp_input = "./assets/temp_input.mp4"
    temp_output = "./assets/temp_output.mp4"
    video.save(temp_input)
    cap = cv2.VideoCapture(temp_input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    out = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    colors = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
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
                draw_bbox(frame, bbox, (255, 0, 0))
        out.write(frame)
    cap.release()
    out.release()
    return temp_output, "Video processed."

def save_frame_gradio(video, frame_number):
    temp_input = "./assets/temp_input.mp4"
    video.save(temp_input)
    cap = cv2.VideoCapture(temp_input)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        return None, "Frame not found."
    temp_frame = "./assets/temp_frame.jpg"
    cv2.imwrite(temp_frame, frame)
    cap.release()
    return temp_frame, "Frame saved."

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
        video_input = gr.Video(label="Upload Video")
        similarity_slider = gr.Slider(0.1, 1.0, value=similarity_thresh, label="Similarity Threshold")
        confidence_slider = gr.Slider(0.1, 1.0, value=confidence_thresh, label="Confidence Threshold")
        process_btn = gr.Button("Process Video")
        video_output = gr.Video(label="Processed Video")
        video_status = gr.Textbox(label="Status", interactive=False)
        process_btn.click(process_video_gradio, inputs=[video_input, similarity_slider, confidence_slider], outputs=[video_output, video_status])
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
        ipcam_url = gr.Textbox(label="IP Camera URL (RTSP/HTTP)")
        ipcam_similarity = gr.Slider(0.1, 1.0, value=similarity_thresh, label="Similarity Threshold")
        ipcam_confidence = gr.Slider(0.1, 1.0, value=confidence_thresh, label="Confidence Threshold")
        start_ipcam_btn = gr.Button("Start IP Camera")
        stop_ipcam_btn = gr.Button("Stop IP Camera")
        ipcam_image = gr.Image(label="IP Camera Live Feed")
        ipcam_names_box = gr.Textbox(label="Recognized Persons (Live)", interactive=False, lines=5)
        start_ipcam_btn.click(ipcam_streamer, inputs=[ipcam_url, ipcam_similarity, ipcam_confidence], outputs=[ipcam_image, ipcam_names_box])
        stop_ipcam_btn.click(ipcam_stop, outputs=[ipcam_image, ipcam_names_box])
        # Poll for new frames every 100ms
        ipcam_image.stream(ipcam_get_frame, outputs=[ipcam_image, ipcam_names_box])
    with gr.Tab("Settings"):
        gr.Markdown("**Detection Model:** ./weights/det_10g.onnx")
        gr.Markdown("**Recognition Model:** ./weights/w600k_r50.onnx")
        gr.Markdown("Adjust thresholds in the Video, Image, Webcam, and IP Camera tabs.")

demo.launch(server_name="0.0.0.0", server_port=7861)  # Use a different port than the FastAPI server