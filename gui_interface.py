import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
import shutil
import threading
import subprocess
import sys
import random
import time
from PIL import Image, ImageTk
import numpy as np
from database import FaceDatabase
from models import SCRFD, ArcFace
from utils.logging import setup_logging
from utils.helpers import compute_similarity, draw_bbox_info, draw_bbox
import argparse


class FaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1200x800")
        
        # Initialize models
        self.detector = None
        self.recognizer = None
        self.face_db = None
        self.webcam_active = False
        self.cap = None
        self.video_thread = None
        
        # Default parameters
        self.similarity_thresh = tk.DoubleVar(value=0.4)
        self.confidence_thresh = tk.DoubleVar(value=0.5)
        
        setup_logging(log_to_file=True)
        self.setup_ui()
        self.load_models()
        
    def setup_ui(self):
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Person Management Tab
        self.person_frame = ttk.Frame(notebook)
        notebook.add(self.person_frame, text="Person Management")
        self.setup_person_management()
        
        # Video Processing Tab
        self.video_frame = ttk.Frame(notebook)
        notebook.add(self.video_frame, text="Video Processing")
        self.setup_video_processing()
        
        # Webcam Tab
        self.webcam_frame = ttk.Frame(notebook)
        notebook.add(self.webcam_frame, text="Live Webcam")
        self.setup_webcam()
        
        # Settings Tab
        self.settings_frame = ttk.Frame(notebook)
        notebook.add(self.settings_frame, text="Settings")
        self.setup_settings()
        
    def setup_person_management(self):
        # Person upload section
        upload_frame = ttk.LabelFrame(self.person_frame, text="Add New Person", padding="10")
        upload_frame.pack(fill='x', padx=10, pady=5)
        
        # Name entry
        ttk.Label(upload_frame, text="Person Name:").grid(row=0, column=0, sticky='w', padx=5)
        self.person_name_var = tk.StringVar()
        name_entry = ttk.Entry(upload_frame, textvariable=self.person_name_var, width=30)
        name_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Image selection
        ttk.Label(upload_frame, text="Select Image:").grid(row=1, column=0, sticky='w', padx=5)
        self.selected_image_var = tk.StringVar()
        image_label = ttk.Label(upload_frame, textvariable=self.selected_image_var, width=40, relief='sunken')
        image_label.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Button(upload_frame, text="Browse Image", command=self.browse_image).grid(row=1, column=2, padx=5)
        
        # Image preview
        self.image_preview_label = ttk.Label(upload_frame, text="No image selected")
        self.image_preview_label.grid(row=2, column=0, columnspan=3, pady=10)
        
        # Upload and update buttons
        button_frame = ttk.Frame(upload_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Add Person", command=self.add_person).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Update Database", command=self.update_database).pack(side='left', padx=5)
        
        # Database info section
        db_frame = ttk.LabelFrame(self.person_frame, text="Database Information", padding="10")
        db_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Person list
        ttk.Label(db_frame, text="Registered Persons:").pack(anchor='w')
        
        list_frame = ttk.Frame(db_frame)
        list_frame.pack(fill='both', expand=True, pady=5)
        
        self.person_listbox = tk.Listbox(list_frame, height=15)
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.person_listbox.yview)
        self.person_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.person_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Control buttons
        button_control_frame = ttk.Frame(db_frame)
        button_control_frame.pack(pady=5)
        
        ttk.Button(button_control_frame, text="Refresh List", command=self.refresh_person_list).pack(side='left', padx=5)
        ttk.Button(button_control_frame, text="Delete Selected", command=self.delete_person).pack(side='left', padx=5)
        
        self.refresh_person_list()
        
    def setup_video_processing(self):
        # Video upload section
        upload_frame = ttk.LabelFrame(self.video_frame, text="Video Selection", padding="10")
        upload_frame.pack(fill='x', padx=10, pady=5)
        
        # Video selection
        ttk.Label(upload_frame, text="Select Video:").grid(row=0, column=0, sticky='w', padx=5)
        self.selected_video_var = tk.StringVar()
        video_label = ttk.Label(upload_frame, textvariable=self.selected_video_var, width=50, relief='sunken')
        video_label.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(upload_frame, text="Browse Video", command=self.browse_video).grid(row=0, column=2, padx=5)
        
        # Control buttons
        control_frame = ttk.Frame(upload_frame)
        control_frame.grid(row=1, column=0, columnspan=3, pady=10)
        
        ttk.Button(control_frame, text="Play Video", command=self.play_video).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Pause", command=self.pause_video).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Stop", command=self.stop_video).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Save Current Frame", command=self.save_output_video).pack(side='left', padx=10)
        
        # Status label
        self.video_status_var = tk.StringVar(value="Select a video file to start")
        ttk.Label(upload_frame, textvariable=self.video_status_var).grid(row=2, column=0, columnspan=3, pady=5)
        
        # Video display section
        display_frame = ttk.LabelFrame(self.video_frame, text="Video Player with Face Recognition", padding="10")
        display_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.video_display_label = ttk.Label(display_frame, text="No video loaded")
        self.video_display_label.pack(expand=True)
        
        # Video processing variables
        self.video_cap = None
        self.video_playing = False
        self.video_paused = False
        
    def setup_webcam(self):
        # Webcam controls
        control_frame = ttk.LabelFrame(self.webcam_frame, text="Webcam Controls", padding="10")
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(control_frame, text="Start Webcam", command=self.start_webcam).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Stop Webcam", command=self.stop_webcam).pack(side='left', padx=5)
        
        # Webcam display
        display_frame = ttk.LabelFrame(self.webcam_frame, text="Live Feed", padding="10")
        display_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.webcam_label = ttk.Label(display_frame, text="Webcam not active")
        self.webcam_label.pack(expand=True)
        
    def setup_settings(self):
        # Threshold settings
        thresh_frame = ttk.LabelFrame(self.settings_frame, text="Detection Thresholds", padding="10")
        thresh_frame.pack(fill='x', padx=10, pady=5)
        
        # Similarity threshold
        ttk.Label(thresh_frame, text="Similarity Threshold:").grid(row=0, column=0, sticky='w', padx=5)
        similarity_scale = ttk.Scale(thresh_frame, from_=0.1, to=1.0, variable=self.similarity_thresh, 
                                   orient='horizontal', length=300)
        similarity_scale.grid(row=0, column=1, padx=5, pady=5)
        similarity_value_label = ttk.Label(thresh_frame, textvariable=self.similarity_thresh)
        similarity_value_label.grid(row=0, column=2, padx=5)
        
        # Confidence threshold
        ttk.Label(thresh_frame, text="Confidence Threshold:").grid(row=1, column=0, sticky='w', padx=5)
        confidence_scale = ttk.Scale(thresh_frame, from_=0.1, to=1.0, variable=self.confidence_thresh,
                                   orient='horizontal', length=300)
        confidence_scale.grid(row=1, column=1, padx=5, pady=5)
        confidence_value_label = ttk.Label(thresh_frame, textvariable=self.confidence_thresh)
        confidence_value_label.grid(row=1, column=2, padx=5)
        
        # Model paths
        model_frame = ttk.LabelFrame(self.settings_frame, text="Model Paths", padding="10")
        model_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(model_frame, text="Detection Model:").grid(row=0, column=0, sticky='w', padx=5)
        ttk.Label(model_frame, text="./weights/det_10g.onnx").grid(row=0, column=1, sticky='w', padx=5)
        
        ttk.Label(model_frame, text="Recognition Model:").grid(row=1, column=0, sticky='w', padx=5)
        ttk.Label(model_frame, text="./weights/w600k_r50.onnx").grid(row=1, column=1, sticky='w', padx=5)
        
    def load_models(self):
        try:
            self.detector = SCRFD("./weights/det_10g.onnx", input_size=(640, 640), 
                                conf_thres=self.confidence_thresh.get())
            self.recognizer = ArcFace("./weights/w600k_r50.onnx")
            self.face_db = FaceDatabase(db_path="./database/face_database")
            self.face_db.load()
            messagebox.showinfo("Success", "Models loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
            
    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.selected_image_var.set(file_path)
            self.show_image_preview(file_path)
            
    def show_image_preview(self, image_path):
        try:
            # Load and resize image for preview
            pil_image = Image.open(image_path)
            pil_image.thumbnail((200, 200), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_image)
            
            self.image_preview_label.configure(image=photo, text="")
            self.image_preview_label.image = photo  # Keep a reference
        except Exception as e:
            self.image_preview_label.configure(text=f"Error loading image: {str(e)}", image="")
            
    def add_person(self):
        name = self.person_name_var.get().strip()
        image_path = self.selected_image_var.get()
        
        if not name:
            messagebox.showerror("Error", "Please enter a person name")
            return
            
        if not image_path or not os.path.exists(image_path):
            messagebox.showerror("Error", "Please select a valid image file")
            return
            
        try:
            # Create faces directory if it doesn't exist
            faces_dir = "./assets/faces"
            os.makedirs(faces_dir, exist_ok=True)
            
            # Get file extension
            _, ext = os.path.splitext(image_path)
            
            # Copy image to faces directory
            dest_path = os.path.join(faces_dir, f"{name}{ext}")
            shutil.copy2(image_path, dest_path)
            
            messagebox.showinfo("Success", f"Person '{name}' added successfully! Click 'Update Database' to process.")
            
            # Clear form
            self.person_name_var.set("")
            self.selected_image_var.set("")
            self.image_preview_label.configure(text="No image selected", image="")
            
            self.refresh_person_list()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add person: {str(e)}")
            
    def update_database(self):
        if not self.detector or not self.recognizer:
            messagebox.showerror("Error", "Models not loaded")
            return
            
        try:
            # Create a simple namespace for parameters
            class Params:
                def __init__(self):
                    self.faces_dir = "./assets/faces"
                    self.db_path = "./database/face_database"
                    
            params = Params()
            
            # Rebuild database
            self.face_db = FaceDatabase(db_path=params.db_path)
            
            faces_dir = params.faces_dir
            if not os.path.exists(faces_dir):
                messagebox.showwarning("Warning", "No faces directory found")
                return
                
            for filename in os.listdir(faces_dir):
                if not (filename.endswith('.jpg') or filename.endswith('.png')):
                    continue
                    
                name = filename.rsplit('.', 1)[0]
                image_path = os.path.join(faces_dir, filename)
                image = cv2.imread(image_path)
                
                if image is None:
                    continue
                    
                bboxes, kpss = self.detector.detect(image, max_num=1)
                
                if len(kpss) == 0:
                    continue
                    
                embedding = self.recognizer.get_embedding(image, kpss[0])
                self.face_db.add_face(embedding, name)
                
            self.face_db.save()
            self.refresh_person_list()
            messagebox.showinfo("Success", "Database updated successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update database: {str(e)}")
            
    def refresh_person_list(self):
        self.person_listbox.delete(0, tk.END)
        
        faces_dir = "./assets/faces"
        if os.path.exists(faces_dir):
            for filename in os.listdir(faces_dir):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    name = filename.rsplit('.', 1)[0]
                    self.person_listbox.insert(tk.END, name)
                    
    def delete_person(self):
        selection = self.person_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a person to delete")
            return
            
        person_name = self.person_listbox.get(selection[0])
        
        # Confirm deletion
        confirm = messagebox.askyesno("Confirm Delete", 
                                    f"Are you sure you want to delete '{person_name}'?\n"
                                    "This will remove their image and database entry.")
        if not confirm:
            return
            
        try:
            faces_dir = "./assets/faces"
            # Find and delete the image file
            for filename in os.listdir(faces_dir):
                if filename.startswith(person_name + '.') and filename.endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(faces_dir, filename)
                    os.remove(image_path)
                    break
                    
            # Update database after deletion
            self.update_database()
            self.refresh_person_list()
            
            messagebox.showinfo("Success", f"Person '{person_name}' deleted successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete person: {str(e)}")
                    
    def browse_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if file_path:
            self.selected_video_var.set(file_path)
            
    def play_video(self):
        video_path = self.selected_video_var.get()
        
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", "Please select a valid video file")
            return
            
        if not self.detector or not self.recognizer or not self.face_db:
            messagebox.showerror("Error", "Models not loaded")
            return
        
        # Stop any currently playing video
        self.stop_video()
        
        # Copy video to assets directory
        try:
            assets_dir = "./assets"
            os.makedirs(assets_dir, exist_ok=True)
            assets_video_path = os.path.join(assets_dir, os.path.basename(video_path))
            if not os.path.exists(assets_video_path) or os.path.getmtime(video_path) > os.path.getmtime(assets_video_path):
                shutil.copy2(video_path, assets_video_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy video: {str(e)}")
            return
            
        # Open video capture
        self.video_cap = cv2.VideoCapture(video_path)
        if not self.video_cap.isOpened():
            messagebox.showerror("Error", f"Could not open video: {video_path}")
            return
            
        self.video_playing = True
        self.video_paused = False
        self.video_status_var.set("Playing video with face recognition...")
        
        # Start video playback
        self.update_video_frame()
        
    def pause_video(self):
        if self.video_playing:
            self.video_paused = not self.video_paused
            status = "Paused" if self.video_paused else "Playing video with face recognition..."
            self.video_status_var.set(status)
            
    def stop_video(self):
        self.video_playing = False
        self.video_paused = False
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        self.video_display_label.configure(image="", text="Video stopped")
        self.video_status_var.set("Video stopped")
        
    def save_output_video(self):
        if not hasattr(self, 'current_processed_frame') or self.current_processed_frame is None:
            messagebox.showwarning("Warning", "No frame to save. Please play a video first.")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.current_processed_frame)
                messagebox.showinfo("Success", f"Frame saved as {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save frame: {str(e)}")
                
    def update_video_frame(self):
        if not self.video_playing or not self.video_cap:
            return
            
        if self.video_paused:
            # Still schedule next update even when paused
            self.root.after(30, self.update_video_frame)
            return
            
        ret, frame = self.video_cap.read()
        if not ret:
            # End of video
            self.video_status_var.set("Video finished")
            self.stop_video()
            return
            
        # Process frame with face recognition
        try:
            colors = {}
            bboxes, kpss = self.detector.detect(frame, max_num=0)
            
            for bbox, kps in zip(bboxes, kpss):
                *bbox, conf_score = bbox.astype(np.int32)
                embedding = self.recognizer.get_embedding(frame, kps)
                
                name, similarity = self.face_db.search(embedding, self.similarity_thresh.get())
                
                if name != "Unknown":
                    if name not in colors:
                        colors[name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    draw_bbox_info(frame, bbox, similarity=similarity, name=name, color=colors[name])
                else:
                    draw_bbox(frame, bbox, (255, 0, 0))  # Red for unknown faces
                    
            # Store current processed frame for saving
            self.current_processed_frame = frame.copy()
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Continue with unprocessed frame
            
        # Convert to PIL and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Resize to fit display while maintaining aspect ratio
        display_width, display_height = 800, 600
        pil_image.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(pil_image)
        self.video_display_label.configure(image=photo, text="")
        self.video_display_label.image = photo  # Keep a reference
        
        # Schedule next frame (approximately 30 FPS)
        self.root.after(33, self.update_video_frame)
        
    def process_video_direct(self, video_path, output_path):
        """Process video directly using the same logic as main.py"""
        print(f"Processing video: {video_path}")
        print(f"Output path: {output_path}")
        
        if not self.detector or not self.recognizer or not self.face_db:
            raise Exception("Models not loaded")
        
        if not os.path.exists(video_path):
            raise Exception(f"Video file does not exist: {video_path}")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        colors = {}
        
        # Try opening video with retry mechanism
        cap = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    break
                else:
                    cap.release()
                    cap = None
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
            except Exception as e:
                if cap:
                    cap.release()
                    cap = None
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    raise Exception(f"Could not open video source after {max_retries} attempts: {video_path}")
        
        if not cap or not cap.isOpened():
            raise Exception(f"Could not open video source: {video_path}")
            
        cap_out = None
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
            
            if fps <= 0:
                fps = 25.0  # Default fps if unable to detect
            
            # Use the same codec as main.py
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            cap_out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not cap_out.isOpened():
                raise Exception(f"Could not create output video: {output_path}")
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"End of video reached at frame {frame_count}")
                    break
                    
                # Process frame using the same logic as main.py
                try:
                    bboxes, kpss = self.detector.detect(frame, max_num=0)
                    
                    for bbox, kps in zip(bboxes, kpss):
                        *bbox, conf_score = bbox.astype(np.int32)
                        embedding = self.recognizer.get_embedding(frame, kps)
                        
                        name, similarity = self.face_db.search(embedding, self.similarity_thresh.get())
                        
                        if name != "Unknown":
                            if name not in colors:
                                colors[name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                            draw_bbox_info(frame, bbox, similarity=similarity, name=name, color=colors[name])
                        else:
                            draw_bbox(frame, bbox, (255, 0, 0))
                except Exception as frame_error:
                    print(f"Error processing frame {frame_count}: {frame_error}")
                    # Continue with unprocessed frame
                
                cap_out.write(frame)
                frame_count += 1
                
                # Update progress periodically
                if frame_count % 30 == 0:  # Every 30 frames
                    progress_percent = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    frame_count_captured = frame_count  # Capture the value
                    self.root.after(0, lambda p=progress_percent, fc=frame_count_captured: 
                                  self.video_status_var.set(f"Processing... {p:.1f}% ({fc}/{total_frames} frames)"))
            
            print(f"Video processing completed. Processed {frame_count} frames.")
                    
        except Exception as e:
            print(f"Error during video processing: {e}")
            raise
        finally:
            cap.release()
            if cap_out:
                cap_out.release()
        
    def start_webcam(self):
        if self.webcam_active:
            return
            
        if not self.detector or not self.recognizer or not self.face_db:
            messagebox.showerror("Error", "Models not loaded")
            return
            
        self.webcam_active = True
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            self.webcam_active = False
            return
            
        self.update_webcam()
        
    def stop_webcam(self):
        self.webcam_active = False
        if self.cap:
            self.cap.release()
        self.webcam_label.configure(image="", text="Webcam stopped")
        
    def update_webcam(self):
        if not self.webcam_active or not self.cap:
            return
            
        ret, frame = self.cap.read()
        if ret:
            # Process frame
            colors = {}
            bboxes, kpss = self.detector.detect(frame, max_num=0)
            
            for bbox, kps in zip(bboxes, kpss):
                *bbox, conf_score = bbox.astype(np.int32)
                embedding = self.recognizer.get_embedding(frame, kps)
                
                name, similarity = self.face_db.search(embedding, self.similarity_thresh.get())
                
                if name != "Unknown":
                    if name not in colors:
                        colors[name] = (0, 255, 0)  # Green for known faces
                    draw_bbox_info(frame, bbox, similarity=similarity, name=name, color=colors[name])
                else:
                    draw_bbox(frame, bbox, (0, 0, 255))  # Red for unknown faces
            
            # Convert to PIL and display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            pil_image = pil_image.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_image)
            
            self.webcam_label.configure(image=photo, text="")
            self.webcam_label.image = photo
            
        # Schedule next update
        if self.webcam_active:
            self.root.after(30, self.update_webcam)


def main():
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    
    def on_closing():
        app.stop_webcam()
        app.stop_video()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()