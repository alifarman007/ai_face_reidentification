import os
import sys
import cv2
import numpy as np
import logging
import tempfile
import asyncio
from typing import Dict, List, Any, Optional
from PIL import Image
import io

# Add parent directory to path to import from main project
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from database.face_db import FaceDatabase
    from models import SCRFD, ArcFace
    from utils.logging import setup_logging
except ImportError as e:
    logging.error(f"Failed to import from main project: {e}")
    raise

logger = logging.getLogger(__name__)

class FaceService:
    """Service layer for managing face database operations safely"""
    
    def __init__(self):
        # Configuration paths (relative to main project)
        self.project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        self.detection_model_path = os.path.join(self.project_root, "weights", "det_10g.onnx")
        self.recognition_model_path = os.path.join(self.project_root, "weights", "w600k_r50.onnx")
        self.faces_dir = os.path.join(self.project_root, "assets", "faces")
        self.db_path = os.path.join(self.project_root, "database", "face_database")
        
        # Models (will be initialized in initialize method)
        self.detector = None
        self.recognizer = None
        self.face_db = None
        
        # Settings
        self.similarity_thresh = 0.4
        self.confidence_thresh = 0.5
        
    async def initialize(self):
        """Initialize AI models and database connection"""
        try:
            # Setup logging
            setup_logging(log_to_file=True)
            
            # Check if model files exist
            if not os.path.exists(self.detection_model_path):
                raise FileNotFoundError(f"Detection model not found: {self.detection_model_path}")
            if not os.path.exists(self.recognition_model_path):
                raise FileNotFoundError(f"Recognition model not found: {self.recognition_model_path}")
            
            # Load models
            logger.info("Loading AI models...")
            self.detector = SCRFD(self.detection_model_path, input_size=(640, 640), conf_thres=self.confidence_thresh)
            self.recognizer = ArcFace(self.recognition_model_path)
            
            # Initialize and load database
            self.face_db = FaceDatabase(db_path=self.db_path)
            self.face_db.load()
            
            # Ensure faces directory exists
            os.makedirs(self.faces_dir, exist_ok=True)
            
            logger.info("Face service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize face service: {e}")
            raise
    
    def _ensure_initialized(self):
        """Ensure service is properly initialized"""
        if self.detector is None or self.recognizer is None or self.face_db is None:
            raise RuntimeError("Face service not initialized. Call initialize() first.")
    
    async def get_all_persons(self) -> List[Dict[str, Any]]:
        """Get all persons from the database"""
        self._ensure_initialized()
        
        try:
            persons = self.face_db.get_all_persons()
            return persons
        except Exception as e:
            logger.error(f"Error getting all persons: {e}")
            raise
    
    async def get_person(self, name: str) -> Optional[Dict[str, Any]]:
        """Get specific person details"""
        self._ensure_initialized()
        
        try:
            persons = self.face_db.get_all_persons()
            for person in persons:
                if person['name'] == name:
                    return person
            return None
        except Exception as e:
            logger.error(f"Error getting person {name}: {e}")
            raise
    
    async def add_person(self, name: str, image_data: bytes, filename: str) -> Dict[str, Any]:
        """Add new person with face image using AI models"""
        self._ensure_initialized()
        
        try:
            # Save image to faces directory
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png']:
                return {"success": False, "message": "Unsupported image format"}
            
            # Use .jpg as default extension
            dest_path = os.path.join(self.faces_dir, f"{name}.jpg")
            
            # Convert image data to PIL Image and save
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save image
            image.save(dest_path, 'JPEG', quality=95)
            
            # Process the saved image with AI models
            cv_image = cv2.imread(dest_path)
            if cv_image is None:
                os.remove(dest_path)  # Clean up saved file
                return {"success": False, "message": "Failed to load saved image"}
            
            # Detect faces in the image
            bboxes, kpss = self.detector.detect(cv_image, max_num=1)
            
            if len(kpss) == 0:
                os.remove(dest_path)  # Clean up saved file
                return {"success": False, "message": "No face detected in the image"}
            
            # Extract face embedding
            embedding = self.recognizer.get_embedding(cv_image, kpss[0])
            
            # Add to database
            self.face_db.add_face(embedding, name)
            self.face_db.save()
            
            logger.info(f"Successfully added person: {name}")
            
            return {
                "success": True,
                "message": f"Person '{name}' added successfully",
                "embeddings_count": 1
            }
            
        except Exception as e:
            logger.error(f"Error adding person {name}: {e}")
            # Clean up saved file if exists
            dest_path = os.path.join(self.faces_dir, f"{name}.jpg")
            if os.path.exists(dest_path):
                try:
                    os.remove(dest_path)
                except:
                    pass
            return {"success": False, "message": f"Failed to add person: {str(e)}"}
    
    async def delete_person(self, name: str) -> Dict[str, Any]:
        """Delete person from database and remove their image"""
        self._ensure_initialized()
        
        try:
            # Check if person exists
            person = await self.get_person(name)
            if person is None:
                return {"success": False, "message": "Person not found"}
            
            # Delete from database
            deleted = self.face_db.delete_person(name)
            
            if deleted:
                # Save database changes
                self.face_db.save()
                
                # Remove image file if exists
                image_extensions = ['.jpg', '.jpeg', '.png']
                for ext in image_extensions:
                    image_path = os.path.join(self.faces_dir, f"{name}{ext}")
                    if os.path.exists(image_path):
                        try:
                            os.remove(image_path)
                            logger.info(f"Removed image file: {image_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove image file {image_path}: {e}")
                
                logger.info(f"Successfully deleted person: {name}")
                return {"success": True, "message": f"Person '{name}' deleted successfully"}
            else:
                return {"success": False, "message": "Failed to delete person from database"}
                
        except Exception as e:
            logger.error(f"Error deleting person {name}: {e}")
            return {"success": False, "message": f"Failed to delete person: {str(e)}"}
    
    async def rebuild_database(self) -> Dict[str, Any]:
        """Rebuild the entire face database from face images"""
        self._ensure_initialized()
        
        try:
            # Create new database instance
            self.face_db = FaceDatabase(db_path=self.db_path)
            
            processed_count = 0
            failed_count = 0
            failed_files = []
            
            # Process all image files in faces directory
            if os.path.exists(self.faces_dir):
                for filename in os.listdir(self.faces_dir):
                    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    
                    name = os.path.splitext(filename)[0]
                    image_path = os.path.join(self.faces_dir, filename)
                    
                    try:
                        # Load and process image
                        image = cv2.imread(image_path)
                        if image is None:
                            failed_files.append(f"{filename}: Failed to load image")
                            failed_count += 1
                            continue
                        
                        # Detect faces
                        bboxes, kpss = self.detector.detect(image, max_num=1)
                        if len(kpss) == 0:
                            failed_files.append(f"{filename}: No face detected")
                            failed_count += 1
                            continue
                        
                        # Extract embedding and add to database
                        embedding = self.recognizer.get_embedding(image, kpss[0])
                        self.face_db.add_face(embedding, name)
                        processed_count += 1
                        
                    except Exception as e:
                        failed_files.append(f"{filename}: {str(e)}")
                        failed_count += 1
            
            # Save the rebuilt database
            self.face_db.save()
            
            result = {
                "processed_count": processed_count,
                "failed_count": failed_count,
                "total_files": processed_count + failed_count
            }
            
            if failed_files:
                result["failed_files"] = failed_files
            
            logger.info(f"Database rebuilt: {processed_count} processed, {failed_count} failed")
            
            return result
            
        except Exception as e:
            logger.error(f"Error rebuilding database: {e}")
            raise
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        self._ensure_initialized()
        
        try:
            stats = self.face_db.get_database_stats()
            return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            raise
    
    async def reload_database(self) -> Dict[str, Any]:
        """Reload the face database from disk to sync with external changes"""
        self._ensure_initialized()
        
        try:
            # Create new database instance and load from disk
            old_count = self.face_db.index.ntotal if self.face_db else 0
            
            self.face_db = FaceDatabase(db_path=self.db_path)
            loaded = self.face_db.load()
            
            new_count = self.face_db.index.ntotal if loaded else 0
            
            if loaded:
                logger.info(f"Database reloaded successfully: {old_count} -> {new_count} faces")
                return {
                    "success": True,
                    "message": "Database reloaded successfully",
                    "old_count": old_count,
                    "new_count": new_count
                }
            else:
                logger.warning("No existing database found to reload")
                return {
                    "success": False,
                    "message": "No existing database found to reload"
                }
                
        except Exception as e:
            logger.error(f"Error reloading database: {e}")
            raise