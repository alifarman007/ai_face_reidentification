from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import os
import sys
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import asyncio

# Add parent directory to path to import from main project
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from services.face_service import FaceService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize face service
face_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global face_service
    try:
        face_service = FaceService()
        await face_service.initialize()
        logger.info("Face service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize face service: {e}")
    
    yield
    
    # Shutdown (cleanup if needed)
    pass

app = FastAPI(
    title="FAISS Face Database API",
    description="REST API for managing FAISS face recognition database",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class PersonResponse(BaseModel):
    name: str
    embedding_count: int
    id: Optional[str] = None
    created_at: Optional[str] = None

class DatabaseStats(BaseModel):
    person_count: int
    embedding_count: int
    video_count: int

class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Any = None

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "FAISS Face Database API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if face_service is None:
        raise HTTPException(status_code=503, detail="Face service not initialized")
    return {"status": "healthy", "service": "face_database_api"}

@app.get("/persons", response_model=List[PersonResponse])
async def get_all_persons():
    """Get all persons in the database"""
    try:
        if face_service is None:
            raise HTTPException(status_code=503, detail="Face service not initialized")
        
        persons = await face_service.get_all_persons()
        return persons
    except Exception as e:
        logger.error(f"Error getting persons: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/persons/{name}", response_model=PersonResponse)
async def get_person(name: str):
    """Get specific person details"""
    try:
        if face_service is None:
            raise HTTPException(status_code=503, detail="Face service not initialized")
        
        person = await face_service.get_person(name)
        if person is None:
            raise HTTPException(status_code=404, detail=f"Person '{name}' not found")
        
        return person
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting person {name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/persons", response_model=ApiResponse)
async def add_person(name: str, image: UploadFile = File(...)):
    """Add new person with face image using AI models"""
    try:
        if face_service is None:
            raise HTTPException(status_code=503, detail="Face service not initialized")
        
        if not name or not name.strip():
            raise HTTPException(status_code=400, detail="Name is required")
        
        # Validate image file
        if not image.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            raise HTTPException(status_code=400, detail="Only JPG, JPEG, and PNG files are supported")
        
        # Read image data
        image_data = await image.read()
        
        # Process the image and add person
        result = await face_service.add_person(name.strip(), image_data, image.filename)
        
        if result["success"]:
            return ApiResponse(
                success=True,
                message=f"Person '{name}' added successfully",
                data={"name": name, "embeddings_added": result.get("embeddings_count", 1)}
            )
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding person {name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/persons/{name}", response_model=ApiResponse)
async def delete_person(name: str):
    """Remove person from database"""
    try:
        if face_service is None:
            raise HTTPException(status_code=503, detail="Face service not initialized")
        
        result = await face_service.delete_person(name)
        
        if result["success"]:
            return ApiResponse(
                success=True,
                message=f"Person '{name}' deleted successfully",
                data={"name": name}
            )
        else:
            if result["message"] == "Person not found":
                raise HTTPException(status_code=404, detail=f"Person '{name}' not found")
            else:
                raise HTTPException(status_code=400, detail=result["message"])
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting person {name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/database/rebuild", response_model=ApiResponse)
async def rebuild_database():
    """Rebuild the face database from face images"""
    try:
        if face_service is None:
            raise HTTPException(status_code=503, detail="Face service not initialized")
        
        result = await face_service.rebuild_database()
        
        return ApiResponse(
            success=True,
            message="Database rebuilt successfully",
            data=result
        )
        
    except Exception as e:
        logger.error(f"Error rebuilding database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/database/stats", response_model=DatabaseStats)
async def get_database_stats():
    """Get database statistics"""
    try:
        if face_service is None:
            raise HTTPException(status_code=503, detail="Face service not initialized")
        
        stats = await face_service.get_database_stats()
        return DatabaseStats(**stats)
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/database/reload", response_model=ApiResponse)
async def reload_database():
    """Reload the face database from disk to sync with external changes"""
    try:
        if face_service is None:
            raise HTTPException(status_code=503, detail="Face service not initialized")
        
        result = await face_service.reload_database()
        
        if result["success"]:
            return ApiResponse(
                success=True,
                message=result["message"],
                data={
                    "old_count": result["old_count"],
                    "new_count": result["new_count"]
                }
            )
        else:
            raise HTTPException(status_code=400, detail=result["message"])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reloading database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )