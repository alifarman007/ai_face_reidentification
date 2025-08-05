# FAISS Face Database API

A REST API service for managing the FAISS face recognition database remotely. This service provides a secure way to interact with your face database without modifying the main AI Face Re-identification project.

## Features

- **Remote Database Access**: Manage your face database from anywhere
- **AI Model Integration**: Uses the same SCRFD + ArcFace models as the main project
- **Safe Operations**: Doesn't modify existing project files
- **REST API**: Standard HTTP endpoints for easy integration
- **CORS Enabled**: Can be accessed from web browsers
- **Async Support**: High-performance async operations

## API Endpoints

### Core Endpoints

- `GET /` - API information and status
- `GET /health` - Health check endpoint

### Person Management

- `GET /persons` - List all persons in database
- `GET /persons/{name}` - Get specific person details
- `POST /persons` - Add new person with face image
- `DELETE /persons/{name}` - Remove person from database

### Database Operations

- `POST /database/rebuild` - Rebuild database from face images
- `GET /database/stats` - Get database statistics

## Installation

1. **Prerequisites**: Ensure the main AI Face Re-identification project is properly set up with:
   - AI models downloaded in `../weights/` directory
   - Python environment with all main project dependencies

2. **Install API dependencies**:
   ```bash
   cd "FAISS database Fastapi"
   pip install -r requirements.txt
   ```

## Usage

### Starting the API Server

```bash
cd "FAISS database Fastapi"
python main.py
```

The API will be available at: `http://localhost:8000`

### API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Example API Calls

#### List all persons
```bash
curl -X GET "http://localhost:8000/persons"
```

#### Add a new person
```bash
curl -X POST "http://localhost:8000/persons" \
  -H "Content-Type: multipart/form-data" \
  -F "name=John_Doe" \
  -F "image=@path/to/john_photo.jpg"
```

#### Get person details
```bash
curl -X GET "http://localhost:8000/persons/John_Doe"
```

#### Delete a person
```bash
curl -X DELETE "http://localhost:8000/persons/John_Doe"
```

#### Get database statistics
```bash
curl -X GET "http://localhost:8000/database/stats"
```

### Python Client Example

```python
import requests

# API base URL
base_url = "http://localhost:8000"

# List all persons
response = requests.get(f"{base_url}/persons")
persons = response.json()
print(f"Found {len(persons)} persons in database")

# Add a new person
with open("photo.jpg", "rb") as f:
    files = {"image": f}
    data = {"name": "Alice"}
    response = requests.post(f"{base_url}/persons", files=files, data=data)
    print(response.json())

# Get database stats
response = requests.get(f"{base_url}/database/stats")
stats = response.json()
print(f"Database contains {stats['person_count']} persons with {stats['embedding_count']} embeddings")
```

## Configuration

The API automatically detects the main project structure and uses:

- **Models**: `../weights/det_10g.onnx` and `../weights/w600k_r50.onnx`
- **Face Images**: `../assets/faces/`
- **Database**: `../database/face_database/`

## Security Considerations

- **Local Network**: By default, the API binds to `0.0.0.0:8000` for network access
- **No Authentication**: Currently no authentication - add authentication for production use
- **CORS**: Enabled for all origins - restrict in production

## Integration with Main Project

This API service:
- ✅ **Does NOT modify** any existing project files
- ✅ **Uses existing** AI models and database
- ✅ **Follows same logic** as the main project for face processing
- ✅ **Safe concurrent access** - can run alongside the main application

## Error Handling

The API provides detailed error responses:

- `400 Bad Request` - Invalid input data
- `404 Not Found` - Person not found
- `500 Internal Server Error` - Processing errors
- `503 Service Unavailable` - Models not loaded

## Development

To run in development mode with auto-reload:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Project Structure

```
FAISS database Fastapi/
├── main.py              # FastAPI application
├── services/
│   ├── __init__.py
│   └── face_service.py  # Business logic layer
├── requirements.txt     # API dependencies
└── README.md           # This file
```

## Logging

The API uses the same logging configuration as the main project. Logs will appear in the same log files if file logging is enabled.

## Troubleshooting

1. **Models not loading**: Ensure the main project's model files are downloaded
2. **Import errors**: Make sure you're running from the correct directory
3. **Database not found**: The main project should be initialized first
4. **Port conflicts**: Change the port in `main.py` if 8000 is in use