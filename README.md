# AI Face Re-Identification System with FAISS, ArcFace & SCRFD

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/alifarman007/ai_face_reidentification)

<h5 align="center"> If you like our project, please give star ⭐ on GitHub for the latest updates.</h5>

This repository implements a comprehensive face re-identification system using SCRFD for face detection and ArcFace for face recognition. It supports multiple interfaces including command-line, Gradio web interface, and supports inference from webcam or video sources with FAISS vector database integration.

## Features

- [x] **FAISS Vector Database Integration**: Enables fast and scalable face re-identification using a FAISS index built from facial embeddings. Faces must be placed in the `assets/faces/` directory.
- [x] **Multiple Interface Options**: 
  - Command-line interface for batch processing
  - **Gradio Web Interface** for easy interaction and testing
  - Real-time webcam and video processing
- [x] **Optimized Models**: Smaller versions of SCRFD face detection models for different performance requirements
- [x] **Face Detection**: Utilizes [Sample and Computation Redistribution for Efficient Face Detection](https://arxiv.org/abs/2105.04714) (SCRFD) for efficient and accurate face detection.
  - Available models: SCRFD 500M (2.41 MB), SCRFD 2.5G (3.14 MB), SCRFD 10G (16.1 MB)
- [x] **Face Recognition**: Employs [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698) for robust face recognition.
  - Available models: ArcFace MobileFace (12.99 MB), ArcFace ResNet-50 (166 MB)
- [x] **Real-Time Processing**: Supports both webcam and video file input for real-time processing.
- [x] **Person Management**: Add, update, and manage known faces through the web interface.
- [x] **Configuration Management**: Environment-based configuration with `.env` support.

Project folder structure:

```
├── assets/
│   ├── demo.mp4
│   ├── in_video.mp4
│   └── faces/
│       ├── Binoy.jpg
│       ├── Sadhon.jpg
│       ├── Tokon.jpg
│       └── ... (your face images)
├── database/
│   ├── __init__.py
│   └── face_db.py
├── models/
│   ├── __init__.py
│   ├── scrfd.py
│   └── arcface.py
├── weights/
│   ├── det_10g.onnx
│   ├── det_2.5g.onnx
│   ├── det_500m.onnx
│   ├── w600k_r50.onnx
│   └── w600k_mbf.onnx
├── utils/
│   ├── logging.py
│   └── helpers.py
├── main.py
├── gradio_interface.py
├── .env.example
├── requirements.txt
├── requirements_web.txt
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/alifarman007/ai_face_reidentification.git
cd ai_face_reidentification
```

2. Install the required dependencies:

For basic functionality:
```bash
pip install -r requirements.txt
```

For web interface and additional features:
```bash
pip install -r requirements_web.txt
```

3. Configure environment variables (optional):

```bash
cp .env.example .env
# Edit .env file with your preferred settings
```

4. Download weight files:

   a) Download weights from following links:

   | Model              | Weights                                                                                                   | Size     | Type             |
   | ------------------ | --------------------------------------------------------------------------------------------------------- | -------- | ---------------- |
   | SCRFD 500M         | [det_500m.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_500m.onnx)   | 2.41 MB  | Face Detection   |
   | SCRFD 2.5G         | [det_2.5g.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_2.5g.onnx)   | 3.14 MB  | Face Detection   |
   | SCRFD 10G          | [det_10g.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_10g.onnx)     | 16.1 MB  | Face Detection   |
   | ArcFace MobileFace | [w600k_mbf.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_mbf.onnx) | 12.99 MB | Face Recognition |
   | ArcFace ResNet-50  | [w600k_r50.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_r50.onnx) | 166 MB   | Face Recognition |

   b) Run below command to download weights to `weights` directory (linux):

   ```bash
   sh download.sh
   ```

5. Put target faces into `assets/faces` folder

```
assets/faces/
    ├── Binoy.jpg
    ├── Sadhon.jpg
    ├── Tokon.jpg
    └── ... (your face images)
```

The file names (without extension) will be used as person identifiers during real-time inference.

## Usage

### 1. Command Line Interface

For video file processing:
```bash
python main.py --source assets/in_video.mp4
```

For webcam processing:
```bash
python main.py --source 0
```

### 2. Gradio Web Interface

Launch the interactive web interface:
```bash
python gradio_interface.py
```

Then open your browser and navigate to the provided URL (typically `http://localhost:7860`).

The web interface allows you to:
- Add new persons by uploading their photos
- Update the face database
- Process video files or use webcam in real-time
- Manage face recognition settings

`main.py` arguments:

```
usage: main.py [-h] [--det-weight DET_WEIGHT] [--rec-weight REC_WEIGHT] [--similarity-thresh SIMILARITY_THRESH] [--confidence-thresh CONFIDENCE_THRESH]
               [--faces-dir FACES_DIR] [--source SOURCE] [--max-num MAX_NUM]

Face Detection-and-Recognition

options:
  -h, --help            show this help message and exit
  --det-weight DET_WEIGHT
                        Path to detection model
  --rec-weight REC_WEIGHT
                        Path to recognition model
  --similarity-thresh SIMILARITY_THRESH
                        Similarity threshold between faces
  --confidence-thresh CONFIDENCE_THRESH
                        Confidence threshold for face detection
  --faces-dir FACES_DIR
                        Path to faces stored dir
  --source SOURCE       Video file or video camera source. i.e 0 - webcam
  --max-num MAX_NUM     Maximum number of face detections from a frame
```

## Configuration

The system can be configured through environment variables or the `.env` file:

```bash
# Model Configuration
DET_WEIGHT_PATH=./weights/det_10g.onnx
REC_WEIGHT_PATH=./weights/w600k_r50.onnx

# Thresholds
SIMILARITY_THRESHOLD=0.4
CONFIDENCE_THRESHOLD=0.5

# Directories
FACES_DIR=./assets/faces
DATABASE_PATH=./database/face_database
```

## Interface Options

The system provides multiple ways to interact with the face recognition system:

1. **Command Line Interface**: Traditional CLI for batch processing and automation
2. **Gradio Web Interface**: User-friendly web interface for interactive use
3. **Real-time Processing**: Both webcam and video file processing support

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).
