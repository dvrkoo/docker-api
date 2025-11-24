# FaceForensics Detector - Docker API

This is a containerized version of the FaceForensics detection system that watches a folder for new images/videos and automatically processes them to detect deepfakes.

## Features

- Watches a folder for new image/video files
- Automatically detects deepfakes using multiple models:
  - MantraNet
  - FaceSwap
  - DeepFake
  - NeuralTextures
  - Face2Face
  - FaceShifter
- Outputs processed results with annotations
- Supports both GPU (CUDA) and CPU modes

## Prerequisites

- Docker
- Docker Compose
- (Optional) NVIDIA GPU with CUDA support for faster processing

## Directory Structure

```
docker-api/
├── Dockerfile
├── docker-compose.yml
├── app.py
├── requirements.txt
├── input/          # Created automatically - place files here
├── output/         # Created automatically - processed results
└── README.md
```

## Setup

1. Create input/output directories:
```bash
mkdir -p input output
```

2. Build the Docker image:
```bash
docker-compose build
```

## Usage

### CPU Mode (default)

```bash
docker-compose up faceforensics-api-cpu
```

### GPU Mode (requires NVIDIA GPU)

```bash
docker-compose --profile gpu up faceforensics-api
```

### Running in Background

```bash
docker-compose up -d faceforensics-api-cpu
```

### Stopping the Service

```bash
docker-compose down
```

## How It Works

1. The container starts and monitors the `./input` folder
2. Drop an image (.jpg, .png) or video (.mp4) into the `./input` folder
3. The system automatically processes the file
4. Results appear in the `./output` folder:
   - For images: Multiple output images showing detection masks
   - For videos: Annotated video with bounding boxes + JSON with predictions

## Environment Variables

You can customize the behavior by modifying the environment variables in `docker-compose.yml`:

- `WATCH_FOLDER`: Input folder path (default: `/data/input`)
- `OUTPUT_FOLDER`: Output folder path (default: `/data/output`)

## Logs

View logs in real-time:
```bash
docker-compose logs -f
```

## Troubleshooting

### Container won't start
- Ensure Docker is running
- Check that the parent `FaceForensicsTrainer` directory exists with all required models

### No output files
- Check container logs for errors
- Ensure input files are valid images (.jpg, .png) or videos (.mp4)
- Verify trained_models directory is accessible

### Slow processing
- Use GPU mode if available
- CPU mode is significantly slower but will work

## Testing

This project uses **pytest** for automated testing with GitHub Actions CI/CD integration.

### Running Tests Locally

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run all tests:
```bash
pytest
```

3. Run tests with coverage:
```bash
pytest --cov=. --cov-report=html
```

4. Run specific test file:
```bash
pytest tests/test_image_processing.py -v
```

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures and test configuration
├── test_image_processing.py  # Unit tests for image processing
└── test_video_processing.py  # Unit tests for video processing
```

### Continuous Integration

Every push and pull request triggers automated tests via GitHub Actions:

- ✅ **Unit Tests**: pytest with code coverage
- ✅ **Docker Build**: Validates Dockerfile builds successfully
- ✅ **Container Smoke Test**: Ensures container starts and runs
- ✅ **Security Scan**: Trivy vulnerability scanning
- ✅ **Multi-arch Build**: Builds for amd64 and arm64 (on main branch)

View the build status and coverage reports in the GitHub Actions tab.

### Test Coverage

Current test coverage focuses on:
- Image processing and format conversion
- Video processing and face detection
- File type detection and routing
- GPU memory management
- File watching and queue management

## Notes

- First run may take longer as models are loaded
- GPU mode requires NVIDIA Docker runtime installed
- The container uses read-only mounts for model files to prevent accidental modifications

## CI/CD Status

![Docker Build](https://github.com/YOUR_USERNAME/docker-api/actions/workflows/docker-build-test.yml/badge.svg)
