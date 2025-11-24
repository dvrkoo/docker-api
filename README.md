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

## Notes

- First run may take longer as models are loaded
- GPU mode requires NVIDIA Docker runtime installed
- The container uses read-only mounts for model files to prevent accidental modifications
