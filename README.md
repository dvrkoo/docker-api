# FaceForensics Detector - Docker API

This is a containerized version of the FaceForensics detection system that watches a folder for new images and automatically processes them to detect deepfakes.

## Features

- Watches a folder for new image files
- Automatically detects deepfakes using MantraNet model
- Outputs processed results with detection masks and annotations
- Supports both GPU (CUDA) and CPU modes
- Image detection only (supports .jpg, .png formats)

## Prerequisites

- Docker
- Docker Compose
- (Optional) NVIDIA GPU with CUDA support for faster processing

## Performance

All Docker builds use **[uv](https://github.com/astral-sh/uv)** instead of pip for **10-100x faster** package installation:
- PyTorch installation: ~9s (vs ~60-90s with pip)
- Total build time: **4-6x faster** than traditional pip-based builds

## Directory Structure

```
docker-api/
├── Dockerfile              # CPU build (default)
├── Dockerfile.cuda         # CUDA/GPU build
├── Dockerfile.mps          # Apple Silicon build
├── docker-compose.yml      # Multi-variant orchestration
├── app.py                  # Main application
├── requirements.txt        # Python dependencies
├── test_native.sh          # Native execution script (M1/M2)
├── input/                  # Created automatically - place files here
├── output/                 # Created automatically - processed results
├── playerModules/          # MantraNet model code
├── trained_models/         # Pre-trained model weights
└── README.md
```

## Quick Start (Using Pre-built Image)

**Pull the latest image from GitHub Container Registry:**

```bash
docker pull ghcr.io/dvrkoo/docker-api/faceforensics-detector:latest
```

**Run the container:**

```bash
docker run -d \
  -v $(pwd)/input:/data/input \
  -v $(pwd)/output:/data/output \
  -e WATCH_FOLDER=/data/input \
  -e OUTPUT_FOLDER=/data/output \
  ghcr.io/dvrkoo/docker-api/faceforensics-detector:latest
```

## Setup (Building from Source)

1. Create input/output directories:
```bash
mkdir -p input output
```

2. Build the Docker image:
```bash
docker-compose build
```

## Usage

This project provides **three build variants** optimized for different hardware:

| Variant | Hardware | Dockerfile | Use Case |
|---------|----------|------------|----------|
| **CPU** | Any x86_64 | `Dockerfile` | Cloud servers, production |
| **CUDA** | NVIDIA GPU | `Dockerfile.cuda` | GPU-accelerated inference |
| **MPS** | Apple Silicon | `Dockerfile.mps` | M1/M2 Macs (native only) |

### CPU Mode (default - works everywhere)

```bash
docker-compose up faceforensics-api-cpu
```

**Image size:** ~600-800MB  
**Performance:** Moderate, suitable for production  
**Requirements:** None

### CUDA Mode (NVIDIA GPU acceleration)

```bash
docker-compose --profile cuda up faceforensics-api-cuda
```

**Image size:** ~4-5GB  
**Performance:** Fast, GPU-accelerated  
**Requirements:** 
- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime installed
- CUDA-compatible drivers

**Install NVIDIA Docker runtime:**
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### MPS Mode (Apple Silicon - Native Execution Recommended)

**Note:** MPS in Docker has known issues causing segfaults. Use native execution instead:

```bash
# Native execution (recommended for M1/M2 Macs)
./test_native.sh
```

If you still want to try Docker on Apple Silicon:
```bash
docker-compose --profile mps up faceforensics-api-mps
```

**Image size:** ~1-1.5GB  
**Performance:** Varies (native is fast, Docker may crash)  
**Requirements:** Apple Silicon Mac (M1/M2/M3)

### Running in Background

```bash
# CPU mode
docker-compose up -d faceforensics-api-cpu

# CUDA mode
docker-compose --profile cuda up -d faceforensics-api-cuda
```

### Stopping the Service

```bash
docker-compose down
```

## How It Works

1. The container starts and monitors the `./input` folder
2. Drop an image (.jpg, .png) into the `./input` folder
3. The system automatically processes the image using MantraNet
4. Results appear in the `./output` folder with detection masks showing manipulated regions

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
- Ensure input files are valid images (.jpg, .png)
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
- MantraNet model integration
- File type detection and routing
- GPU memory management
- File watching and queue management

## Notes

- First run may take longer as models are loaded
- GPU mode requires NVIDIA Docker runtime installed
- The container uses read-only mounts for model files to prevent accidental modifications

## CI/CD Status

![Docker Build](https://github.com/dvrkoo/docker-api/actions/workflows/docker-build-test.yml/badge.svg)

## Published Docker Images

Pre-built Docker images are automatically published to GitHub Container Registry on every commit to main:

- **Latest:** `ghcr.io/dvrkoo/docker-api/faceforensics-detector:latest`
- **Main branch:** `ghcr.io/dvrkoo/docker-api/faceforensics-detector:main`
- **Specific commits:** `ghcr.io/dvrkoo/docker-api/faceforensics-detector:main-<commit-sha>`

### Available Tags

View all available tags at:
**https://github.com/dvrkoo/docker-api/pkgs/container/docker-api%2Ffaceforensics-detector**

### Pull and Run Pre-built Images

#### CPU Version (Works Everywhere)

**Pull the image:**
```bash
docker pull ghcr.io/dvrkoo/docker-api/faceforensics-detector:latest
```

**Run the container:**
```bash
docker run -d \
  --name faceforensics-cpu \
  -v $(pwd)/input:/data/input \
  -v $(pwd)/output:/data/output \
  -e WATCH_FOLDER=/data/input \
  -e OUTPUT_FOLDER=/data/output \
  -e FORCE_CPU=true \
  ghcr.io/dvrkoo/docker-api/faceforensics-detector:latest
```

**Use case:** Production servers, cloud environments, or any system without NVIDIA GPU  
**Performance:** Moderate speed, suitable for most workloads

#### CUDA Version (NVIDIA GPU Acceleration)

**Pull the image:**
```bash
docker pull ghcr.io/dvrkoo/docker-api/faceforensics-detector:cuda
```

**Run the container:**
```bash
docker run -d \
  --name faceforensics-cuda \
  --gpus all \
  -v $(pwd)/input:/data/input \
  -v $(pwd)/output:/data/output \
  -e WATCH_FOLDER=/data/input \
  -e OUTPUT_FOLDER=/data/output \
  -e NVIDIA_VISIBLE_DEVICES=all \
  ghcr.io/dvrkoo/docker-api/faceforensics-detector:cuda
```

**Requirements:**
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

**Use case:** GPU-accelerated servers for faster inference  
**Performance:** 5-10x faster than CPU, ideal for high-throughput workloads

**Verify GPU access:**
```bash
docker run --rm --gpus all ghcr.io/dvrkoo/docker-api/faceforensics-detector:cuda nvidia-smi
```

### Image Details

**CPU Variant (default):**
- **Platform:** linux/amd64
- **Size:** ~600-800MB
- **Base:** Python 3.11-slim
- **PyTorch:** CPU-optimized (2.1.0+cpu)

**CUDA Variant:**
- **Platform:** linux/amd64
- **Size:** ~4-5GB
- **Base:** nvidia/cuda:12.1.0-runtime-ubuntu22.04
- **PyTorch:** CUDA-enabled (2.1.0+cu121)

**MPS Variant:**
- **Platform:** linux/arm64
- **Size:** ~1-1.5GB
- **Base:** Python 3.11
- **PyTorch:** ARM64 native with MPS support
