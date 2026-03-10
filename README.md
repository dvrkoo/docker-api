# FaceForensics Detector Docker API

Containerized image deepfake detection service (ManTraNet-based), built with the same operational pattern as the other repos:
- watch input folder
- process automatically
- write outputs

## What it does

- Watches `input/` for new images.
- Runs image-level deepfake detection.
- Writes outputs to `output/`.
- Supports CPU, CUDA, and Apple Silicon profile image variants.

## Prebuilt images (GHCR)

Published from GitHub Actions on pushes to `main`.

Base image path:

```text
ghcr.io/dvrkoo/docker-api/faceforensics-detector
```

Main tags:
- `latest` (CPU)
- `latest-cuda`
- `latest-mps`

## Run with docker-compose

Create folders:

```bash
mkdir -p input output logs
```

### CPU

```bash
docker compose up faceforensics-detector-cpu
```

### CUDA

```bash
docker compose --profile cuda up faceforensics-detector-cuda
```

### MPS profile image (Apple Silicon)

```bash
docker compose --profile mps up faceforensics-detector-mps
```

Notes:
- `latest-mps` is a `linux/arm64` profile image.
- Native macOS execution can still be preferable for some MPS workloads.

## Run prebuilt image directly

### CPU

```bash
docker run -d \
  --name faceforensics-detector-cpu \
  -v $(pwd)/input:/data/input \
  -v $(pwd)/output:/data/output \
  -v $(pwd)/logs:/data/logs \
  -e WATCH_FOLDER=/data/input \
  -e OUTPUT_FOLDER=/data/output \
  -e FORCE_CPU=true \
  ghcr.io/dvrkoo/docker-api/faceforensics-detector:latest
```

### CUDA

```bash
docker run -d \
  --name faceforensics-detector-cuda \
  --gpus all \
  -v $(pwd)/input:/data/input \
  -v $(pwd)/output:/data/output \
  -v $(pwd)/logs:/data/logs \
  -e WATCH_FOLDER=/data/input \
  -e OUTPUT_FOLDER=/data/output \
  -e NVIDIA_VISIBLE_DEVICES=all \
  ghcr.io/dvrkoo/docker-api/faceforensics-detector:latest-cuda
```

## Environment variables

- `WATCH_FOLDER` default `/data/input`
- `OUTPUT_FOLDER` default `/data/output`
- `LOG_FILE` default `/data/logs/app.log`
- `FORCE_CPU` default `false`

## Faster rebuilds

BuildKit is supported by Dockerfiles. Use:

```bash
DOCKER_BUILDKIT=1 docker compose build
```

## Testing

Install dev dependencies and run tests:

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## CI/CD

On each push/PR:
- unit tests
- CPU docker build + smoke test

On pushes to `main`:
- publish CPU/CUDA/MPS images to GHCR

Security scans run in a separate workflow (`security-scan.yml`).
