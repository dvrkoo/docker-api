# Local Development on Apple Silicon (M1/M2/M3)

## Known Issue

MantraNet inference crashes with a segmentation fault (exit code 139) when running inside Docker on ARM64 (Apple Silicon) architectures, even though the same code works perfectly when run natively on macOS.

This is due to incompatibilities between:
- ARM64 Docker containerization
- PyTorch CPU inference operations
- System BLAS/OpenBLAS libraries

The production Docker image targets `linux/amd64` (x86_64) and works correctly.

## Recommended Local Testing Approach

**Instead of using Docker locally**, run the application natively on your Mac:

### Setup

1. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application directly:
```bash
python app.py
```

The application will:
- Watch `./input` for new images
- Process them with MantraNet
- Save results to `./output`

### Testing

Drop a test image into `./input`:
```bash
cp /path/to/test/image.jpg input/
```

Or create a random test image:
```bash
python3 - << 'EOF'
from pathlib import Path
from PIL import Image
import numpy as np
import time

inp = Path('input')
inp.mkdir(exist_ok=True)
img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
img.save(inp / f'test_{int(time.time())}.jpg')
EOF
```

Check the console output and `./output` folder for results.

## Docker Testing (x86_64 only)

If you need to test the Docker image that will be deployed:

### Option 1: Use Docker Buildx for cross-platform build

```bash
docker buildx build --platform linux/amd64 -t faceforensics-detector:amd64 --load .
```

Then run it (will use QEMU emulation, slower but works):
```bash
docker run -d \
  --name faceforensics-detector-test \
  --platform linux/amd64 \
  -v $(pwd)/input:/data/input \
  -v $(pwd)/output:/data/output \
  -e WATCH_FOLDER=/data/input \
  -e OUTPUT_FOLDER=/data/output \
  faceforensics-detector:amd64
```

### Option 2: Pull and test the published GHCR image

Once CI builds and pushes the image:
```bash
docker pull ghcr.io/dvrkoo/docker-api/faceforensics-detector:latest

docker run -d \
  --name faceforensics-detector-ghcr \
  -v $(pwd)/input:/data/input \
  -v $(pwd)/output:/data/output \
  -e WATCH_FOLDER=/data/input \
  -e OUTPUT_FOLDER=/data/output \
  ghcr.io/dvrkoo/docker-api/faceforensics-detector:latest
```

### Option 3: Test on an x86_64 machine

Deploy and test on a Linux x86_64 server or VM where the target architecture matches.

## Summary

- **✅ Native macOS:** Works perfectly, use this for local development
- **❌ Docker on ARM64:** Known segfault issue, not recommended
- **✅ Docker on x86_64:** Production target, works correctly
- **✅ GHCR published image:** Use this for production deployments
