FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libopencv-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libx11-dev \
    liblapack-dev \
    libblas-dev \
    libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install uv for faster package installation
RUN pip install --no-cache-dir uv

# Install PyTorch CPU wheels using uv (10-100x faster than pip)
RUN uv pip install --system --no-cache \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

# Install minimal dependencies that torch/torchvision/torchaudio need
RUN uv pip install --system --no-cache \
      filelock sympy networkx jinja2 fsspec requests

RUN python - << 'EOF'
from pathlib import Path
req_path = Path('requirements.txt')
lines = req_path.read_text().splitlines()
skip = {'torch', 'torchvision', 'torchaudio'}
filtered = [l for l in lines if not any(l.startswith(name + '==') for name in skip) and l.strip()]
Path('requirements-no-torch.txt').write_text("\n".join(filtered) + "\n")
EOF

RUN uv pip install --system --no-cache -r requirements-no-torch.txt

COPY . .

ENV WATCH_FOLDER=/data/input
ENV OUTPUT_FOLDER=/data/output
ENV LOG_FILE=/data/logs/app.log
ENV FORCE_CPU=true

RUN mkdir -p /data/input /data/output /data/logs

VOLUME ["/data/input", "/data/output", "/data/logs"]

CMD ["python", "app.py"]
