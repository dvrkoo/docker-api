FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
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

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir --no-deps \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

RUN python - << 'EOF'
from pathlib import Path
req_path = Path('requirements.txt')
lines = req_path.read_text().splitlines()
skip = {'torch', 'torchvision', 'torchaudio'}
filtered = [l for l in lines if not any(l.startswith(name + '==') for name in skip) and l.strip()]
Path('requirements-no-torch.txt').write_text("\n".join(filtered) + "\n")
EOF

RUN pip install --no-cache-dir -r requirements-no-torch.txt

COPY . .

ENV WATCH_FOLDER=/data/input
ENV OUTPUT_FOLDER=/data/output

RUN mkdir -p /data/input /data/output

VOLUME ["/data/input", "/data/output"]

CMD ["python", "app.py"]
