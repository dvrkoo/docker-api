# Dockerfile.local - Optimized for local ARM64 (Apple Silicon) development
FROM python:3.11

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
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

# Install all dependencies from requirements.txt using uv (10-100x faster than pip)
# PyPI will automatically select ARM64-compatible wheels
RUN uv pip install --system --no-cache -r requirements.txt

COPY . .

ENV WATCH_FOLDER=/data/input
ENV OUTPUT_FOLDER=/data/output
ENV LOG_FILE=/data/logs/app.log
ENV FORCE_CPU=true

RUN mkdir -p /data/input /data/output /data/logs

VOLUME ["/data/input", "/data/output", "/data/logs"]

CMD ["python", "app.py"]
