FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install only runtime dependencies (removed build tools to save space)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /root/.cache

# Copy only requirements first for better caching
COPY requirements.txt .

# Install dependencies
# Note: Using standard torch (not +cpu) for multi-architecture support
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    pytorch-lightning==2.1.0 \
    opencv-python-headless==4.8.1.78 \
    numpy==1.24.3 \
    Pillow==10.1.0 \
    watchdog==3.0.0 \
    matplotlib==3.8.0 \
    dlib-bin==19.24.6 && \
    rm -rf /root/.cache/pip

# Copy application files
COPY playerModules/ ./playerModules/
COPY trained_models/ ./trained_models/
COPY app.py .

ENV WATCH_FOLDER=/data/input
ENV OUTPUT_FOLDER=/data/output

RUN mkdir -p /data/input /data/output

VOLUME ["/data/input", "/data/output"]

CMD ["python", "app.py"]
