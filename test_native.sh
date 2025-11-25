#!/bin/bash
set -e

echo "Testing native execution..."
echo "=========================="

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install deps if needed
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -q -r requirements.txt
fi

# Create test image
echo "Creating test image..."
python3 - << 'PYEOF'
from pathlib import Path
from PIL import Image
import numpy as np
import time

inp = Path('input')
inp.mkdir(exist_ok=True)
img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
test_file = inp / f'native_test_{int(time.time())}.jpg'
img.save(test_file)
print(f'✅ Created {test_file}')
PYEOF

echo ""
echo "Starting app (Ctrl+C to stop)..."
echo "Watch ./output for results"
python app.py
