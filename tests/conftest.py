"""
Pytest configuration file with shared fixtures for testing the FaceForensics detector.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to Python path to allow importing app module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Mock playerModules before importing app to avoid missing dependency errors
sys.modules['playerModules'] = MagicMock()
sys.modules['playerModules.mantranet'] = MagicMock()
sys.modules['playerModules.model_functions'] = MagicMock()

import pytest
import numpy as np
from PIL import Image
import tempfile
import os


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def test_image():
    """Create a simple test image (RGB)."""
    # Create a 224x224 RGB image with random data
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='RGB')
    return img


@pytest.fixture
def test_image_path(temp_dir, test_image):
    """Save test image to a temporary file and return the path."""
    img_path = temp_dir / "test_image.jpg"
    test_image.save(img_path)
    return str(img_path)


@pytest.fixture
def test_image_grayscale():
    """Create a grayscale test image."""
    img_array = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='L')
    return img


@pytest.fixture
def test_image_rgba():
    """Create an RGBA test image."""
    img_array = np.random.randint(0, 255, (224, 224, 4), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='RGBA')
    return img


@pytest.fixture
def mock_mantra_model(mocker):
    """Mock the MantraNet model to avoid loading actual weights."""
    mock_model = mocker.MagicMock()
    return mock_model


@pytest.fixture
def mock_face():
    """Create a mock dlib face object."""
    class MockFace:
        def __init__(self):
            self._left = 50
            self._top = 50
            self._width = 100
            self._height = 100
        
        def left(self):
            return self._left
        
        def top(self):
            return self._top
        
        def width(self):
            return self._width
        
        def height(self):
            return self._height
    
    return MockFace()


@pytest.fixture
def mock_predictions():
    """Create mock prediction arrays."""
    return [
        [0.1, 0.2, 0.3, 0.8, 0.1],  # Example predictions for 5 models
    ]


@pytest.fixture
def mock_predictions_genuine():
    """Create mock predictions indicating genuine face."""
    return [
        [0.1, 0.2, 0.3, 0.4, 0.1],  # All below 0.5
    ]


@pytest.fixture
def test_frame():
    """Create a test video frame (RGB numpy array)."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def input_output_dirs(temp_dir):
    """Create input and output directories for testing."""
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    return str(input_dir), str(output_dir)


@pytest.fixture(autouse=True)
def mock_environment_variables(monkeypatch, temp_dir):
    """Mock environment variables for testing."""
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    monkeypatch.setenv("WATCH_FOLDER", str(input_dir))
    monkeypatch.setenv("OUTPUT_FOLDER", str(output_dir))
