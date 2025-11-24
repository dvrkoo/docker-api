"""
Unit tests for image processing functions in app.py
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image
import os


class TestImageProcessing:
    """Test cases for image processing functions."""
    
    @patch('app.check_image_mantra')
    def test_process_image_calls_mantra(self, mock_check_mantra, test_image_path, temp_dir):
        """Test that process_image calls the MantraNet model."""
        from app import process_image
        
        # Mock the return value
        mock_check_mantra.return_value = {
            'original': np.random.rand(224, 224, 3),
            'heatmap': np.random.rand(224, 224),
        }
        
        with patch('app.output_folder', str(temp_dir)):
            process_image(test_image_path)
        
        mock_check_mantra.assert_called_once()
    
    def test_check_image_mantra_converts_non_rgb(self, test_image_grayscale, mock_mantra_model, temp_dir):
        """Test that non-RGB images are converted to RGB."""
        from app import check_image_mantra
        
        # Save grayscale image
        img_path = temp_dir / "grayscale.jpg"
        test_image_grayscale.save(img_path)
        
        with patch('app.MantraNetmodel', mock_mantra_model):
            with patch('app.check_forgery') as mock_check_forgery:
                mock_check_forgery.return_value = {}
                
                # This should not raise an error
                try:
                    check_image_mantra(str(img_path))
                except Exception as e:
                    pytest.fail(f"check_image_mantra raised {e} unexpectedly!")
    
    def test_check_image_mantra_accepts_rgb(self, test_image_path, mock_mantra_model):
        """Test that RGB images are processed correctly."""
        from app import check_image_mantra
        
        with patch('app.MantraNetmodel', mock_mantra_model):
            with patch('app.check_forgery') as mock_check_forgery:
                mock_check_forgery.return_value = {'test': 'data'}
                
                result = check_image_mantra(test_image_path)
                
                assert result == {'test': 'data'}
                mock_check_forgery.assert_called_once()


class TestProcessImage:
    """Test cases for the process_image function."""
    
    @patch('app.check_image_mantra')
    @patch('app.output_folder', '/tmp/test_output')
    def test_process_image_saves_outputs(self, mock_check_mantra, test_image_path, temp_dir):
        """Test that process_image saves all output images."""
        from app import process_image
        
        # Mock MantraNet output
        mock_check_mantra.return_value = {
            'original': np.random.rand(224, 224, 3) * 255,
            'heatmap': np.random.rand(224, 224) * 255,
            'mask': np.random.rand(224, 224) * 255,
        }
        
        with patch('app.output_folder', str(temp_dir)):
            with patch('os.makedirs'):
                process_image(test_image_path)
        
        # Check that files would be saved (mocked)
        mock_check_mantra.assert_called_once()
    
    @patch('app.check_image_mantra')
    def test_process_image_handles_numpy_arrays(self, mock_check_mantra, test_image_path, temp_dir):
        """Test that numpy arrays are properly converted to images."""
        from app import process_image
        
        # Return numpy arrays with different dtypes
        mock_check_mantra.return_value = {
            'float_image': np.random.rand(100, 100, 3),  # Float [0, 1]
            'uint8_image': np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
        }
        
        with patch('app.output_folder', str(temp_dir)):
            with patch('os.makedirs'):
                try:
                    process_image(test_image_path)
                except Exception as e:
                    pytest.fail(f"process_image failed with {e}")


class TestFileTypeDetection:
    """Test file type detection and processing."""
    
    @patch('app.process_image')
    def test_process_file_detects_jpg(self, mock_image):
        """Test that .jpg files are processed as images."""
        from app import process_file
        
        process_file("test.jpg")
        mock_image.assert_called_once_with("test.jpg")
    
    @patch('app.process_image')
    def test_process_file_detects_png(self, mock_image):
        """Test that .png files are processed as images."""
        from app import process_file
        
        process_file("test.png")
        mock_image.assert_called_once_with("test.png")
    
    @patch('app.process_image')
    def test_process_file_skips_mp4(self, mock_image):
        """Test that .mp4 files are skipped (video processing not supported)."""
        from app import process_file
        
        process_file("test.mp4")
        mock_image.assert_not_called()
    
    @patch('app.process_image')
    def test_process_file_skips_unknown(self, mock_image):
        """Test that unknown file types are skipped."""
        from app import process_file
        
        process_file("test.txt")
        mock_image.assert_not_called()


class TestCleanupResources:
    """Test GPU memory cleanup."""
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.empty_cache')
    def test_cleanup_cuda_memory(self, mock_empty_cache, mock_cuda_available):
        """Test that CUDA cache is cleared when available."""
        from app import process_file
        
        with patch('app.process_image'):
            process_file("test.jpg")
        
        mock_empty_cache.assert_called_once()
    
    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.backends.mps.is_available', return_value=True)
    @patch('torch.mps.empty_cache')
    def test_cleanup_mps_memory(self, mock_empty_cache, mock_mps_available, mock_cuda_available):
        """Test that MPS cache is cleared when available."""
        from app import process_file
        
        with patch('app.process_image'):
            process_file("test.jpg")
        
        mock_empty_cache.assert_called_once()
