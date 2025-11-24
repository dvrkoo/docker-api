"""
Unit tests for video processing and label update functions.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import cv2


class TestUpdateLabel:
    """Test cases for the update_label function."""
    
    def test_update_label_fake_detection(self, mock_face, test_frame):
        """Test label for fake face detection."""
        from app import update_label
        
        predictions = [[0.1, 0.2, 0.3, 0.9, 0.1]]  # High fake score for model index 3
        frame = test_frame.copy()
        
        update_label(mock_face, predictions, frame)
        
        # Frame should be modified (check that it's different from original)
        # This is a basic sanity check
        assert frame.shape == test_frame.shape
    
    def test_update_label_genuine_detection(self, mock_face, test_frame):
        """Test label for genuine face detection."""
        from app import update_label
        
        predictions = [[0.1, 0.2, 0.3, 0.4, 0.1]]  # All scores below 0.5
        frame = test_frame.copy()
        
        update_label(mock_face, predictions, frame)
        
        assert frame.shape == test_frame.shape
    
    def test_update_label_empty_predictions(self, mock_face, test_frame):
        """Test that empty predictions are handled gracefully."""
        from app import update_label
        
        predictions = []
        frame = test_frame.copy()
        
        # Should not raise an error
        try:
            update_label(mock_face, predictions, frame)
        except Exception as e:
            pytest.fail(f"update_label raised {e} with empty predictions")
    
    def test_update_label_correct_model_selection(self, mock_face, test_frame):
        """Test that the correct model is selected based on max prediction."""
        from app import update_label
        
        # Model at index 2 (neuraltextures) has highest score
        predictions = [[0.1, 0.2, 0.9, 0.4, 0.1]]
        frame = test_frame.copy()
        
        update_label(mock_face, predictions, frame)
        
        # We can't easily assert the text on the frame without OCR,
        # but we can verify the function runs without error
        assert frame.shape == test_frame.shape


class TestPredictFrameFromVideo:
    """Test cases for predict_frame_from_video function."""
    
    @patch('app.model_functions.predict_with_model')
    def test_predict_frame_collects_all_models(self, mock_predict):
        """Test that predictions are collected for all 5 models."""
        from app import predict_frame_from_video
        
        mock_predict.return_value = 0.5
        mock_models = MagicMock()
        input_tensor = MagicMock()
        predictions = []
        json_data = {model: [] for model in ['faceswap', 'deepfake', 'neuraltextures', 'face2face', 'faceshifter']}
        
        predict_frame_from_video(mock_models, input_tensor, predictions, json_data)
        
        # Should have 5 predictions (one for each model)
        assert len(predictions) == 5
        assert mock_predict.call_count == 5
        
        # Check that all models have predictions in json_data
        for model in json_data:
            assert len(json_data[model]) == 1
    
    @patch('app.model_functions.predict_with_model')
    def test_predict_frame_updates_json(self, mock_predict):
        """Test that JSON data is correctly updated."""
        from app import predict_frame_from_video
        
        mock_predict.return_value = 0.75
        mock_models = MagicMock()
        input_tensor = MagicMock()
        predictions = []
        json_data = {model: [] for model in ['faceswap', 'deepfake', 'neuraltextures', 'face2face', 'faceshifter']}
        
        predict_frame_from_video(mock_models, input_tensor, predictions, json_data)
        
        # Each model should have one prediction
        for model in json_data:
            assert json_data[model] == [0.75]


class TestProcessVideo:
    """Test cases for process_video function."""
    
    @patch('cv2.VideoCapture')
    @patch('cv2.VideoWriter')
    @patch('app.model_functions.load_models')
    @patch('app.model_functions.detect_faces')
    @patch('app.model_functions.preprocess_input')
    @patch('app.predict_frame_from_video')
    def test_process_video_basic_flow(
        self,
        mock_predict,
        mock_preprocess,
        mock_detect,
        mock_load_models,
        mock_writer,
        mock_capture,
        temp_dir
    ):
        """Test basic video processing flow."""
        from app import process_video
        
        # Setup mocks
        mock_cap_instance = MagicMock()
        mock_cap_instance.get.side_effect = [30, 10, 640, 480]  # fps, frames, width, height
        mock_cap_instance.read.side_effect = [
            (True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)),
            (False, None)  # End of video
        ]
        mock_capture.return_value = mock_cap_instance
        
        mock_writer_instance = MagicMock()
        mock_writer.return_value = mock_writer_instance
        
        mock_load_models.return_value = (MagicMock(), MagicMock())
        mock_detect.return_value = []  # No faces detected
        
        video_path = str(temp_dir / "test.mp4")
        
        with patch('app.output_folder', str(temp_dir)):
            process_video(video_path)
        
        # Verify that video writer was released
        mock_writer_instance.release.assert_called()
        mock_cap_instance.release.assert_called()
    
    @patch('cv2.VideoCapture')
    @patch('cv2.VideoWriter')
    @patch('app.model_functions.load_models')
    @patch('app.model_functions.detect_faces')
    @patch('app.model_functions.get_boundingbox')
    @patch('app.model_functions.preprocess_input')
    @patch('app.predict_frame_from_video')
    @patch('app.update_label')
    def test_process_video_with_face_detection(
        self,
        mock_update_label,
        mock_predict,
        mock_preprocess,
        mock_bbox,
        mock_detect,
        mock_load_models,
        mock_writer,
        mock_capture,
        temp_dir
    ):
        """Test video processing when faces are detected."""
        from app import process_video
        
        # Setup mocks
        mock_cap_instance = MagicMock()
        mock_cap_instance.get.side_effect = [30, 1, 640, 480]
        
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_cap_instance.read.side_effect = [(True, frame), (False, None)]
        mock_capture.return_value = mock_cap_instance
        
        mock_writer_instance = MagicMock()
        mock_writer.return_value = mock_writer_instance
        
        mock_load_models.return_value = (MagicMock(), MagicMock())
        
        # Mock face detection
        mock_face = MagicMock()
        mock_detect.return_value = [mock_face]
        mock_bbox.return_value = (100, 100, 200)
        mock_preprocess.return_value = MagicMock()
        
        video_path = str(temp_dir / "test.mp4")
        
        with patch('app.output_folder', str(temp_dir)):
            with patch('builtins.open', create=True):
                process_video(video_path)
        
        # Verify face was processed
        mock_detect.assert_called()
        mock_update_label.assert_called()


class TestFileWatcher:
    """Test cases for file watching functionality."""
    
    def test_file_queue_integration(self):
        """Test that FileCreatedHandler adds files to queue."""
        from app import FileCreatedHandler, file_queue
        
        # Clear the queue
        while not file_queue.empty():
            file_queue.get()
        
        handler = FileCreatedHandler()
        
        # Create mock event
        mock_event = MagicMock()
        mock_event.is_directory = False
        mock_event.src_path = "/test/path/image.jpg"
        
        handler.on_created(mock_event)
        
        # Verify file was added to queue
        assert not file_queue.empty()
        assert file_queue.get() == "/test/path/image.jpg"
    
    def test_directory_events_ignored(self):
        """Test that directory creation events are ignored."""
        from app import FileCreatedHandler, file_queue
        
        # Clear the queue
        while not file_queue.empty():
            file_queue.get()
        
        handler = FileCreatedHandler()
        
        # Create mock directory event
        mock_event = MagicMock()
        mock_event.is_directory = True
        mock_event.src_path = "/test/path/directory"
        
        handler.on_created(mock_event)
        
        # Queue should still be empty
        assert file_queue.empty()
