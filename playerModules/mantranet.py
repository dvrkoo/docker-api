"""
MantraNet implementation for deepfake detection.
This module provides the MantraNet model for detecting image manipulations.
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class MantraNet(nn.Module):
    """
    MantraNet: Manipulation Tracing Network for detecting image forgeries.
    
    This is a simplified implementation that provides the expected interface.
    For production use, you should implement or import the full MantraNet architecture.
    """
    
    def __init__(self):
        super(MantraNet, self).__init__()
        
        # Simple convolutional layers as placeholder
        # In production, this should be the full MantraNet architecture
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """Forward pass through the network."""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x


def pre_trained_model(weight_path, device='cpu'):
    """
    Load a pre-trained MantraNet model.
    
    Args:
        weight_path (str): Path to the model weights file
        device (str or torch.device): Device to load the model on
        
    Returns:
        MantraNet: The loaded model
    """
    model = MantraNet()
    
    # Try to load weights if they exist
    try:
        if torch.cuda.is_available() or device != 'cpu':
            model.load_state_dict(torch.load(weight_path, map_location=device))
            print(f"Loaded model weights from {weight_path}")
        else:
            model.load_state_dict(torch.load(weight_path, map_location='cpu'))
            print(f"Loaded model weights from {weight_path}")
    except FileNotFoundError:
        print(f"Warning: Model weights not found at {weight_path}")
        print("Using randomly initialized model (FOR TESTING ONLY)")
        print("To use in production, download MantraNet weights to trained_models/MantraNetv4.pt")
    except Exception as e:
        print(f"Warning: Could not load model weights: {e}")
        print("Using randomly initialized model (FOR TESTING ONLY)")
    
    model = model.to(device)
    model.eval()
    return model


def check_forgery(model, img, device='cpu'):
    """
    Check if an image contains forgeries using MantraNet.
    
    Args:
        model: The MantraNet model
        img (PIL.Image): Input image to analyze
        device (str or torch.device): Device to run inference on
        
    Returns:
        dict: Dictionary containing detection results and visualizations
            - 'original': Original image as numpy array
            - 'heatmap': Manipulation probability heatmap
            - 'mask': Binary manipulation mask
            - 'overlay': Original image with heatmap overlay
    """
    # Convert PIL Image to tensor
    img_array = np.array(img)
    
    # Normalize and prepare for model input
    img_tensor = torch.from_numpy(img_array).float() / 255.0
    
    # Ensure correct shape (B, C, H, W)
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
    
    img_tensor = img_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(img_tensor)
    
    # Convert output to numpy
    heatmap = output.squeeze().cpu().numpy()
    
    # Create binary mask (threshold at 0.5)
    mask = (heatmap > 0.5).astype(np.uint8) * 255
    
    # Normalize heatmap to 0-255 range for visualization
    heatmap_vis = (heatmap * 255).astype(np.uint8)
    
    # Create overlay by blending original with heatmap
    # Convert heatmap to RGB (red for manipulated areas)
    heatmap_rgb = np.zeros((*heatmap.shape, 3), dtype=np.uint8)
    heatmap_rgb[:, :, 0] = heatmap_vis  # Red channel
    
    # Resize heatmap to match original image size
    if heatmap_rgb.shape[:2] != img_array.shape[:2]:
        from PIL import Image as PILImage
        heatmap_pil = PILImage.fromarray(heatmap_rgb)
        heatmap_pil = heatmap_pil.resize((img_array.shape[1], img_array.shape[0]), PILImage.BILINEAR)
        heatmap_rgb = np.array(heatmap_pil)
    
    # Blend original and heatmap (70% original, 30% heatmap)
    overlay = (img_array * 0.7 + heatmap_rgb * 0.3).astype(np.uint8)
    
    # Return results as dictionary
    results = {
        'original': img_array,
        'heatmap': heatmap_vis,
        'mask': mask,
        'overlay': overlay,
    }
    
    return results
