# MantraNet Model Weights

This directory should contain the pre-trained MantraNet model weights.

## Required File

- `MantraNetv4.pt` - Pre-trained MantraNet model weights

## Getting the Model Weights

The MantraNet model weights are not included in this repository due to their size.

### Option 1: Train Your Own Model
Follow the MantraNet training instructions from the original paper:
- Paper: "MantraNet: Manipulation Tracing Network For Detection And Localization of Image ForgeriesWith Anomalous Features"
- GitHub: https://github.com/ISICV/MantraNet

### Option 2: Use Pre-trained Weights
If you have access to pre-trained MantraNet weights:
1. Place the `.pt` file in this directory
2. Name it `MantraNetv4.pt` or update the path in `app.py`

### For Testing Without Real Weights

The application will run with randomly initialized weights if no model file is found, but it will only produce random outputs. This is useful for:
- Testing the API infrastructure
- Development and integration testing
- CI/CD pipeline validation

**Note:** For actual deepfake detection, you MUST use properly trained model weights.
