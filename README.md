# ComfyUI Classification Nodes

Custom nodes for ComfyUI that enable using custom classification models for image classification tasks.

## Features

- Load custom classification models from checkpoints
- Perform image classification with custom models
- Support for both CPU and CUDA execution
- Dynamic checkpoint selection from models directory

## Directory Structure

```
ComfyUI-Classification/
├── modules/               # Classification model implementation
├── __init__.py           # Node registration and display names
├── nodes.py              # Custom node implementation
├── pyproject.toml        # Project metadata for ComfyUI registry
└── requirements.txt      # Project dependencies
```

## Installation

1. Clone this repository into ComfyUI's custom_nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/avivbenyosef/comfyio_classification.git
```

2. Install the required dependencies:

```bash
cd comfyio_classification
pip install -r requirements.txt
```

For ComfyUI portable version:

```bash
python_embeded/python.exe -m pip install -r ComfyUI/custom_nodes/comfyio_classification/requirements.txt
```

## Usage

1. Place your classification model checkpoints in:

```
ComfyUI/models/classification/
```

2. In ComfyUI:
   - Find the "Image Classification" node in the node menu
   - Connect an input image
   - Select your model checkpoint from the dropdown
   - Connect an output image for visualization
   - The node will output a prediction score

## Model Requirements

Classification models should be saved as .pth.tar files and follow the expected format:

- Model checkpoint files should end with .pth.tar
- Default model name: model_ckpt_10_400.pth.tar

## License

This project is licensed under the MIT License - see the LICENSE file for details.
