import os
import sys
import logging
import argparse
import torch
from typing import Tuple

import folder_paths
from .modules import classification

# Basic practice to get paths from ComfyUI
custom_nodes_script_dir = os.path.dirname(os.path.abspath(__file__))
custom_nodes_model_dir = os.path.join(folder_paths.models_dir, "classification")
custom_nodes_output_dir = os.path.join(folder_paths.get_output_directory(), "classification")

logger = logging.getLogger('comfyui_classification')

def get_local_filepath(dirname: str, local_file_name: str) -> str:
    """Get the full path for a model file in the models directory."""
    folder = os.path.join(folder_paths.models_dir, dirname)
    if not os.path.exists(folder):
        os.makedirs(folder)

    destination = os.path.join(folder, local_file_name)
    if not os.path.exists(destination):
        raise FileNotFoundError(f'{destination} not found')
    return destination

class ClassificationNode:
    """
    A ComfyUI node for image classification using custom models.

    This node loads a classification model from a checkpoint and performs inference
    on input images, returning a prediction score.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define input parameters for the classification node."""
        try:
            # Get list of available checkpoints
            folder = os.path.join(folder_paths.models_dir, "classification")
            if not os.path.exists(folder):
                os.makedirs(folder)
            ckpt_files = [f for f in os.listdir(folder) if f.endswith('.pth.tar')]
            if not ckpt_files:
                ckpt_files = ["model_ckpt_10_400.pth.tar"]  # Default if no checkpoints found
        except Exception:
            ckpt_files = ["model_ckpt_10_400.pth.tar"]  # Fallback default
            
        return {
            "required": {
                "ckpt_name": (ckpt_files,),
                "input_image": ("IMAGE",),
                "output_image": ("IMAGE",),
            }
        }

    CATEGORY = "Classification"
    FUNCTION = "classify"
    RETURN_TYPES = ("NUMBER",)
    RETURN_NAMES = ("prediction_score",)

    def __init__(self):
        """Initialize the classification node with CUDA support if available."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None  # Model will be loaded on first use

    def classify(self, ckpt_name: str, input_image: torch.Tensor, output_image: torch.Tensor) -> Tuple[float]:
        """
        Perform classification on the input image.

        Args:
            ckpt_name: Name of the model checkpoint file
            input_image: Input image to classify
            output_image: Output image for visualization

        Returns:
            Tuple containing the prediction score as a float
        """
        try:
            model_path = get_local_filepath("classification", ckpt_name)
            
            args = argparse.Namespace(
                model_path=model_path,
                input_image_path=input_image,
                output_image_path=output_image,
                gpu=0 if torch.cuda.is_available() else -1
            )

            pred_score = classification(args)
            return (float(pred_score),)
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model checkpoint not found: {e}")
        except Exception as e:
            raise RuntimeError(f"Classification failed: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, ckpt_name: str, input_image: torch.Tensor, output_image: torch.Tensor) -> str:
        """Control node re-execution based on checkpoint changes."""
        return ckpt_name
