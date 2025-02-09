import os
from typing import Tuple
import argparse
import torch
import folder_paths
from modules.classification.inference import main as classification_inference

# Get paths from ComfyUI
models_path = os.path.join(folder_paths.models_dir, "classification")
if not os.path.exists(models_path):
    os.makedirs(models_path)

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
            ckpt_files = [f for f in os.listdir(models_path) if f.endswith('.pth.tar')]
            if not ckpt_files:
                ckpt_files = ["model_ckpt_10_400.pth.tar"]
        except Exception:
            ckpt_files = ["model_ckpt_10_400.pth.tar"]
            
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
        """Initialize the classification node."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def classify(self, ckpt_name: str, input_image: torch.Tensor, output_image: torch.Tensor) -> Tuple[float]:
        """
        Perform classification on the input image.

        Args:
            ckpt_name: Name of the model checkpoint file
            input_image: Input image tensor
            output_image: Output image tensor for visualization

        Returns:
            Tuple containing the prediction score as a float
        """
        try:
            model_path = os.path.join(models_path, ckpt_name)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
            
            args = argparse.Namespace(
                model_path=model_path,
                input_image_path=input_image,
                output_image_path=output_image,
                gpu=0 if str(self.device) == 'cuda' else -1
            )

            pred_score = classification_inference(args)
            return (float(pred_score),)
            
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e))
        except Exception as e:
            raise RuntimeError(f"Classification failed: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, ckpt_name: str, input_image: torch.Tensor, output_image: torch.Tensor) -> str:
        """Control node re-execution based on checkpoint changes."""
        return ckpt_name
