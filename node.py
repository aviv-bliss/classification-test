import os
import sys
import logging
import argparse
import torch

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

import folder_paths
from classification.inference import main as classification

logger = logging.getLogger('comfyui_classification')

classification_ckpts_dir_name = "classification"  # in /home/bliss/comfy/ComfyUI/models


def get_local_filepath(dirname, local_file_name):
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

    Class methods
    -------------
    INPUT_TYPES (dict):
        Defines the input parameters for the classification node.
    IS_CHANGED:
        Controls when the node should be re-executed based on checkpoint changes.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The types of the output values (NUMBER for prediction score)
    RETURN_NAMES (`tuple`):
        Names of the output values
    FUNCTION (`str`):
        The name of the entry-point method
    CATEGORY (`str`):
        UI category for the node
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": ("STRING", {
                    "default": "model_ckpt_10_400.pth.tar",
                    "multiline": False,
                }),
                "input_image": ("IMAGE", {}),
                "output_image": ("IMAGE", {}),
            }
        }

    CATEGORY = "Classification"
    FUNCTION = "main"
    RETURN_TYPES = ["NUMBER"]
    RETURN_NAMES = ["prediction_score"]

    def __init__(self):
        """Initialize the classification node with CUDA support if available."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def main(self, ckpt_name, input_image, output_image):
        """
        Perform classification on the input image.

        Parameters
        ----------
        ckpt_name : str
            Name of the model checkpoint file
        input_image : torch.Tensor
            Input image to classify
        output_image : torch.Tensor
            Output image for visualization

        Returns
        -------
        tuple
            Contains the prediction score as a float
        """
        try:
            model_path = get_local_filepath(classification_ckpts_dir_name, ckpt_name)

            args = argparse.Namespace(
                model_path=model_path,
                input_image_path=input_image,
                output_image_path=output_image,
                gpu=0 if torch.cuda.is_available() else -1
            )

            pred_score = classification(args)
            return (float(pred_score), )
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model checkpoint not found: {e}")
        except Exception as e:
            raise RuntimeError(f"Classification failed: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, ckpt_name, input_image, output_image):
        """
        Control node re-execution based on checkpoint changes.
        
        Returns the checkpoint name as a string to force re-execution when the
        checkpoint file changes.
        """
        return ckpt_name


# Set the web directory to the current directory for any frontend extensions
WEB_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Add custom API routes, using router
from aiohttp import web
from server import PromptServer

@PromptServer.instance.routes.get("/hello")
async def get_hello(request):
    return web.json_response("hello")

NODE_CLASS_MAPPINGS = {
    'ClassificationNode': ClassificationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClassificationNode": "Classification Node"
}
