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
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": ("CKPT_NAME", {}),            # model_ckpt_10_400.pth.tar
                "input_image": ("IMAGE", {}),
                "output_image": ("IMAGE", {}),
            }
        }

    CATEGORY = "Classification"
    FUNCTION = "main"
    RETURN_TYPES = ["NUMBER"]
    RETURN_NAMES = ["prediction_score"]

    def __init__(self):
        # Load your model once when the node is created.
        # For example, if your model file is at 'models/segmentation_model.pt':
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def main(self, ckpt_name, input_image, output_image):
        model_path = get_local_filepath(classification_ckpts_dir_name, ckpt_name)

        args = argparse.Namespace(
            model_path=model_path,
            input_image_path=input_image,
            output_image_path=output_image,
            gpu=0
        )

        pred_score = classification(args)

        return (pred_score, )


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
