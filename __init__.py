from .node import *
from .install import *


NODE_CLASS_MAPPINGS = {
    'ClassificationNode': ClassificationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClassificationNode": "Classification Node"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
