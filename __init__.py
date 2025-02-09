from .nodes import *

NODE_CLASS_MAPPINGS = {
    "Classification": ClassificationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Classification": "Image Classification"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

try:
    import cm_global
    cm_global.register_extension('Bliss Classification',
                               {'version': "1.0.0",
                                'name': 'Classification Pack',
                                'nodes': set(NODE_CLASS_MAPPINGS.keys()),
                                'description': 'Image Classification Nodes', })
except:
    pass
