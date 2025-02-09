from .node import *
from .install import *

comfy_path = os.path.dirname(folder_paths.__file__)
modules_path = os.path.join(os.path.dirname(__file__), "modules")

sys.path.append(modules_path)
NODE_CLASS_MAPPINGS = {
    'ClassificationNode': ClassificationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClassificationNode": "Classification Node"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


try:
    import cm_global
    cm_global.register_extension('Bliss Classification',
                                 {'version': "1",
                                  'name': 'Bliss Pack',
                                  'nodes': set(NODE_CLASS_MAPPINGS.keys()),
                                  'description': 'Bliss Comfy Nodes', })
except:
    pass