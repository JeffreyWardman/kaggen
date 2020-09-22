import importlib
from typing import Any, Dict


def augment(config: Dict[str, Any]) -> Any:
    """Generates augmentations from configuration dictionary.

    Arguments:
        config {Dict[str, Any]} -- Configuration dictionary

    Returns:
        Any -- augmentations
    """
    module = importlib.import_module(config['module_name'])
    augs = []
    if 'types' in config:
        if config['types']:
            for aug in config['types']:
                parameters = config['types'][aug]['parameters']
                if parameters:
                    augs.append(getattr(module, aug)(**parameters))
                else:
                    augs.append(getattr(module, aug)())

    transforms = module.Compose(augs)
    # TODO image only vs image and mask for segmentation tasks
    return transforms
