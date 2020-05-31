import time

import torch

from util.logging import logger


def save(obj, filename):
    """Useful wrapping of pickle method
    Args:
        obj: Any
        filename: str, obj will be saved to `current_path/saves/time/filename/`
    Returns:
        path: str, full path of the file saved
    """
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    path = 'saves/' + rq + '/' + filename
    with open(path, 'wb+') as f:
        torch.save(obj, f)
    logger.info("a {} object saved to {}".format(obj.__class__.__name__, path))
    return path


def load(path):
    """Useful wrapping of pickle method
    Args:
        path: full or relative path of the file to be loaded
    Returns:
        obj: any
    """
    with open(path, 'rb+') as f:
        obj = torch.load(f)
    logger.info("a {} object loaded from {}".format(obj.__class__.__name__,
                                                    path))
    return obj
