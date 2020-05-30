from util.logging import logger
import functools

changed_list = []


def changed(func, name):
    """Mark a changed function [DEBUG]
    """
    @functools.wraps(func)
    def decorateit(*args, **kwargs):
        if func.__name__ not in changed_list:
            changed_list.append(func.__name__)
            logger.debug("function {} changed by {}! Check if it works as you think".format(name))
        return func(*args, **kwargs)
    return decorateit
