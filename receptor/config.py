import logging
import random
import sys

import numpy as np
import torch

version = '0.0.1'
__RANDOM_SEED__ = None


def set_random_seed(seed):
    if not isinstance(seed, int):
        raise ValueError('Random seed must be an integer value')
    global __RANDOM_SEED__
    __RANDOM_SEED__ = seed
    random.seed(seed)
    np.random.seed(seed)
    from gym import spaces
    spaces.prng.seed(seed)
    torch.manual_seed(seed)


def get_random_seed():
    global __RANDOM_SEED__
    return __RANDOM_SEED__


_handler = logging.StreamHandler(sys.stdout)
_formatter = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s', '%d.%m.%y %H:%M:%S')
_handler.setFormatter(_formatter)
logger = logging.getLogger('.'.join(__name__.split('.')))
logger.propagate = False


def logger_setup():
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)


def undo_logger_setup():
    logger.removeHandler(_handler)
    logger.setLevel(logging.NOTSET)
