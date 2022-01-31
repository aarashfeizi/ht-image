from . import utils
from .SOP import SOP
from .base import BaseDataset
from .cars import Cars
from .cub import CUBirds
from .hotels import Hotels

_type = {
    'cars': Cars,
    'cub': CUBirds,
    'SOP': SOP,
    'hotels': Hotels,
    'hotels_small': Hotels,

}


def load(name, root, transform=None, valset=1, small=False):
    if 'hotels' not in name:
        return _type[name](root=root, mode='eval', transform=transform)
    else:
        return _type[name](root=root, mode='eval', transform=transform, valset=valset, small=small)
