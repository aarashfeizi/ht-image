from .cars import Cars
from .cub import CUBirds
from .SOP import SOP
from .hotels import Hotels
from .import utils
from .base import BaseDataset


_type = {
    'cars': Cars,
    'cub': CUBirds,
    'SOP': SOP,
    'hotels': Hotels
}

def load(name, root, transform = None, valset=1):
    if name != 'hotels':
        return _type[name](root = root, mode = 'eval', transform = transform)
    else:
        return _type[name](root = root, mode = 'eval', transform = transform, valset = valset)

