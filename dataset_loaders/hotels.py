import pandas as pd

from .base import *
from tqdm import tqdm

class Hotels(BaseDataset):
    def __init__(self, root, mode, transform=None, valset=1):
        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        self.mode = mode
        self.root = root + '/hotels50k/'
        self.config_file = pd.read_csv(root + f'/hotels50k/v5_splits/val{valset}_small.csv')
        self.transform = transform
        print('getting classes')
        self.classes = np.unique(self.config_file.label)
        # if self.mode == 'train':
        #     self.classes = range(0, 100)
        # elif self.mode == 'eval':
        #     self.classes = range(100, 200)

        self.ys = list(self.config_file.label)
        self.I = [i for i in range(len(self.ys))]
        relative_im_paths = list(self.config_file.image)
        self.im_paths = [os.path.join(self.root, i) for i in relative_im_paths]