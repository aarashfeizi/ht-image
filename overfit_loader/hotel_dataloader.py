import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.utils import save_image

from utils import get_shuffled_data, loadDataToMem

ANCHOR_IDX = 100
NEGATIVE_IDX = 101

class HotelTrain_Metric(Dataset):
    def __init__(self, args, transform=None, mode='f', save_pictures=False):
        super(HotelTrain_Metric, self).__init__()
        np.random.seed(args.seed)
        self.transform = transform
        self.save_pictures = save_pictures
        self.no_negative = args.no_negative
        self.mask = args.mask

        self.datas, self.num_classes, self.length, self.labels, _ = loadDataToMem(args.dataset_path, args.dataset_name,
                                                                                  mode=mode,
                                                                                  split_file_name=args.splits_file_name,
                                                                                  portion=args.portion,
                                                                                  split_path=args.splits_path,
                                                                                  dataset_folder=args.dataset_folder)

        self.shuffled_data = get_shuffled_data(datas=self.datas, seed=args.seed)
        # self.masks =

        print('hotel train classes: ', self.num_classes)
        print('hotel train length: ', self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        #anch_idx = random.randint(0, self.num_classes - 1)
        anch_class = self.labels[ANCHOR_IDX]
        #anch = Image.open(random.choice(self.datas[anch_class]))
        anch = Image.open(self.datas[anch_class][0])

        # get pos image from same class
        #pos = Image.open(random.choice(self.datas[anch_class]))
        pos = Image.open(self.datas[anch_class][1])

        # get neg image from different class
        negs = []
        for i in range(self.no_negative):
            #neg_idx = random.randint(0, self.num_classes - 1)
            neg_class = self.labels[NEGATIVE_IDX]

            while anch_class == neg_class:
                neg_idx = random.randint(0, self.num_classes - 1)
                neg_class = self.labels[neg_idx]

                # class1 = self.labels[idx1]

                # image1 = Image.open(random.choice(self.datas[self.class1]))
            #neg = Image.open(random.choice(self.datas[neg_class]))
            neg = Image.open(self.datas[neg_class][i])
            neg = neg.convert('RGB')
            negs.append(neg)

        anch = anch.convert('RGB')
        pos = pos.convert('RGB')
        save = False
        if self.transform:
            if self.save_pictures and random.random() < 0.0001:
                save = True
                img1_random = random.randint(0, 1000)
                img2_random = random.randint(0, 1000)
                anch.save(f'hotel_imagesamples/train/train_{anch_class}_{img1_random}_before.png')
                negs[0].save(f'hotel_imagesamples/train/train_{neg_class}_{img2_random}_before.png')

            anch = self.transform(anch)
            pos = self.transform(pos)
            for i, neg in enumerate(negs):
                negs[i] = self.transform(neg)

            neg = torch.stack(negs)

            if save:
                save_image(anch, f'hotel_imagesamples/train/train_{anch_class}_{img1_random}_after.png')
                save_image(negs[0], f'hotel_imagesamples/train/train_{neg_class}_{img2_random}_after.png')

        return anch, pos, neg


class HotelTrain_FewShot(Dataset):
    def __init__(self, args, transform=None, mode='train', save_pictures=False):
        super(HotelTrain_FewShot, self).__init__()
        np.random.seed(args.seed)
        self.transform = transform
        self.save_pictures = save_pictures
        self.class1 = 0
        self.image1 = None
        self.no_negative = args.no_negative

        self.datas, self.num_classes, self.length, self.labels, _ = loadDataToMem(args.dataset_path, args.dataset_name,
                                                                                  mode=mode,
                                                                                  split_file_name=args.splits_file_name,
                                                                                  portion=args.portion,
                                                                                  split_path=args.splits_path,
                                                                                  dataset_folder=args.dataset_folder)

        self.shuffled_data = get_shuffled_data(datas=self.datas, seed=args.seed)

        print('hotel train classes: ', self.num_classes)
        print('hotel train length: ', self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # image1 = random.choice(self.dataset.imgs)
        label = None  # label is the distance between the two image. 0: same, 1: different
        img1 = None
        img2 = None
        # get image from same class
        if index % (self.no_negative + 1) == 0:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            self.class1 = self.labels[ANCHOR_IDX]
            class2 = self.class1
            self.image1 = Image.open(self.datas[self.class1][0])
            image2 = Image.open(self.datas[class2][1])
        # get image from different class
        else:
            label = 1.0
            # idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            class2 = self.labels[NEGATIVE_IDX]

            while self.class1 == class2:
                idx2 = random.randint(0, self.num_classes - 1)
                class2 = self.labels[idx2]

            # class1 = self.labels[idx1]

            # image1 = Image.open(random.choice(self.datas[self.class1]))
            # image2 = Image.open(random.choice(self.datas[class2]))
            image2 = Image.open(self.datas[class2][index % (self.no_negative + 1) - 1])


        image1 = self.image1.convert('RGB')
        image2 = image2.convert('RGB')
        save = False
        if self.transform:
            if self.save_pictures and random.random() < 0.0001:
                save = True
                img1_random = random.randint(0, 1000)
                img2_random = random.randint(0, 1000)
                image1.save(f'hotel_imagesamples/train/train_{self.class1}_{img1_random}_before.png')
                image2.save(f'hotel_imagesamples/train/train_{class2}_{img2_random}_before.png')

            image2 = self.transform(image2)
            image1 = self.transform(image1)

            if save:
                save_image(image1, f'hotel_imagesamples/train/train_{self.class1}_{img1_random}_after.png')
                save_image(image2, f'hotel_imagesamples/train/train_{class2}_{img2_random}_after.png')

        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))

    def _get_single_item(self, index):
        label, image_path = self.shuffled_data[index]

        image = Image.open(image_path)

        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.from_numpy(np.array(label, dtype=np.float32))

    def get_k_samples(self, k=100):
        ks = np.random.randint(len(self.shuffled_data), size=k)
        imgs = []
        lbls = []
        for i in ks:
            img, lbl = self._get_single_item(i)
            imgs.append(img)
            lbls.append(lbl)

        return imgs, lbls


class HotelTest_FewShot(Dataset):

    def __init__(self, args, transform=None, mode='test_seen', save_pictures=False):
        np.random.seed(args.seed)
        super(HotelTest_FewShot, self).__init__()
        self.transform = transform
        self.save_pictures = save_pictures
        self.times = args.times
        self.way = args.way
        self.img1 = None
        self.c1 = None

        self.datas, self.num_classes, _, self.labels, self.datas_bg = loadDataToMem(args.dataset_path,
                                                                                    args.dataset_name,
                                                                                    mode=mode,
                                                                                    split_file_name=args.splits_file_name,
                                                                                    portion=args.portion,
                                                                                    split_path=args.splits_path,
                                                                                    dataset_folder=args.dataset_folder)

        print(f'hotel {mode} classes: ', self.num_classes)
        print(f'hotel {mode} length: ', self.__len__())

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        # generate image pair from same class
        if idx == 0:
            self.c1 = self.labels[ANCHOR_IDX]
            c2 = self.c1
            self.img1 = Image.open(self.datas[self.c1][0]).convert('RGB')
            img2 = Image.open(self.datas[c2][1]).convert('RGB')
        # generate image pair from different class
        else:
            c2 = list(self.datas_bg.keys())[NEGATIVE_IDX]
            while self.c1 == c2:
                c2 = list(self.datas_bg.keys())[random.randint(0, len(self.datas_bg.keys()) - 1)]
            img2 = Image.open(self.datas_bg[c2][0][0]).convert('RGB')

        save = False
        if self.transform:
            if self.save_pictures and random.random() < 0.001:
                save = True
                img1_random = random.randint(0, 1000)
                img2_random = random.randint(0, 1000)
                self.img1.save(f'hotel_imagesamples/val/val_{self.c1}_{img1_random}_before.png')
                img2.save(f'hotel_imagesamples/val/val_{c2}_{img2_random}_before.png')

            img1 = self.transform(self.img1)
            img2 = self.transform(img2)

            if save:
                save_image(img1, f'hotel_imagesamples/val/val_{self.c1}_{img1_random}_after.png')
                save_image(img2, f'hotel_imagesamples/val/val_{c2}_{img2_random}_after.png')

        return img1, img2


class Hotel_DB(Dataset):
    def __init__(self, args, transform=None, mode='test'):
        np.random.seed(args.seed)
        super(Hotel_DB, self).__init__()
        self.transform = transform

        total = True
        if mode == 'val' or mode == 'test':  # mode == *_seen or *_unseen or train
            mode_tmp = mode + '_seen'
            total = True
        else:
            mode_tmp = mode
            total = False

        self.datas, self.num_classes, _, self.labels, self.all_data = loadDataToMem(args.dataset_path,
                                                                                    args.dataset_name,
                                                                                    mode=mode_tmp,
                                                                                    split_file_name=args.splits_file_name,
                                                                                    portion=args.portion,
                                                                                    split_path=args.splits_path,
                                                                                    dataset_folder=args.dataset_folder,
                                                                                    return_bg=(mode != 'train'))
        self.all_shuffled_data = get_shuffled_data(self.all_data,
                                                   seed=args.seed,
                                                   one_hot=False,
                                                   both_seen_unseen=True,
                                                   shuffle=False)
        # else: # todo
        #     self.all_shuffled_data = get_shuffled_data(self.datas, seed=args.seed, one_hot=False)

        print(f'hotel {mode} classes: ', self.num_classes)
        print(f'hotel {mode} length: ', self.__len__())

    def __len__(self):
        return len(self.all_shuffled_data)

    def __getitem__(self, index):
        lbl = self.all_shuffled_data[index][0]
        img = Image.open(self.all_shuffled_data[index][1]).convert('RGB')
        bl = self.all_shuffled_data[index][2]

        path = self.all_shuffled_data[index][1].split('/')

        id = path[-4]
        id += '-' + path[-3]
        id += '-' + path[-1].split('.')[0]

        if self.transform:
            img = self.transform(img)

        return img, lbl, bl, id
