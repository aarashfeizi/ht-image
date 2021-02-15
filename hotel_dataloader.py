import os
import random
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.utils import save_image

import utils
from utils import get_shuffled_data, loadDataToMem, get_overfit, get_masks


class HotelTrain_Metric(Dataset):
    def __init__(self, args, transform=None, mode='f', save_pictures=False, overfit=False, return_paths=False,
                 batchhard=[False, 0, 0]):
        super(HotelTrain_Metric, self).__init__()
        self.fourth_dim = args.fourth_dim
        np.random.seed(args.seed)
        self.transform = transform
        self.save_pictures = save_pictures
        self.no_negative = args.no_negative
        self.aug_mask = args.aug_mask
        self.return_paths = return_paths
        self.normalize = utils.TransformLoader(-1).transform_normalize
        self.colored_mask = args.colored_mask
        self.batchhard = batchhard[0]
        self.bh_P = batchhard[1]
        self.bh_K = batchhard[2]

        start = time.time()
        self.datas, self.num_classes, self.length, self.labels, _ = loadDataToMem(args.dataset_path, args.dataset_name,
                                                                                  mode=mode,
                                                                                  split_file_path=args.splits_file_path,
                                                                                  portion=args.portion,
                                                                                  dataset_folder=args.dataset_folder)
        end = time.time()
        if utils.MY_DEC.enabled:
            print(f'HotelTrain_Metric loadDataToMem time: {end - start}')

        if overfit and args.overfit_num > 0:
            self.overfit = True
            self.overfit_samples = get_overfit(data=self.datas, labels=self.labels, anchors=args.overfit_num,
                                               neg_per_pos=self.no_negative)
            print(f'Overfitting to {args.overfit_num} triplet[s]: {self.overfit_samples}')
        else:
            self.overfit = False

        self.shuffled_data = get_shuffled_data(datas=self.datas, seed=args.seed)

        if self.aug_mask:
            self.masks = get_masks(args.dataset_path, args.dataset_folder,
                                   os.path.join(args.project_path, args.mask_path))

        else:
            self.masks = []

        print('HotelTrain_Metric hotel train classes: ', self.num_classes)
        print('HotelTrain_Metric hotel train length: ', self.length)

    def __len__(self):
        return self.length

    def __triplet_getitem__(self, index):

        paths = []
        start = time.time()
        if self.overfit:
            overfit_triplet = np.random.choice(self.overfit_samples, 1)[0]
            paths.append(overfit_triplet['anch'])
            anch = Image.open(overfit_triplet['anch'])
            anch = anch.convert('RGB')

            paths.append(overfit_triplet['pos'])
            pos = Image.open(overfit_triplet['pos'])

            negs = []
            for neg_path in overfit_triplet['neg']:
                paths.append(neg_path)
                neg = Image.open(neg_path)
                neg = neg.convert('RGB')

                negs.append(neg)

        else:  # not overfitting
            anch_idx = random.randint(0, self.num_classes - 1)
            anch_class = self.labels[anch_idx]
            random_path = random.choice(self.datas[anch_class])
            paths.append(random_path)
            anch = Image.open(random_path)

            # get pos image from same class

            random_path = random.choice(self.datas[anch_class])
            paths.append(random_path)
            pos = Image.open(random_path)

            # get neg image from different class
            negs = []
            for i in range(self.no_negative):
                neg_idx = random.randint(0, self.num_classes - 1)
                neg_class = self.labels[neg_idx]

                while anch_class == neg_class:
                    neg_idx = random.randint(0, self.num_classes - 1)
                    neg_class = self.labels[neg_idx]

                    # class1 = self.labels[idx1]

                    # image1 = Image.open(random.choice(self.datas[self.class1]))

                random_path = random.choice(self.datas[neg_class])
                paths.append(random_path)
                neg = Image.open(random_path)

                neg = neg.convert('RGB')
                negs.append(neg)

            anch = anch.convert('RGB')
            pos = pos.convert('RGB')

        save = False
        end = time.time()
        if utils.MY_DEC.enabled:
            print(f'HotelTrain_Metric Dataloader, choose images time: {end - start}')
        if self.transform:
            start = time.time()
            if self.save_pictures and random.random() < 0.0001:
                save = True
                img1_random = random.randint(0, 1000)
                img2_random = random.randint(0, 1000)
                anch.save(f'hotel_imagesamples/train/train_{anch_class}_{img1_random}_before.png')
                negs[0].save(f'hotel_imagesamples/train/train_{neg_class}_{img2_random}_before.png')

            if self.aug_mask:
                anch_mask = Image.open(self.masks[np.random.randint(len(self.masks))])

                pos_mask = Image.open(self.masks[np.random.randint(len(self.masks))])

                anch, masked_anch, anch_mask, _ = utils.add_mask(anch, anch_mask, colored=self.colored_mask)
                pos, masked_pos, pos_mask, _ = utils.add_mask(pos, pos_mask, colored=self.colored_mask)

                if not self.fourth_dim:
                    anch = masked_anch
                    pos = masked_pos

                masked_negs = []
                neg_masks = []

                for neg in negs:
                    neg_mask = Image.open(self.masks[np.random.randint(len(self.masks))])
                    neg, masked_neg, neg_mask, _ = utils.add_mask(neg, neg_mask, colored=self.colored_mask)

                    if not self.fourth_dim:
                        neg = masked_neg

                    masked_negs.append(neg)
                    neg_masks.append(neg_mask)

                negs = masked_negs

                # if random.random() < 0.00001:
                # rand = random.random()
                # masked_anch.save(f'train_anch_{rand}_masked.png')
                # masked_pos.save(f'train_pos_{rand}_masked.png')
                # masked_neg.save(f'train_neg_{rand}_masked.png')

            # import pdb
            # pdb.set_trace()

            anch = self.do_transform(anch)
            pos = self.do_transform(pos)

            for i, neg in enumerate(negs):
                negs[i] = self.do_transform(neg)

            neg = torch.stack(negs)

            end = time.time()
            if utils.MY_DEC.enabled:
                print(f'HotelTrain_Metric Dataloader, transform images time: {end - start}')

            if save:
                save_image(anch, f'hotel_imagesamples/train/train_{anch_class}_{img1_random}_after.png')
                save_image(negs[0], f'hotel_imagesamples/train/train_{neg_class}_{img2_random}_after.png')

        if self.return_paths:
            return anch, pos, neg, paths
        else:
            return anch, pos, neg

    def __batchhard_getitem__(self, index):
        paths = []
        labels_to_return = []
        start = time.time()
        imgs = []
        if self.overfit:  # todo
            raise Exception('Not implemented')

        else:  # not overfitting

            label_idx = np.random.choice(self.num_classes, size=1)[0]

            label = self.labels[label_idx]

            if len(self.datas[label]) >= self.bh_K:
                random_paths = np.random.choice(self.datas[label], size=self.bh_K, replace=False)
            else:
                random_paths = np.random.choice(self.datas[label], size=self.bh_K, replace=True)

            for random_path in random_paths:
                labels_to_return.append(label)
                paths.append(random_path)
                imgs.append(Image.open(random_path).convert('RGB'))

        end = time.time()
        if utils.MY_DEC.enabled:
            print(f'HotelTrain_Metric Dataloader, choose images time: {end - start}')


        # import pdb
        # pdb.set_trace()

        if self.transform:
            start = time.time()

            if self.aug_mask:
                masked_imgs = []
                for img in imgs:
                    mask = Image.open(self.masks[np.random.randint(len(self.masks))])
                    img, masked_img, mask, _ = utils.add_mask(img, mask, colored=self.colored_mask)

                    if not self.fourth_dim:
                        img = masked_img

                    masked_imgs.append(img)

                imgs = masked_imgs

            for i, im in enumerate(imgs):
                imgs[i] = self.do_transform(im)

            imgs = torch.stack(imgs)

            end = time.time()
            if utils.MY_DEC.enabled:
                print(f'HotelTrain_Metric Dataloader, transform images time: {end - start}')

        if self.return_paths:
            return imgs, paths
        else:
            return imgs

    @utils.MY_DEC
    def __getitem__(self, index):
        if self.batchhard:
            return self.__batchhard_getitem__(index)
        else:
            return self.__triplet_getitem__(index)


    def do_transform(self, img):
        img = self.transform(img)
        img = self.normalize(img)
        return img


class HotelTrain_FewShot(Dataset):
    def __init__(self, args, transform=None, mode='train', save_pictures=False):
        super(HotelTrain_FewShot, self).__init__()
        np.random.seed(args.seed)
        self.fourth_dim = args.fourth_dim
        self.transform = transform
        self.save_pictures = save_pictures
        self.class1 = 0
        self.image1 = None
        self.no_negative = args.no_negative
        self.normalize = utils.TransformLoader(-1).transform_normalize
        self.colored_mask = args.colored_mask

        self.datas, self.num_classes, self.length, self.labels, _ = loadDataToMem(args.dataset_path, args.dataset_name,
                                                                                  mode=mode,
                                                                                  split_file_path=args.splits_file_path,
                                                                                  portion=args.portion,
                                                                                  dataset_folder=args.dataset_folder)

        self.shuffled_data = get_shuffled_data(datas=self.datas, seed=args.seed)

        self.aug_mask = args.aug_mask

        if self.aug_mask:
            self.masks = get_masks(args.dataset_path, args.dataset_folder,
                                   os.path.join(args.project_path, args.mask_path))

        else:
            self.masks = []

        print('HotelTrain_FewShot hotel train classes: ', self.num_classes)
        print('HotelTrain_FewShot hotel train length: ', self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # image1 = random.choice(self.dataset.imgs)
        label = None  # label is the distance between the two image. 1: same, 0: different
        img1 = None
        img2 = None

        # get image from same class
        if index % (self.no_negative + 1) == 0:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            self.class1 = self.labels[idx1]
            class2 = self.class1
            self.image1 = Image.open(random.choice(self.datas[self.class1]))
            image2 = Image.open(random.choice(self.datas[class2]))
        # get image from different class
        else:
            label = 0.0
            # idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            class2 = self.labels[idx2]

            while self.class1 == class2:
                idx2 = random.randint(0, self.num_classes - 1)
                class2 = self.labels[idx2]

            # class1 = self.labels[idx1]

            # image1 = Image.open(random.choice(self.datas[self.class1]))
            image2 = Image.open(random.choice(self.datas[class2]))

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

            if self.aug_mask:
                image1_mask = Image.open(self.masks[np.random.randint(len(self.masks))])

                image2_mask = Image.open(self.masks[np.random.randint(len(self.masks))])

                image1, masked_img1, image1_mask, _ = utils.add_mask(image1, image1_mask, colored=self.colored_mask)
                image2, masked_img2, image2_mask, _ = utils.add_mask(image2, image2_mask, colored=self.colored_mask)

                if not self.fourth_dim:
                    image1 = masked_img1
                    image2 = masked_img2

            image1 = self.do_transform(image1)
            image2 = self.do_transform(image2)

            if save:
                save_image(image1, f'hotel_imagesamples/train/train_{self.class1}_{img1_random}_after.png')
                save_image(image2, f'hotel_imagesamples/train/train_{class2}_{img2_random}_after.png')

        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))

    def _get_single_item(self, index):
        label, image_path = self.shuffled_data[index]

        image = Image.open(image_path)

        image = image.convert('RGB')

        if self.transform:
            image = self.do_transform(image)

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

    def do_transform(self, img):
        img = self.transform(img)
        img = self.normalize(img)
        return img


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
        self.mode = mode
        self.normalize = utils.TransformLoader(-1).transform_normalize
        self.fourth_dim = args.fourth_dim
        self.colored_mask = args.colored_mask

        self.datas, self.num_classes, _, self.labels, self.datas_bg = loadDataToMem(args.dataset_path,
                                                                                    args.dataset_name,
                                                                                    mode=mode,
                                                                                    split_file_path=args.splits_file_path,
                                                                                    portion=args.portion,
                                                                                    dataset_folder=args.dataset_folder)

        self.aug_mask = args.aug_mask

        if self.aug_mask:
            self.masks = get_masks(args.dataset_path, args.dataset_folder,
                                   os.path.join(args.project_path, args.mask_path))

        else:
            self.masks = []

        print(f'HotelTest_FewShot hotel {mode} classes: ', self.num_classes)
        print(f'HotelTest_FewShot hotel {mode} length: ', self.__len__())

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        # generate image pair from same class
        if idx == 0:
            self.c1 = self.labels[random.randint(0, self.num_classes - 1)]
            c2 = self.c1
            self.img1 = Image.open(random.choice(self.datas[self.c1])).convert('RGB')
            img2 = Image.open(random.choice(self.datas[c2])).convert('RGB')
        # generate image pair from different class
        else:
            c2 = list(self.datas_bg.keys())[random.randint(0, len(self.datas_bg.keys()) - 1)]
            while self.c1 == c2:
                c2 = list(self.datas_bg.keys())[random.randint(0, len(self.datas_bg.keys()) - 1)]
            # print(f'idx = {idx}, c1 was {self.c1} and negative c2 is {c2}')
            if self.mode == 'train':
                img2 = Image.open(random.choice(self.datas_bg[c2])).convert('RGB')
            else:
                img2 = Image.open(random.choice(self.datas_bg[c2])[0]).convert('RGB')
        save = False
        img1 = self.img1
        if self.transform:
            if self.save_pictures and random.random() < 0.001:
                save = True
                img1_random = random.randint(0, 1000)
                img2_random = random.randint(0, 1000)
                img1.save(f'hotel_imagesamples/val/val_{self.c1}_{img1_random}_before.png')
                img2.save(f'hotel_imagesamples/val/val_{c2}_{img2_random}_before.png')

            if self.aug_mask:
                img1_mask = Image.open(self.masks[np.random.randint(len(self.masks))])

                img2_mask = Image.open(self.masks[np.random.randint(len(self.masks))])

                img1, masked_img1, img1_mask, _ = utils.add_mask(img1, img1_mask, colored=self.colored_mask)
                img2, masked_img2, img2_mask, _ = utils.add_mask(img2, img2_mask, colored=self.colored_mask)

                if not self.fourth_dim:
                    img1 = masked_img1
                    img2 = masked_img2

            img1 = self.do_transform(img1)
            img2 = self.do_transform(img2)

            if save:
                save_image(img1, f'hotel_imagesamples/val/val_{self.c1}_{img1_random}_after.png')
                save_image(img2, f'hotel_imagesamples/val/val_{c2}_{img2_random}_after.png')

        return img1, img2

    def do_transform(self, img):
        img = self.transform(img)
        img = self.normalize(img)
        return img


class HotelTest_EdgePred(Dataset):

    def __init__(self, args, transform=None, mode='test_seen', save_pictures=False):
        np.random.seed(args.seed)
        super(HotelTest_EdgePred, self).__init__()
        self.transform = transform
        self.save_pictures = save_pictures
        self.times = args.times
        self.way = args.way  # number of classes
        self.k = args.test_k  # number of images per class

        self.img_idx = 0
        self.class_idx = 0
        self.imgs = []
        self.classes = []

        self.mode = mode
        self.normalize = utils.TransformLoader(-1).transform_normalize
        self.fourth_dim = args.fourth_dim
        self.colored_mask = args.colored_mask

        self.datas, self.num_classes, _, self.labels, self.datas_bg = loadDataToMem(args.dataset_path,
                                                                                    args.dataset_name,
                                                                                    mode=mode,
                                                                                    split_file_path=args.splits_file_path,
                                                                                    portion=args.portion,
                                                                                    dataset_folder=args.dataset_folder)

        self.aug_mask = args.aug_mask

        if self.aug_mask:
            self.masks = get_masks(args.dataset_path, args.dataset_folder,
                                   os.path.join(args.project_path, args.mask_path))

        else:
            self.masks = []

        print(f'HotelTest_EdgePred hotel {mode} classes: ', self.num_classes)
        print(f'HotelTest_EdgePred hotel {mode} length: ', self.__len__())

    def __len__(self):
        return self.times * self.way * self.k

    def __getitem__(self, index):

        # generate image pair from same class
        if index % (self.way * self.k) == 0:
            self.classes = self.labels[np.random.randint(0, self.num_classes, size=self.way)]
            self.class_idx = 0
            self.img_idx = 0
            label = self.classes[self.class_idx]
            self.imgs = self.datas[label][
                np.random.randint(0, len(self.datas[label]),
                                  size=self.way)]

        else:
            label = self.classes[self.class_idx]

        if self.img_idx == self.k:
            self.class_idx += 1
            self.img_idx = 0
            label = self.classes[self.class_idx]
            self.imgs = self.datas[label][
                np.random.randint(0, len(self.datas[label]),
                                  size=self.way)]
        else:
            self.img_idx += 1

        img = Image.open(self.imgs[self.img_idx]).convert('RGB')

        save = False

        if self.transform:
            if self.save_pictures and random.random() < 0.001:
                save = True
                img_random = random.randint(0, 1000)

                img.save(f'hotel_imagesamples/val/val_{label}_{img_random}_before.png')

            if self.aug_mask:
                img_mask = Image.open(self.masks[np.random.randint(len(self.masks))])

                img, masked_img, img_mask, _ = utils.add_mask(img, img_mask, colored=self.colored_mask)

                if not self.fourth_dim:
                    img = masked_img

            img = self.do_transform(img)

            if save:
                save_image(img, f'hotel_imagesamples/val/val_{label}_{img_random}_after.png')

        return img

    def do_transform(self, img):
        img = self.transform(img)
        img = self.normalize(img)
        return img


class Hotel_DB(Dataset):
    def __init__(self, args, transform=None, mode='test'):
        np.random.seed(args.seed)
        super(Hotel_DB, self).__init__()
        self.transform = transform
        self.fourth_dim = args.fourth_dim

        self.mode = mode
        self.normalize = utils.TransformLoader(-1).transform_normalize
        self.colored_mask = args.colored_mask

        total = True
        if self.mode == 'val' or self.mode == 'test':  # mode == *_seen or *_unseen or train
            self.mode_tmp = self.mode + '_seen'
            total = True
        else:
            self.mode_tmp = self.mode
            total = False

        self.datas, self.num_classes, _, self.labels, self.all_data = loadDataToMem(args.dataset_path,
                                                                                    args.dataset_name,
                                                                                    mode=self.mode_tmp,
                                                                                    split_file_path=args.splits_file_path,
                                                                                    portion=args.portion,
                                                                                    dataset_folder=args.dataset_folder,
                                                                                    return_bg=(self.mode != 'train'))
        self.all_shuffled_data = get_shuffled_data(self.all_data,
                                                   seed=args.seed,
                                                   one_hot=False,
                                                   both_seen_unseen=(self.mode != 'train'),
                                                   shuffle=False)

        self.aug_mask = args.aug_mask

        if self.aug_mask:
            self.masks = get_masks(args.dataset_path, args.dataset_folder,
                                   os.path.join(args.project_path, args.mask_path))

        else:
            self.masks = []

        # else: # todo
        #     self.all_shuffled_data = get_shuffled_data(self.datas, seed=args.seed, one_hot=False)

        print(f'Hotel_DB hotel {self.mode} classes: ', self.num_classes)
        print(f'Hotel_DB hotel {self.mode} length: ', self.__len__())

    def __len__(self):
        return len(self.all_shuffled_data)

    def __getitem__(self, index):
        lbl = self.all_shuffled_data[index][0]
        img = Image.open(self.all_shuffled_data[index][1]).convert('RGB')
        if self.mode != 'train':
            bl = self.all_shuffled_data[index][2]

        path = self.all_shuffled_data[index][1].split('/')

        id = path[-4]
        id += '-' + path[-3]
        id += '-' + path[-1].split('.')[0]

        if self.transform:
            if self.aug_mask:
                img_mask = Image.open(self.masks[np.random.randint(len(self.masks))])
                img, masked_img, img_mask, _ = utils.add_mask(img, img_mask, colored=self.colored_mask)

                if not self.fourth_dim:
                    img = masked_img

            img = self.do_transform(img)

        if self.mode != 'train':
            return img, lbl, bl, id
        else:
            return img, lbl, id

    def do_transform(self, img):
        img = self.transform(img)
        img = self.normalize(img)
        return img
