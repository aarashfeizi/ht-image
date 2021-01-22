import argparse
import datetime
import json
import math
import multiprocessing
import os
import time

import cv2
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import metrics

matplotlib.rc('font', size=24)

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


class SwitchedDecorator:
    def __init__(self, enabled_func):
        self._enabled = False
        self._enabled_func = enabled_func

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, new_value):
        if not isinstance(new_value, bool):
            raise ValueError("enabled can only be set to a boolean value")
        self._enabled = new_value

    def __call__(self, target):
        if self._enabled:
            return self._enabled_func(target)
        return target


def time_it(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = fn(*args, **kwargs)
        end = time.time()
        print(f'Function {str(fn.__name__)}, time:  {end - start}')
        return ret

    return wrapper


MY_DEC = SwitchedDecorator(time_it)


class TransformLoader:

    def __init__(self, image_size, rotate=0,
                 normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
        self.rotate = rotate
        self.normalize = transforms.Normalize(**self.normalize_param)

    def parse_transform(self, transform_type):
        # if transform_type == 'ImageJitter':
        #     method = add_transforms.ImageJitter(self.jitter_param)
        #     return method
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomResizedCrop':
            return method(self.image_size, scale=[0.5, 1.0], ratio=[1.0, 1.0])
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Resize':
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        elif transform_type == 'RandomRotation':
            return method(self.rotate)
        else:
            return method()

    def get_composed_transform(self, aug=False, random_crop=False, for_network=True):
        transform_list = []

        if aug:
            transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip']
        elif not aug and self.rotate == 0:
            transform_list = ['Resize']
        elif not aug and self.rotate != 0:
            transform_list = ['Resize', 'RandomRotation']

        if random_crop:
            transform_list.extend(['RandomResizedCrop'])
        else:
            transform_list.extend(['CenterCrop'])

        if for_network:
            transform_list.extend(['ToTensor'])

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform, transform_list

    def transform_normalize(self, img):
        if img.shape[0] == 3:
            return self.normalize(img)
        else:
            half_normalized = self.normalize(img[0:3, :, :])
            ret = torch.cat([half_normalized, img[3, :, :].unsqueeze(dim=0)], dim=0)
            return ret

    # '../../dataset/omniglot/python/images_background'


# '../../dataset/omniglot/python/images_evaluation'
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cuda', '--cuda', default=False, action='store_true')
    parser.add_argument('-gpu', '--gpu_ids', default='', help="gpu ids used to train")  # before: default="0,1,2,3"
    parser.add_argument('-env', '--env', default='local',
                        help="where the code is being run, e.g. local, beluga, graham")  # before: default="0,1,2,3"
    parser.add_argument('-on', '--overfit_num', default=0, type=int)
    parser.add_argument('-dsn', '--dataset_name', default='omniglot', choices=['omniglot', 'cub', 'hotels'])
    parser.add_argument('-dsp', '--dataset_path', default='')
    parser.add_argument('-por', '--portion', default=0, type=int)
    parser.add_argument('-ls', '--limit_samples', default=0, type=int, help="Limit samples per class for val and test")
    parser.add_argument('-nor', '--number_of_runs', default=1, type=int, help="Number of times to sample for k@n")
    parser.add_argument('-sp', '--save_path', default='savedmodels/', help="path to store model")
    parser.add_argument('-lp', '--log_path', default='logs/', help="path to log")
    parser.add_argument('-tbp', '--tb_path', default='tensorboard/', help="path for tensorboard")
    parser.add_argument('-a', '--aug', default=False, action='store_true')
    # parser.add_argument('-m', '--mask', default=False, action='store_true')
    parser.add_argument('-r', '--rotate', default=0.0, type=float)
    parser.add_argument('-mn', '--pretrained_model_name', default='')
    parser.add_argument('-pmd', '--pretrained_model_dir', default='')
    parser.add_argument('-ev', '--eval_mode', default='fewshot', choices=['fewshot', 'simple'])
    parser.add_argument('-fe', '--feat_extractor', default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'vgg16'])
    parser.add_argument('-pool', '--pooling', default='spoc',
                        choices=['spoc', 'gem', 'mac', 'rmac'])
    parser.add_argument('-fr', '--freeze_ext', default=False, action='store_true')
    parser.add_argument('-el', '--extra_layer', default=0, type=int,
                        help="Number of 512 extra layers in the Li-Siamese")
    parser.add_argument('-nn', '--no_negative', default=1, type=int)
    parser.add_argument('-en', '--extra_name', default='')

    parser.add_argument('-s', '--seed', default=402, type=int, help="random seed")
    parser.add_argument('-w', '--way', default=20, type=int, help="how much way one-shot learning")
    parser.add_argument('-t', '--times', default=400, type=int, help="number of samples to test accuracy")
    parser.add_argument('-wr', '--workers', default=4, type=int, help="number of dataLoader workers")
    parser.add_argument('-pim', '--pin_memory', default=False, action='store_true')
    parser.add_argument('-fbw', '--find_best_workers', default=False, action='store_true')
    parser.add_argument('-bs', '--batch_size', default=128, type=int, help="number of batch size")
    parser.add_argument('-dbb', '--db_batch', default=128, type=int, help="number of batch size for db")
    parser.add_argument('-lrs', '--lr_siamese', default=1e-3, type=float, help="siamese learning rate")
    parser.add_argument('-lrr', '--lr_resnet', default=1e-6, type=float, help="resnet learning rate")
    parser.add_argument('-lf', '--log_freq', default=10, type=int, help="show result after each show_every iter.")
    parser.add_argument('-sf', '--save_freq', default=100, type=int, help="save model after each save_every iter.")
    parser.add_argument('-tf', '--test_freq', default=100, type=int, help="test model after each test_every iter.")
    # parser.add_argument('-ms', '--max_steps', default=50000, type=int, help="number of steps before stopping")
    parser.add_argument('-ep', '--epochs', default=1, type=int, help="number of epochs before stopping")
    parser.add_argument('-es', '--early_stopping', default=20, type=int, help="number of tol for validation acc")
    parser.add_argument('-tst', '--test', default=False, action='store_true')
    parser.add_argument('-katn', '--katn', default=False, action='store_true')
    parser.add_argument('-cbir', '--cbir', default=False, action='store_true')
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('-sr', '--sampled_results', default=True, action='store_true')
    parser.add_argument('-pcr', '--per_class_results', default=True, action='store_true')
    parser.add_argument('-ptb', '--project_tb', default=False, action='store_true')

    parser.add_argument('-mtlr', '--metric_learning', default=False, action='store_true')
    parser.add_argument('-mg', '--margin', default=0.0, type=float, help="margin for triplet loss")
    parser.add_argument('-lss', '--loss', default='bce', choices=['bce', 'trpl', 'maxmargin'])
    parser.add_argument('-mm', '--merge_method', default='sim', choices=['sim', 'diff', 'diff-sim'])
    parser.add_argument('-bco', '--bcecoefficient', default=1.0, type=float, help="BCE loss weight")
    parser.add_argument('-kbm', '--k_best_maps', nargs='+', help="list of k best activation maps")

    parser.add_argument('-n', '--normalize', default=False, action='store_true')
    parser.add_argument('-dg', '--debug_grad', default=False, action='store_true')
    parser.add_argument('-cam', '--cam', default=False, action='store_true')
    parser.add_argument('-dat', '--draw_all_thresh', default=32, type=int, help="threshold for drawing all heatmaps")
    parser.add_argument('-m', '--aug_mask', default=False, action='store_true')
    parser.add_argument('-cm', '--colored_mask', default=False, action='store_true')
    parser.add_argument('-fs', '--from_scratch', default=False, action='store_true')
    parser.add_argument('-fd', '--fourth_dim', default=False, action='store_true')
    parser.add_argument('-camp', '--cam_path', default='cam_info.txt')
    parser.add_argument('-ppth', '--project_path',
                        default='/home/aarash/projects/def-rrabba/aarash/ht-image-twoloss/ht-image/')
    parser.add_argument('-lpth', '--local_path',
                        default='/home/aarash/projects/def-rrabba/aarash/ht-image-twoloss/ht-image/')
    parser.add_argument('-jid', '--job_id', default='')

    args = parser.parse_args()

    return args


def loading_time(args, train_set, use_cuda, num_workers, pin_memory, logger):
    kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    start = time.time()
    for epoch in range(4):
        for batch_idx, (_, _, _) in enumerate(train_loader):
            if batch_idx == 15:
                break
            pass
    end = time.time()
    logger.info("  Used {} second with num_workers = {}".format(end - start, num_workers))
    return end - start


def get_best_workers_pinmemory(args, train_set, pin_memories=[False, True], starting_from=0, logger=None):
    use_cuda = torch.cuda.is_available()
    core_number = multiprocessing.cpu_count()
    batch_size = 64
    best_num_worker = [0, 0]
    best_time = [99999999, 99999999]
    logger.info(f'cpu_count = {core_number}')

    for pin_memory in pin_memories:
        logger.info(f'While pin_memory = {pin_memory}')
        for num_workers in range(starting_from, core_number * 2 + 1, 4):
            current_time = loading_time(args, train_set, use_cuda, num_workers, pin_memory, logger)
            if current_time < best_time[pin_memory]:
                best_time[pin_memory] = current_time
                best_num_worker[pin_memory] = num_workers
            else:  # assuming its a convex function
                if best_num_worker[pin_memory] == 0:
                    the_range = []
                else:
                    the_range = list(range(best_num_worker[pin_memory] - 3, best_num_worker[pin_memory]))
                for num_workers in (
                        the_range + list(range(best_num_worker[pin_memory] + 1, best_num_worker[pin_memory] + 4))):
                    current_time = loading_time(args, train_set, use_cuda, num_workers, pin_memory, logger)
                    if current_time < best_time[pin_memory]:
                        best_time[pin_memory] = current_time
                        best_num_worker[pin_memory] = num_workers
                break
    if best_time[0] < best_time[1]:
        logger.info(f"Best num_workers = {best_num_worker[0]} with pin_memory = False")
        workers = best_num_worker[0]
        pin_memory = False
    else:
        logger.info(f"Best num_workers = {best_num_worker[1]} with pin_memory = True")
        workers = best_num_worker[1]
        pin_memory = True

    return workers, pin_memory


def get_val_loaders(args, val_set, val_set_known, val_set_unknown, workers, pin_memory, batch_size=None):
    val_loaders = []
    if batch_size is None:
        batch_size = args.way
    if (val_set is not None) or (val_set_known is not None):

        val_loaders.append(
            DataLoader(val_set_known, batch_size=batch_size, shuffle=False, num_workers=workers,
                       pin_memory=pin_memory, drop_last=True))
        val_loaders.append(
            DataLoader(val_set_unknown, batch_size=batch_size, shuffle=False, num_workers=workers,
                       pin_memory=pin_memory, drop_last=True))
    else:
        raise Exception('No validation data is set!')

    return val_loaders


def save_h5(data_description, data, data_type, path):
    h5_feats = h5py.File(path, 'w')
    h5_feats.create_dataset(data_description, data=data, dtype=data_type)
    h5_feats.close()


def load_h5(data_description, path):
    data = None
    with h5py.File(path, 'r') as hf:
        data = hf[data_description][:]
    return data


def calculate_k_at_n(args, img_feats, img_lbls, seen_list, logger, limit=0, run_number=0, sampled=True,
                     per_class=False, save_path='', mode=''):
    if per_class:
        logger.info('K@N per class')
        total, seen, unseen = _get_per_class_distance(args, img_feats, img_lbls, seen_list, logger, mode)
        total.to_csv(os.path.join(save_path, f'{mode}_per_class_total_avg_k@n.csv'), header=True, index=False)
        seen.to_csv(os.path.join(save_path, f'{mode}_per_class_seen_avg_k@n.csv'), header=True, index=False)
        unseen.to_csv(os.path.join(save_path, f'{mode}_per_class_unseen_avg_k@n.csv'), header=True, index=False)

    if sampled:
        logger.info('K@N for sampled')
        kavg, kruns, total, seen, unseen = _get_sampled_distance(args, img_feats, img_lbls, seen_list, logger, limit,
                                                                 run_number, mode)
        kavg.to_csv(os.path.join(save_path, f'{mode}_sampled_avg_k@n.csv'), header=True, index=False)
        kruns.to_csv(os.path.join(save_path, f'{mode}_sampled_runs_k@n.csv'), header=True, index=False)
        total.to_csv(os.path.join(save_path, f'{mode}_sampled_per_class_total_avg_k@n.csv'), header=True, index=False)
        seen.to_csv(os.path.join(save_path, f'{mode}_sampled_per_class_seen_avg_k@n.csv'), header=True, index=False)
        unseen.to_csv(os.path.join(save_path, f'{mode}_sampled_per_class_unseen_avg_k@n.csv'), header=True, index=False)

    return True


def _get_per_class_distance(args, img_feats, img_lbls, seen_list, logger, mode):
    all_lbls = np.unique(img_lbls)
    seen_lbls = np.unique(img_lbls[seen_list == 1])
    unseen_lbls = np.unique(img_lbls[seen_list == 0])

    sim_mat = cosine_similarity(img_feats)

    metric_total = metrics.Accuracy_At_K(classes=np.array(all_lbls))
    metric_seen = metrics.Accuracy_At_K(classes=np.array(seen_lbls))
    metric_unseen = metrics.Accuracy_At_K(classes=np.array(unseen_lbls))

    for idx, (row, lbl, seen) in enumerate(zip(sim_mat, img_lbls, seen_list)):
        ret_scores = np.delete(row, idx)
        ret_lbls = np.delete(img_lbls, idx)
        ret_seens = np.delete(seen_list, idx)

        ret_lbls = _sort_according_to(ret_lbls, ret_scores)
        ret_seens = _sort_according_to(ret_seens, ret_scores)

        metric_total.update(lbl, ret_lbls)

        if seen == 1:
            metric_seen.update(lbl, ret_lbls[ret_seens == 1])
        else:
            metric_unseen.update(lbl, ret_lbls[ret_seens == 0])

    total = metric_total.get_per_class_metrics()
    seen = metric_seen.get_per_class_metrics()
    unseen = metric_unseen.get_per_class_metrics()

    logger.info(f'{mode}')
    logger.info('Without sampling Total: ' + str(metric_total.n))
    logger.info(metric_total)

    logger.info(f'{mode}')
    _log_per_class(logger, total, split_kind='Total')

    logger.info(f'{mode}')
    logger.info('Without sampling Seen: ' + str(metric_seen.n))
    logger.info(metric_seen)

    logger.info(f'{mode}')
    _log_per_class(logger, seen, split_kind='Seen')

    logger.info(f'{mode}')
    logger.info('Without sampling Unseen: ' + str(metric_unseen.n))
    logger.info(metric_unseen)

    logger.info(f'{mode}')
    _log_per_class(logger, unseen, split_kind='Unseen')

    return total, seen, unseen


def _sort_according_to(arr1, arr2):
    arr1 = [x for _, x in sorted(zip(arr2, arr1), reverse=True)]

    arr1 = np.array(arr1)

    return arr1


def _log_per_class(logger, df, split_kind=''):
    logger.info(f'Per class {split_kind}: {np.array(df["n"]).sum()}')
    logger.info(f'Average per class {split_kind}: {np.array(df["n"]).mean()}')
    logger.info(f'k@1 per class average: {np.array(df["k@1"]).mean()}')
    logger.info(f'k@5 per class average: {np.array(df["k@5"]).mean()}')
    logger.info(f'k@10 per class average: {np.array(df["k@10"]).mean()}')
    logger.info(f'k@100 per class average: {np.array(df["k@100"]).mean()}\n')


def _get_sampled_distance(args, img_feats, img_lbls, seen_list, logger, limit=0, run_number=0, mode=''):
    all_lbls = np.unique(img_lbls)
    seen_lbls = np.unique(img_lbls[seen_list == 1])
    unseen_lbls = np.unique(img_lbls[seen_list == 0])

    k1s = []
    k5s = []
    k10s = []
    k100s = []

    k1s_s = []
    k5s_s = []
    k10s_s = []
    k100s_s = []

    k1s_u = []
    k5s_u = []
    k10s_u = []
    k100s_u = []

    if args.dataset_name != 'cub':
        sampled_indices_all = pd.read_csv(
            os.path.join(args.project_path, 'sample_index_por' + str(args.portion) + '.csv'))
        sampled_label_all = pd.read_csv(
            os.path.join(args.project_path, 'sample_label_por' + str(args.portion) + '.csv'))

    for run in range(run_number):
        column_name = f'run{run}'
        if args.dataset_name != 'cub':
            sampled_indices = np.array(sampled_indices_all[column_name]).astype(int)
            sampled_labels = np.array(sampled_label_all[column_name]).astype(int)

            logger.info(f'{mode}')
            logger.info('### Run ' + str(run) + "...")
            chosen_img_feats = img_feats[sampled_indices]
            chosen_img_lbls = img_lbls[sampled_indices]
            chosen_seen_list = seen_list[sampled_indices]

            assert np.array_equal(sampled_labels, chosen_img_lbls)

        else:  # dataset os cub
            logger.info(f'### Run {run} on cub...')

            chosen_img_feats = img_feats
            chosen_img_lbls = img_lbls
            chosen_seen_list = seen_list

        sim_mat = cosine_similarity(chosen_img_feats)
        metric_total = metrics.Accuracy_At_K(classes=all_lbls)
        metric_seen = metrics.Accuracy_At_K(classes=seen_lbls)
        metric_unseen = metrics.Accuracy_At_K(classes=unseen_lbls)

        for idx, (row, lbl, seen) in enumerate(zip(sim_mat, chosen_img_lbls, chosen_seen_list)):
            ret_scores = np.delete(row, idx)
            ret_lbls = np.delete(chosen_img_lbls, idx)
            ret_seens = np.delete(chosen_seen_list, idx)

            ret_lbls = _sort_according_to(ret_lbls, ret_scores)
            ret_seens = _sort_according_to(ret_seens, ret_scores)

            metric_total.update(lbl, ret_lbls)

            if seen == 1:
                metric_seen.update(lbl, ret_lbls[ret_seens == 1])
            else:
                metric_unseen.update(lbl, ret_lbls[ret_seens == 0])

        total = metric_total.get_per_class_metrics()
        seen = metric_seen.get_per_class_metrics()
        unseen = metric_unseen.get_per_class_metrics()

        logger.info('Total: ' + str(metric_total.n))
        logger.info(metric_total)
        k1, k5, k10, k100 = metric_total.get_tot_metrics()
        k1s.append(k1)
        k5s.append(k5)
        k10s.append(k10)
        k100s.append(k100)
        logger.info("*" * 50)

        logger.info('Seen: ' + str(metric_seen.n))
        logger.info(metric_seen)
        k1, k5, k10, k100 = metric_seen.get_tot_metrics()
        k1s_s.append(k1)
        k5s_s.append(k5)
        k10s_s.append(k10)
        k100s_s.append(k100)
        logger.info("*" * 50)

        logger.info('Unseen: ' + str(metric_unseen.n))
        logger.info(metric_unseen)
        k1, k5, k10, k100 = metric_unseen.get_tot_metrics()
        k1s_u.append(k1)
        k5s_u.append(k5)
        k10s_u.append(k10)
        k100s_u.append(k100)
        logger.info("*" * 50)

        _log_per_class(logger, total, split_kind='Total')
        _log_per_class(logger, seen, split_kind='Seen')
        _log_per_class(logger, unseen, split_kind='Unseen')

    total = metric_total.get_per_class_metrics()
    seen = metric_seen.get_per_class_metrics()
    unseen = metric_unseen.get_per_class_metrics()

    logger.info('Avg Total: ' + str(metric_total.n))
    logger.info('k@1: ' + str(np.array(k1s).mean()))
    logger.info('k@5: ' + str(np.array(k5s).mean()))
    logger.info('k@10: ' + str(np.array(k10s).mean()))
    logger.info('k@100: ' + str(np.array(k100s).mean()))
    logger.info("*" * 50)

    logger.info('Avg Seen: ' + str(metric_seen.n))
    logger.info('k@1: ' + str(np.array(k1s_s).mean()))
    logger.info('k@5: ' + str(np.array(k5s_s).mean()))
    logger.info('k@10: ' + str(np.array(k10s_s).mean()))
    logger.info('k@100: ' + str(np.array(k100s_s).mean()))
    logger.info("*" * 50)

    logger.info('Avg Unseen: ' + str(metric_unseen.n))
    logger.info('k@1: ' + str(np.array(k1s_u).mean()))
    logger.info('k@5: ' + str(np.array(k5s_u).mean()))
    logger.info('k@10: ' + str(np.array(k10s_u).mean()))
    logger.info('k@100: ' + str(np.array(k100s_u).mean()))
    logger.info("*" * 50)

    d = {'run': [i for i in range(run_number)],
         'kAT1': k1s,
         'kAT5': k5s,
         'kAT10': k10s,
         'kAT100': k100s,
         'kAT1_seen': k1s_s,
         'kAT5_seen': k5s_s,
         'kAT10_seen': k10s_s,
         'kAT100_seen': k100s_s,
         'kAT1_unseen': k1s_u,
         'kAT5_unseen': k5s_u,
         'kAT10_unseen': k10s_u,
         'kAT100_unseen': k100s_u}

    average_tot = pd.DataFrame(data={'avg_kAT1': [np.array(k1s).mean()],
                                     'avg_kAT5': [np.array(k5s).mean()],
                                     'avg_kAT10': [np.array(k10s).mean()],
                                     'avg_kAT100': [np.array(k100s).mean()],
                                     'avg_kAT1_seen': [np.array(k1s_s).mean()],
                                     'avg_kAT5_seen': [np.array(k5s_s).mean()],
                                     'avg_kAT10_seen': [np.array(k10s_s).mean()],
                                     'avg_kAT100_seen': [np.array(k100s_s).mean()],
                                     'avg_kAT1_unseen': [np.array(k1s_u).mean()],
                                     'avg_kAT5_unseen': [np.array(k5s_u).mean()],
                                     'avg_kAT10_unseen': [np.array(k10s_u).mean()],
                                     'avg_kAT100_unseen': [np.array(k100s_u).mean()]})

    return average_tot, pd.DataFrame(data=d), total, seen, unseen


def get_shuffled_data(datas, seed=0, one_hot=True, both_seen_unseen=False, shuffle=True):  # for sequential labels only

    labels = sorted(datas.keys())

    if one_hot:
        lbl2idx = {labels[idx]: idx for idx in range(len(labels))}
        one_hot_labels = np.eye(len(np.unique(labels)))
    # print(one_hot_labels)

    np.random.seed(seed)

    data = []
    for key, value_list in datas.items():
        if one_hot:
            lbl = one_hot_labels[lbl2idx[key]]
        else:
            lbl = key

        if both_seen_unseen:
            ls = [(lbl, value, bl) for value, bl in value_list]  # todo to be able to separate seen and unseen in k@n
        else:
            ls = [(lbl, value) for value in value_list]

        data.extend(ls)

    if shuffle:
        np.random.shuffle(data)

    return data


def _read_org_split(dataset_path, mode):
    image_labels = []
    image_path = []

    if mode == 'train':  # train

        with open(os.path.join(dataset_path, 'base.json'), 'r') as f:
            base_dict = json.load(f)
        image_labels.extend(base_dict['image_labels'])
        image_path.extend(base_dict['image_names'])

    elif mode == 'val':  # val

        with open(os.path.join(dataset_path, 'val.json'), 'r') as f:
            val_dict = json.load(f)
        image_labels.extend(val_dict['image_labels'])
        image_path.extend(val_dict['image_names'])

    elif mode == 'test':  # novel classes

        with open(os.path.join(dataset_path, 'novel.json'), 'r') as f:
            novel_dict = json.load(f)
        image_labels.extend(novel_dict['image_labels'])
        image_path.extend(novel_dict['image_names'])

    return image_path, image_labels


def _read_new_split(dataset_path, mode,
                    dataset_name='cub'):  # mode = [test_seen, val_seen, train, test_unseen, test_unseen]

    file_name = f'{dataset_name}_' + mode + '.csv'

    file = pd.read_csv(os.path.join(dataset_path, file_name))
    image_labels = np.array(file.label)
    image_path = np.array(file.image)

    return image_path, image_labels


def loadDataToMem(dataPath, dataset_name, mode='train', split_file_path='',
                  portion=0, return_bg=True, dataset_folder=''):
    print(split_file_path, '!!!!!!!!')
    dataset_path = os.path.join(dataPath, dataset_folder)

    return_bg = return_bg and (mode != 'train')

    background_datasets = {'val_seen': 'val_unseen',
                           'val_unseen': 'val_seen',
                           'test_seen': 'test_unseen',
                           'test_unseen': 'test_seen',
                           'train_seen': 'train_seen'}

    print("begin loading dataset to memory")
    datas = {}
    datas_bg = {}  # in case of mode == val/test_seen/unseen

    image_path, image_labels = _read_new_split(split_file_path, mode, dataset_name)
    if return_bg:
        image_path_bg, image_labels_bg = _read_new_split(split_file_path,
                                                         background_datasets[mode], dataset_name)

    if portion > 0:
        image_path = image_path[image_labels < portion]
        image_labels = image_labels[image_labels < portion]

        if return_bg:
            image_path_bg = image_path_bg[image_labels_bg < portion]
            image_labels_bg = image_labels_bg[image_labels_bg < portion]

    print(f'{mode} number of imgs:', len(image_labels))
    print(f'{mode} number of labels:', len(np.unique(image_labels)))

    if return_bg:
        print(f'{mode} number of bg imgs:', len(image_labels_bg))
        print(f'{mode} number of bg lbls:', len(np.unique(image_labels_bg)))
    else:
        print(f'Just {mode}, background not required.')

    num_instances = len(image_labels)

    num_classes = len(np.unique(image_labels))

    for idx, path in zip(image_labels, image_path):
        if idx not in datas.keys():
            datas[idx] = []
            if return_bg:
                datas_bg[idx] = []

        datas[idx].append(os.path.join(dataset_path, path))
        if return_bg:
            datas_bg[idx].append((os.path.join(dataset_path, path), True))

    if return_bg:
        for idx, path in zip(image_labels_bg, image_path_bg):
            if idx not in datas_bg.keys():
                datas_bg[idx] = []
            if (os.path.join(dataset_path, path), False) not in datas_bg[idx] and \
                    (os.path.join(dataset_path, path), True) not in datas_bg[idx]:
                datas_bg[idx].append((os.path.join(dataset_path, path), False))

    labels = np.unique(image_labels)
    print(f'Number of labels in {mode}: ', len(labels))

    if return_bg:
        all_labels = np.unique(np.concatenate((image_labels, image_labels_bg)))
        print(f'Number of all labels (bg + fg) in {mode} and {background_datasets[mode]}: ', len(all_labels))

    if not return_bg:
        datas_bg = datas

    print(f'finish loading {mode} dataset to memory')
    return datas, num_classes, num_instances, labels, datas_bg


def project_2d(features, labels, title):
    pca = PCA(n_components=2)
    pca_feats = pca.fit_transform(features)
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    plt.scatter(pca_feats[:, 0], pca_feats[:, 1], c=labels, cmap=cmap, alpha=0.2)
    plt.colorbar()
    plt.title(title)

    return plt


def choose_n_from_all(df, n=4):
    chosen_labels = []
    chosen_images = []

    lbls = np.array(df.label)
    images = np.array(df.image)

    lbls_unique = np.unique(lbls)

    for lbl in lbls_unique:
        mask = lbls == lbl
        single_lbl_paths = images[mask]

        if len(single_lbl_paths) > n:
            temp = np.random.choice(single_lbl_paths, size=n, replace=False)
            chosen_images.extend(temp)
            chosen_labels.extend([lbl for _ in range(n)])
        else:
            chosen_images.extend(single_lbl_paths)
            chosen_labels.extend([lbl for _ in range(len(single_lbl_paths))])

    data = {'label': chosen_labels, 'image': chosen_images}

    return pd.DataFrame(data=data)


def create_save_path(path, id_str, logger):
    if not os.path.exists(path):
        os.mkdir(path)
        logger.info(
            f'Created save and tensorboard directories:\n{path}\n')
    else:
        logger.info(f'Save directory {path} already exists, but how?? {id_str}')  # almost impossible



def read_masks(path):
    # create mask csv
    # read mask csv and paths
    masks = pd.read_csv(path)
    return masks


def get_overfit(data, labels, anchors=1, neg_per_pos=1):
    anch_class = np.random.choice(labels, 1)
    neg_class = anch_class
    while neg_class == anch_class:
        neg_class = np.random.choice(labels, 1)

    triplets = []

    for i in range(anchors):
        anch_class = np.random.choice(labels, 1)[0]
        neg_class = anch_class

        while neg_class == anch_class:
            neg_class = np.random.choice(labels, 1)[0]

        anch_path = np.random.choice(data[anch_class], 1)[0]
        pos_path = np.random.choice(data[anch_class], 1)[0]

        while anch_path == pos_path:
            pos_path = np.random.choice(data[anch_class], 1)[0]

        negs = []
        for j in range(neg_per_pos):
            neg_path = np.random.choice(data[neg_class], 1)[0]

            while neg_path in negs:
                neg_path = np.random.choice(data[neg_class], 1)[0]

            negs.append(neg_path)

            triplets.append({'anch': anch_path, 'pos': pos_path, 'neg': negs})

    return triplets


def bar_plot_grad_flow(args, named_parameters, label, batch_id, epoch, save_path):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    plt.figure(figsize=(64, 48))
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            if n == 'ft_net.fc.weight':
                continue
            if p.grad is None:
                # print(f'{label} {n} none grad!!! *********')
                continue
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers)
    plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title(f"Gradient flow for  {label}_epoch{epoch}_batch{batch_id}")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(os.path.join(save_path, f'bars_{args.loss}_bco{args.bcecoefficient}_{label}_batch{batch_id}.png'))
    plt.close('all')


def line_plot_grad_flow(args, named_parameters, label, batch_id, epoch, save_path):
    ave_grads = []
    layers = []
    plt.figure(figsize=(64, 48))
    for n, p in named_parameters.items():
        if (p.requires_grad) and ("bias" not in n):
            if n == 'ft_net.fc.weight':
                continue
            if p.grad is None:
                # print(f'{label} {n} none grad!!! *********')
                continue
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title(f"Gradient flow for {label}_epoch{epoch}_batch{batch_id}")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'line_{args.loss}_bco{args.bcecoefficient}_{label}_batch{batch_id}.png'))
    plt.close('all')


def two_line_plot_grad_flow(args, triplet_np, bce_np, name_label, batch_id, epoch, save_path):
    ave_grads = []
    layers = []
    plt.figure(figsize=(64, 48))
    plt.rcParams.update({'font.size': 15})  # must set in top

    colors = {'average_bce': 'blue',
              'average_triplet': 'red'}

    for label, lists in [('average_bce', bce_np), ('average_triplet', triplet_np)]:
        ave_grads = lists[0]
        layers = lists[2]

        plt.plot(ave_grads, color=colors[label])
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.legend([Line2D([0], [0], color=colors[f'average_bce'], lw=4),
                Line2D([0], [0], color=colors[f'average_triplet'], lw=4),
                Line2D([0], [0], color="k", lw=4)],
               ['bce-mean-gradient',
                'triplet-mean-gradient', 'zero-gradient'])
    plt.title(f"Gradient flow for {name_label}_epoch{epoch}_batch{batch_id}")
    plt.grid(True)
    plt.savefig(
        os.path.join(save_path, f'two_line_{args.loss}_bco{args.bcecoefficient}_{name_label}_batch{batch_id}.png'))
    plt.close('all')


def two_bar_plot_grad_flow(args, triplet_np, bce_np, name_label, batch_id, epoch, save_path):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

    colors = {'average_bce': 'cyan',
              'max_bce': 'blue',
              'average_triplet': 'hotpink',
              'max_triplet': 'red'}

    dict_avg_grads = {}

    plt.figure(figsize=(8, 6))
    df = None
    # for (label, named_parameters) in [('bce', bce_np), ('triplet', triplet_np)]:
    #     # plt.figure(figsize=(64, 48))
    #     ave_grads = []
    #     max_grads = []
    #     layers = []
    #     for n, p in named_parameters.items():
    #         if (p.requires_grad) and ("bias" not in n):
    #             if n == 'ft_net.fc.weight':
    #                 continue
    #             if p.grad is None:
    #                 print(f'{label} {n} none grad!!! *********')
    #                 continue
    #             ave_grads.append(p.grad.abs().mean())
    #             max_grads.append(p.grad.abs().max())
    #             layers.append(n)

    for label, lists in [('bce', bce_np), ('triplet', triplet_np)]:
        ave_grads = lists[0]
        max_grads = lists[1]
        layers = lists[2]

        if df is None:
            df = pd.DataFrame(index=layers)

        df[f'average_{label}'] = ave_grads
        df[f'max_{label}'] = max_grads
        # dict_avg_grads[label] = max_grads
        # plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.05, lw=1, color=colors[f'average_{label}'])
        # plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.05, lw=1, color=colors[f'max_{label}'])

    # plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    # # plt.ylim(bottom=-0.001, top=np.mean([np.mean(dict_avg_grads['bce']), np.mean(dict_avg_grads['triplet'])]))  # zoom in on the lower gradient regions
    # plt.xticks(range(0, len(ave_grads), 1), layers)
    # plt.xlim(left=0, right=len(ave_grads))
    # # plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    # plt.xlabel("Layers")
    # plt.ylabel("average gradient")
    # plt.title(f"Gradient flow for  {name_label}_epoch{epoch}_batch{batch_id}")
    # plt.grid(True)
    # plt.legend([Line2D([0], [0], color=colors[f'max_bce'], lw=4),
    #             Line2D([0], [0], color=colors[f'average_bce'], lw=4),
    #             Line2D([0], [0], color=colors[f'max_triplet'], lw=4),
    #             Line2D([0], [0], color=colors[f'average_triplet'], lw=4),
    #             Line2D([0], [0], color="k", lw=4)],
    #            ['bce-max-gradient', 'bce-mean-gradient',
    #             'triplet-max-gradient', 'triplet-mean-gradient', 'zero-gradient'])
    # plt.savefig(os.path.join(save_path, f'two_bars_{args.loss}_bco{args.bcecoefficient}_{label}_batch{batch_id}.png'))
    # plt.show()
    # plt.close()

    # plt.figure(figsize=(32, 24))
    plt.rcParams.update({'font.size': 15})  # must set in top
    df = df.applymap(lambda x: x.item())
    df.plot(kind='bar', colormap='plasma', figsize=(64, 48), alpha=0.5)
    plt.ylim(bottom=0, top=0.02 * args.bcecoefficient)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title(f"Gradient flow for  {name_label}_epoch{epoch}_batch{batch_id}")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'two_bars_{args.loss}_bco{args.bcecoefficient}_{label}_batch{batch_id}.png'))

    plt.close('all')


def __post_create_heatmap(heatmap, shape):
    # draw the heatmap
    plt.matshow(heatmap.squeeze())
    # plt.savefig(f'cam_{id}.png')

    # import pdb
    # pdb.set_trace()

    heatmap = cv2.resize(np.float32(heatmap), shape)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap

def activation_reduction(activations, method='avg'): # avg or max
    if method == 'avg':
        act = torch.mean(activations, dim=1).squeeze().data.cpu().numpy()
    elif method == 'max':
        act = torch.max(activations, dim=1)[0].squeeze().data.cpu().numpy()
    else:
        raise Exception(f'Heatmap method {method} not defined')

    return act


@MY_DEC
def get_heatmap(activations, shape, save_path=None, label=None, method='avg'):
    # heatmap = torch.mean(activations, dim=1).squeeze()
    # heatmap = torch.max(activations, dim=1)[0].squeeze()

    heatmap = activation_reduction(activations, method=method)

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf


    max_value = np.max(np.abs(heatmap))  # to keep it normalized between positive and negative

    heatmap = np.maximum(heatmap, 0)
    # abs_heatmap = np.abs(heatmap)

    # normalize the heatmap
    if max_value != 0:
        heatmap /= max_value

    heatmap = __post_create_heatmap(heatmap, shape)
    plt.close('all')

    return heatmap


def get_heatmaps(activations, shape, save_path=None, label=None, normalize=[], method='avg', classifier_weights=None):
    # activations = np.array(list(map(lambda act: torch.mean(act, dim=1).squeeze().data.cpu().numpy(),
    #                                 activations)))

    # activations = np.array(list(map(lambda act: torch.mean(act, dim=1).squeeze().data.cpu().numpy(),
    #                                 activations)))
    cpu_activations = []
    if classifier_weights is not None:
        for activation in activations:
            cpu_activations.append(activation * classifier_weights)
    else:
        for activation in activations:
            cpu_activations.append(activation)

    method_list = [method for _ in range(len(cpu_activations))]
    cpu_activations = np.array(list(map(activation_reduction,
                                    cpu_activations, method_list)))

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf

    # heatmap = heatmap
    final_heatmaps = []

    # import pdb
    # pdb.set_trace()

    heatmaps = np.maximum(cpu_activations, 0)
    # activations[0][0] *= -1
    # heatmaps = activations

    # abs_heatmap = np.abs(activations)

    # normalize the heatmap
    print(f'heatmaps max: {np.max(heatmaps)}')

    if np.max(heatmaps) != 0:
        heatmaps = heatmaps / np.max(heatmaps)

    for heatmap in heatmaps:
        final_heatmaps.append(__post_create_heatmap(heatmap, shape))

    return final_heatmaps


def merge_heatmap_img(img, heatmap):
    pic = img.copy()
    # print('img shape:', temp.shape)
    # print('heatmap shape:', heatmap.shape)
    cv2.addWeighted(heatmap, 0.4, img, 0.6, 0, pic)
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)

    return pic


def read_img_paths(path, local_path='.'):
    import re
    final_lines = []
    with open(path, 'r') as file:
        cam_path = file.readline().strip()
        lines = list(map(lambda x:
                         tuple(map(lambda y:
                                   os.path.join(local_path, cam_path, y) if len(y) != 0 else '',
                                   re.split(" +", x.strip()))),
                         file.readlines()))

        for l in lines:
            if len(l[0]) != 0:
                # splits = l.split('/')
                # file_name = splits[len(splits) - 1].split('.')[0]
                # final_lines.append((l, file_name))
                final_lines.append(l)

    return final_lines


def vector_merge_function(v1, v2, method='sim', normalize=True):
    if method == 'diff':
        ret = torch.pow((v1 - v2), 2)
        if normalize:
            ret = F.normalize(ret, p=2, dim=1)
        # ret = torch.div(ret, ret.max(dim=1, keepdim=True)[0])
        # return torch.div(ret, torch.sqrt(torch.sum(torch.pow(ret, 2))))
        return ret
    elif method == 'sim':

        if normalize:
            ret = F.normalize(v1, p=2, dim=1) * F.normalize(v2, p=2, dim=1)
        else:
            ret = v1 * v2
        # ret = torch.div(ret, ret.max(dim=1, keepdim=True)[0])
        # ret = F.normalize(ret, p=2, dim=1)
        return ret
        # return torch.div(ret, torch.sqrt(torch.sum(torch.pow(ret, 2))))
    elif method == 'diff-sim':
        diff_merged = torch.pow((v1 - v2), 2)
        if normalize:
            diff_merged = F.normalize(diff_merged, p=2, dim=1)
            sim_merged = F.normalize(v1, p=2, dim=1) * F.normalize(v2, p=2, dim=1)
        else:
            sim_merged = v1 * v2
        # diff_merged = torch.div(diff_merged, diff_merged.max(dim=1, keepdim=True)[0])
        # sim_merged = torch.div(sim_merged, sim_merged.max(dim=1, keepdim=True)[0])

        # diff_merged = F.normalize(diff_merged, p=2, dim=1)
        # sim_merged = F.normalize(sim_merged, p=2, dim=1)
        # ret1 = torch.div(diff_merged, torch.sqrt(torch.sum(torch.pow(diff_merged, 2))))
        # ret2 = torch.div(sim_merged, torch.sqrt(torch.sum(torch.pow(sim_merged, 2))))
        #
        # ret1 = torch.nn.BatchNorm1d(diff_merged)
        # ret2 = torch.nn.BatchNorm1d(sim_merged)

        return torch.cat([diff_merged, sim_merged], dim=1)
    else:
        raise Exception(f'Merge method {method} not implemented.')


@MY_DEC
def add_mask(org_img, mask, offsets=None, resize_factors=None, colored=False):
    img = org_img.copy()

    angle = np.random.uniform(0, 360)
    mask = Image.Image.rotate(mask, angle=angle, expand=True)

    img_shape = img.size
    mask_shape = mask.size

    if resize_factors is None:
        x_factor = img_shape[0] / mask_shape[0]
        y_factor = img_shape[1] / mask_shape[1]

        while x_factor <= 1 or y_factor <= 1:
            print(f'!!!MASK WAS BIGGER!!!! img_size = {img_shape}, mask_size = {mask_shape}')
            mask = mask.resize((mask_shape[0] // 2, mask_shape[1] // 2))
            mask_shape = mask.size
            x_factor = img_shape[0] / mask_shape[0]
            y_factor = img_shape[1] / mask_shape[1]

        x_resize_factor = np.random.uniform(1, x_factor)
        y_resize_factor = np.random.uniform(1, y_factor)
    else:
        x_resize_factor, y_resize_factor = resize_factors

    mask = mask.resize((int(x_resize_factor * mask_shape[0]),
                        int(y_resize_factor * mask_shape[1])))

    mask_np = np.array(mask)

    # import pdb
    # pdb.set_trace()
    # mask_np[mask_np > 0] = 255
    if colored:
        random_mask_color = np.random.randint(0, 256, mask_np[np.where(mask_np[:, :, 3] > 0)].shape)
        random_mask_color[:, 3] = 255
        mask_np[np.where(mask_np[:, :, 3] > 0)] = random_mask_color
    else:
        mask_np[mask_np > 0] = 255

    # mask_np[mask_np > 0] = np.random.randint(0, 256, len(mask_np[mask_np > 0]))
    mask = Image.fromarray(mask_np)

    mask_shape = mask.size

    if offsets is None:
        if mask_shape[0] < img_shape[0]:
            x_offset = np.random.randint(0, img_shape[0] - mask_shape[0])
        else:
            x_offset = np.random.randint(0, img_shape[0] // 2)

        if mask_shape[1] < img_shape[1]:
            y_offset = np.random.randint(0, img_shape[1] - mask_shape[1])
        else:
            y_offset = np.random.randint(0, img_shape[1] // 2)
    else:
        x_offset, y_offset = offsets
    # x_offset = np.random.randint(img_shape[0]//5, 3 * img_shape[0]//5)
    # y_offset = np.random.randint(img_shape[1]//5, 3 * img_shape[1]//5)

    padding = (x_offset,  # left
               y_offset,  # top
               img_shape[0] - (x_offset + mask_shape[0]),  # right
               img_shape[1] - (y_offset + mask_shape[1]))  # bottom

    mask = ImageOps.expand(mask, padding)
    img.paste(mask, (0, 0), mask=mask)
    assert mask.size == img.size

    # plt.imshow(img)
    # plt.show()
    # # import pdb
    # pdb.set_trace()


    four_channel_img = Image.fromarray(np.dstack((np.asarray(img), np.asarray(mask)[:, :, 3] // 255)))

    return four_channel_img, img, mask, {'offsets': (x_offset, y_offset),
                                         'resize_factors': (x_resize_factor, y_resize_factor)}


def get_masks(data_path, data_setname, mask_path):
    masks = np.array(pd.read_csv(mask_path).mask_path)

    masks = np.array(list(map(lambda x: os.path.join(data_path, data_setname, x), masks)))

    return masks
    # print(f'angle = {angle}')
    # print(f'x_offset = {x_offset}')
    # print(f'y_offset = {y_offset}')
    # print(f'x_resize_factor = {x_resize_factor}')
    # print(f'y_resize_factor = {y_resize_factor}')
    # print(f'mask shape: {(mask_shape[0], mask_shape[1])}')


def create_subplot(ax, label, img):
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(label)


def draw_act_histograms(ax, acts, titles, plot_title, classifier_weights=None):
    # plt.rcParams.update({'font.size': 5})
    # fig = plt.Figure(figsize=(20, 20))

    new_acts = []

    if classifier_weights is not None:
        for act in acts:
            new_acts.append(act * classifier_weights)
    else:
        for act in acts:
            new_acts.append(act)

    new_acts = list(map(lambda x: x.cpu().numpy(), new_acts))

    legends = list(map(lambda x: x + ' value distribution', titles))
    if len(new_acts) == 1:
        colors = ['b']
        lines = [Line2D([0], [0], color="b", lw=4)]
    elif len(new_acts) == 2:
        colors = ['b', 'r']
        lines = [Line2D([0], [0], color="b", lw=4),
                 Line2D([0], [0], color="r", lw=4)]
    elif len(new_acts) == 3:
        colors = ['b', 'r', 'g']
        lines = [Line2D([0], [0], color="b", lw=4),
                 Line2D([0], [0], color="r", lw=4),
                 Line2D([0], [0], color="g", lw=4)]

    max = 0
    for act, title, color in zip(new_acts, titles, colors):
        flatten_act = act.flatten()
        if max < flatten_act.max():
            max = flatten_act.max()
        ax.hist(flatten_act, bins=100, alpha=0.4, color=color)

    ax.axis('on')
    ax.set_xlim(left=-0.1, right=max + 1)
    ax.legend(lines, legends)
    ax.set_title(plot_title)


@MY_DEC
def apply_grad_heatmaps(grads, activations, img_dict, label, id, path, plot_title):
    pooled_gradients = torch.mean(grads, dim=[0, 2, 3])

    for i in range(len(pooled_gradients)):
        activations[:, i, :, :] *= pooled_gradients[i]

    anch_org = img_dict['anch']

    for method in ['avg', 'max']:
        heatmap = get_heatmap(activations, shape=(anch_org.shape[0], anch_org.shape[1]), method=method)
        heatmap_negative = get_heatmap(-1 * activations, shape=(anch_org.shape[0], anch_org.shape[1]), method=method)

        pos_pics = []
        neg_pics = []

        paths = []
        titles = []

        for l, i in img_dict.items():
            pos_pics.append(merge_heatmap_img(i, heatmap))
            neg_pics.append(merge_heatmap_img(i, heatmap_negative))
            titles.append(l)

        path_ = os.path.join(path, f'{method}_backward_triplet{id}_{label}.png')
        plt.rcParams.update({'font.size': 19})
        fig, axes = plt.subplots(2, len(pos_pics), figsize=(len(pos_pics) * 10, 20))

        for ax, pic, title in zip(axes[0], pos_pics, titles):
            ax.imshow(pic)
            ax.axis('off')
            ax.set_title(title + ' POSITIVE')

        for ax, pic, title in zip(axes[1], neg_pics, titles):
            ax.imshow(pic)
            ax.axis('off')
            ax.set_title(title + ' NEGATIVE')

        fig.suptitle(plot_title + '\n(' + method + ')')
        plt.savefig(path_)
        plt.close('all')

    # for pic, path in zip(pics, paths):
    #     cv2.imwrite(path, pic)


@MY_DEC
def apply_forward_heatmap(acts, img_list, id, heatmap_path, overall_title,
                          titles=[''], histogram_path='',
                          merge_method='sim', classifier_weights=None):
    """

    :param acts: [anch_activation_p,
                    pos_activation,
                    anch_activation_n,
                    neg_activation]
    :param img_list: [(anch_lbl, anch_img),
                        (pos_lbl, pos_img),
                        (neg_lbl, neg_img)]
    :param id:
    :param heatmap_path:
    :param individual_paths:
    :param pair_paths:
    :return:
    """

    # classifier_weights = classifier_weights.cpu().numpy()
    shape = img_list[0][1].shape[0:2]

    acts.append(vector_merge_function(acts[0], acts[1], method=merge_method))  # anch_pos_subtraction
    titles.append(f'anch_pos_{merge_method}')

    acts.append(vector_merge_function(acts[0], acts[2], method=merge_method))  # anch_neg_subtraction
    titles.append(f'anch_neg_{merge_method}')
    # print('#################################################################################################\n' + overall_title)
    # print(f'anch_{merge_method} median: {acts[0].median()}')
    # print(f'anch_{merge_method} max: {acts[0].max()}')
    # print(f'anch_{merge_method} min: {acts[0].min()}')
    #
    # print(f'pos_{merge_method} median: {acts[1].median()}')
    # print(f'pos_{merge_method} max: {acts[1].max()}')
    # print(f'pos_{merge_method} min: {acts[1].min()}')
    #
    # print(f'neg_{merge_method} median: {acts[2].median()}')
    # print(f'neg_{merge_method} max: {acts[2].max()}')
    # print(f'neg_{merge_method} min: {acts[2].min()}')
    #
    # print(f'pos_anch_{merge_method} median: {acts[3].median()}')
    # print(f'neg_anch_{merge_method} median: {acts[4].median()}')
    #
    # print(f'pos_anch_{merge_method} max: {acts[3].max()}')
    # print(f'neg_anch_{merge_method} max: {acts[4].max()}')
    #
    # print(f'pos_anch_{merge_method} min: {acts[3].min()}')
    # print(f'neg_anch_{merge_method} min: {acts[4].min()}')
    #
    # import pdb
    # pdb.set_trace()

    plt.rcParams.update({'font.size': 25})

    fig, axes = plt.subplots(1, 3, figsize=(55, 18))
    draw_act_histograms(axes[0], [classifier_weights], ['weights'],
                        f'Classifier weights {merge_method}', classifier_weights=None)
    draw_act_histograms(axes[1], acts[0:3], titles[0:3],
                        'Heatmaps', classifier_weights=classifier_weights)
    draw_act_histograms(axes[2], acts[3:5], titles[3:5],
                        f'Heatmap {merge_method}s', classifier_weights=classifier_weights)

    fig.suptitle(overall_title)

    plt.savefig(histogram_path)
    plt.close('all')
    # import pdb
    # pdb.set_trace()
    for method in ['avg', 'max']:
        heatmaps = get_heatmaps(acts[:3], shape=shape, method=method,
                                classifier_weights=None)  # seperated for normalization, heatmaps withOUT classifier weights
        heatmaps.extend(get_heatmaps(acts[:3], shape=shape, method=method,
                                classifier_weights=classifier_weights)) # seperated for normalization, heatmaps WITH classifier weights
        heatmaps.extend(get_heatmaps(acts[3:5], shape=shape, method=method, classifier_weights=None))
        heatmaps.extend(get_heatmaps(acts[3:5], shape=shape, method=method, classifier_weights=classifier_weights))

        # path_ = os.path.join(path, f'cam_{id}_{l}.png')
        # merge_heatmap_img(i, heatmap, path=path_)

        pics = [merge_heatmap_img(img_list[0][1], heatmaps[0]),
                merge_heatmap_img(img_list[1][1], heatmaps[1]),
                merge_heatmap_img(img_list[2][1], heatmaps[2]),

                merge_heatmap_img(img_list[0][1], heatmaps[3]),
                merge_heatmap_img(img_list[1][1], heatmaps[4]),
                merge_heatmap_img(img_list[2][1], heatmaps[5]),

                merge_heatmap_img(img_list[0][1], heatmaps[6]),
                merge_heatmap_img(img_list[1][1], heatmaps[6]),
                merge_heatmap_img(img_list[0][1], heatmaps[7]),
                merge_heatmap_img(img_list[2][1], heatmaps[7]),

                merge_heatmap_img(img_list[0][1], heatmaps[8]),
                merge_heatmap_img(img_list[1][1], heatmaps[8]),
                merge_heatmap_img(img_list[0][1], heatmaps[9]),
                merge_heatmap_img(img_list[2][1], heatmaps[9])
                ]

        plt.rcParams.update({'font.size': 10})
        # plt.rcParams.update({'figure.figsize': (10, 10)})

        subplot_grid_shape = (13, 12)
        fig = plt.figure(figsize=(10, 10))
        ax_anch = plt.subplot2grid(subplot_grid_shape, (1, 0), colspan=3, rowspan=3)
        ax_pos = plt.subplot2grid(subplot_grid_shape, (5, 0), colspan=3, rowspan=3)
        ax_neg = plt.subplot2grid(subplot_grid_shape, (9, 0), colspan=3, rowspan=3)

        ax_anch_ww = plt.subplot2grid(subplot_grid_shape, (1, 3), colspan=3, rowspan=3)
        ax_pos_ww = plt.subplot2grid(subplot_grid_shape, (5, 3), colspan=3, rowspan=3)
        ax_neg_ww = plt.subplot2grid(subplot_grid_shape, (9, 3), colspan=3, rowspan=3)

        ax_anchpos_anch = plt.subplot2grid(subplot_grid_shape, (0, 6), rowspan=3, colspan=3)
        ax_anchneg_anch = plt.subplot2grid(subplot_grid_shape, (7, 6), rowspan=3, colspan=3)
        ax_anchpos_pos = plt.subplot2grid(subplot_grid_shape, (0, 9), rowspan=3, colspan=3)
        ax_anchneg_neg = plt.subplot2grid(subplot_grid_shape, (7, 9), rowspan=3, colspan=3)

        ax_anchpos_anch_ww = plt.subplot2grid(subplot_grid_shape, (3, 6), rowspan=3, colspan=3)
        ax_anchneg_anch_ww = plt.subplot2grid(subplot_grid_shape, (10, 6), rowspan=3, colspan=3)
        ax_anchpos_pos_ww = plt.subplot2grid(subplot_grid_shape, (3, 9), rowspan=3, colspan=3)
        ax_anchneg_neg_ww = plt.subplot2grid(subplot_grid_shape, (10, 9), rowspan=3, colspan=3)

        create_subplot(ax_anch, titles[0] + ' org', pics[0])
        create_subplot(ax_pos, titles[1] + ' org', pics[1])
        create_subplot(ax_neg, titles[2] + ' org', pics[2])

        create_subplot(ax_anch_ww, titles[0], pics[3])
        create_subplot(ax_pos_ww, titles[1], pics[4])
        create_subplot(ax_neg_ww, titles[2], pics[5])

        create_subplot(ax_anchpos_anch, titles[3] + ' org', pics[6])
        create_subplot(ax_anchpos_pos, titles[3] + ' org', pics[7])
        create_subplot(ax_anchneg_anch, titles[4] + ' org', pics[8])
        create_subplot(ax_anchneg_neg, titles[4] + ' org', pics[9])


        create_subplot(ax_anchpos_anch_ww, titles[3], pics[10])
        create_subplot(ax_anchpos_pos_ww, titles[3], pics[11])
        create_subplot(ax_anchneg_anch_ww, titles[4], pics[12])
        create_subplot(ax_anchneg_neg_ww, titles[4], pics[13])

        fig.suptitle(overall_title + '\n(' + method + ')')

        plt.savefig(heatmap_path[method])
        plt.close('all')

    # for pic, title in zip(pics, titles):
    #
    #     new_pics.append(self.put_text_on_pic(pic, title))

    # for idx, (l, i) in enumerate(img_list):
    #     pics.append(merge_heatmap_img(i, heatmaps[idx], path=individual_paths[idx]))

    # titles = ['anch', 'pos', 'subtraction']
    # new_pics = []
    # for pic, l in zip(pics, titles):
    #     new_pics.append(self.put_text_on_pic(pic, l))

    # row1 = np.concatenate([new_pics[0], new_pics[1], new_pics[4]], axis=1)
    # row2 = np.concatenate([new_pics[2], new_pics[3], new_pics[5]], axis=1)
    #
    # pic = np.concatenate([row1, row2], axis=0)

    # cv2.imshow('nahji', pic)

    # dist_heatmap = torch.pow((heatmaps[0] - heatmaps[1]), 2)
    # import pdb
    # pdb.set_trace()

    # dist_heatmap = vector_merge_function(heatmaps[0], heatmaps[1])
    # dist_heatmap = np.power(heatmaps[0] - heatmaps[1], 2)
    # pics = []
    # for idx, (l, i) in enumerate(img_list):
    #     pics.append(merge_heatmap_img(i, heatmaps[2], path=pair_paths[idx]))
    #
    # titles = ['anch', 'pos', 'subtraction']
    # new_pics = []
    # for pic, l in zip(pics, titles):
    #     new_pics.append(self.put_text_on_pic(pic, l))
    #
    # pics = np.concatenate(new_pics, axis=1)
    # cv2.imwrite(path, pic)


@MY_DEC
def get_euc_distances(img_feats, img_classes):
    dists = euclidean_distances(img_feats)
    diff_average_dist = np.zeros_like(dists[0])
    diff_min_dist = np.zeros_like(dists[0])
    diff_max_dist = np.zeros_like(dists[0])

    same_average_dist = np.zeros_like(dists[0])
    same_max_dist = np.zeros_like(dists[0])
    same_min_dist = np.zeros_like(dists[0])

    for idx, (row, label) in enumerate(zip(dists, img_classes)):
        row = np.delete(row, idx)
        img_classes_temp = np.delete(img_classes, idx)

        diff_class_dists = row[img_classes_temp != label]
        same_class_dists = row[img_classes_temp == label]

        diff_average_dist[idx] = diff_class_dists.mean()
        diff_min_dist[idx] = diff_class_dists.min()
        diff_max_dist[idx] = diff_class_dists.max()

        if len(same_class_dists) == 0:
            same_average_dist[idx] = 0
            same_min_dist[idx] = 0
            same_max_dist[idx] = 0
        else:
            same_average_dist[idx] = same_class_dists.mean()
            same_max_dist[idx] = same_class_dists.max()
            same_min_dist[idx] = same_class_dists.min()

    diff_average_dist_mean = diff_average_dist.mean()
    diff_min_dist_mean = diff_min_dist.mean()
    diff_max_dist_mean = diff_max_dist.mean()
    same_average_dist_mean = same_average_dist.mean()
    same_max_dist_mean = same_max_dist.mean()
    same_min_dist_mean = same_min_dist.mean()

    return {'between_class_average': diff_average_dist_mean,
            'between_class_min': diff_min_dist_mean,
            'between_class_max': diff_max_dist_mean,
            'in_class_average': same_average_dist_mean,
            'in_class_min': same_min_dist_mean,
            'in_class_max': same_max_dist_mean}


def draw_all_heatmaps(actss, imgs, subplot_titles, path, supplot_title):
    plt.rcParams.update({'font.size': 5})
    fig, axes = plt.subplots(1, 3)
    for acts, img, plot_title, ax in zip(actss, imgs, subplot_titles, axes):
        acts = acts.cpu().numpy()
        # acts = np.maximum(acts, 0)
        # plt.rcParams.update({'figure.figsize': (20, 10)})
        print(f'Begin drawing all activations for {plot_title}')

        acts_pos = np.maximum(acts, 0) # todo fucking it up
        acts_pos /= np.max(acts_pos)

        # acts = acts[0, 0:4, :, :]

        rows = []
        all = []
        row = []
        channel_length = len(acts_pos.squeeze(axis=0))

        row_length = 1
        all_length_power = 0

        while np.power(2, all_length_power) <= channel_length:
            if (all_length_power) % 2 == 0:
                row_length = np.power(2, int((all_length_power + 1) / 2))
            all_length_power += 1

        for i, act in enumerate(acts_pos.squeeze(axis=0)):

            heatmap = __post_create_heatmap(act, (img.shape[0], img.shape[1]))
            pic = merge_heatmap_img(img, heatmap)
            row.append(pic)
            if (i + 1) % row_length == 0:
                rows.append(row)
                row = []

        for row in rows:
            all.append(np.concatenate(row, axis=1))

        all = np.concatenate(all, axis=0)

        print(all.shape)
        ax.imshow(all)
        ax.set_title(plot_title)
        ax.axis('off')
        print(f'saving... {plot_title}')

        # plt.show()
    fig.suptitle(supplot_title)
    fig.savefig(path, dpi=5000)
    plt.close('all')


def get_logname(args, model):
    id_str = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')
    id_str = '-time_' + id_str.replace('.', '-')

    if args.job_id != '':
        id_str += '_' + args.job_id

    # name = 'model-betteraug-distmlp-' + self.model

    name = f'model-bs{args.batch_size}'

    if args.cuda and args.gpu_ids != '':
        gpus_num = len(args.gpu_ids.split(','))
        name += f'-{gpus_num}gpus'
    else:
        name += f'-cpu'

    if args.extra_name != '':
        name += '-' + args.extra_name

    name_replace_dict = {'dataset_name': 'dsn',
                         'batch_size': 'bs',
                         'lr_siamese': 'lrs',
                         'lr_resnet': 'lrr',
                         # 'early_stopping',
                         'feat_extractor': 'fe',
                         'extra_layer': 'el',
                         # 'normalize',
                         'number_of_runs': 'nor',
                         'no_negative': 'nn',
                         'margin': 'm',
                         'loss': 'loss',
                         'overfit_num': 'on',
                         'bcecoefficient': 'bco',
                         'debug_grad': 'dg',
                         'aug_mask': 'am',
                         'colored_mask': 'colmask',
                         'from_scratch': 'fs',
                         'fourth_dim': 'fd',
                         'image_size': 'igsz',
                         'pooling': 'pool',
                         'merge_method': 'mm'}

    important_args = ['dataset_name',
                      'batch_size',
                      'lr_siamese',
                      'lr_resnet',
                      # 'early_stopping',
                      'feat_extractor',
                      'extra_layer',
                      # 'normalize',
                      'number_of_runs',
                      'no_negative',
                      'loss',
                      'overfit_num',
                      'debug_grad',
                      'aug_mask',
                      'from_scratch',
                      'image_size',
                      'fourth_dim',
                      'pooling',
                      'merge_method',
                      'colored_mask']

    if args.loss != 'bce':
        important_args.extend(['bcecoefficient',
                               'margin'])

    for arg in vars(args):
        if str(arg) in important_args:
            if str(arg) == 'debug_grad' and not getattr(args, arg):
                continue
            elif str(arg) == 'overfit_num' and getattr(args, arg) == 0:
                continue
            elif str(arg) == 'aug_mask' and not getattr(args, arg):
                continue
            elif str(arg) == 'from_scratch' and not getattr(args, arg):
                continue
            elif str(arg) == 'fourth_dim' and not getattr(args, arg):
                continue
            elif str(arg) == 'colored_mask' and not getattr(args, arg):
                continue

            if type(getattr(args, arg)) is not bool:
                name += '-' + name_replace_dict[str(arg)] + '_' + str(getattr(args, arg))
            else:
                name += '-' + name_replace_dict[str(arg)]

    if args.pretrained_model_dir != '':
        name = args.pretrained_model_dir + '_pretrained'

    name += id_str

    return name, id_str


def print_gpu_stuff(cuda, state):
    if cuda:
        print('****************************')
        print(f'GPU: {torch.cuda.current_device()}')
        print(f'Total memory {state}: {torch.cuda.get_device_properties(0).total_memory / (2 ** 30)} GB')
        print(f'current memory allocated {state}: ', torch.cuda.memory_allocated() / (2 ** 30), ' GB')
        print(f'current memory cached {state}: ', torch.cuda.memory_cached() / (2 ** 30), ' GB')
        print(f'current cached free memory {state}: ',
              (torch.cuda.memory_cached() - torch.cuda.memory_allocated()) / (2 ** 30), ' GB')

    # pdb.set_trace()


def mac(x):
    return F.max_pool2d(x, (x.size(-2), x.size(-1)))


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative


def rmac(x, L=3, eps=1e-6):
    ovr = 0.4  # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7])  # possible regions for the long dimension

    W = x.size(3)
    H = x.size(2)

    w = min(W, H)
    w2 = math.floor(w / 2.0 - 1)

    b = (max(H, W) - w) / (steps - 1)
    (tmp, idx) = torch.min(torch.abs(((w ** 2 - w * b) / w ** 2) - ovr), 0)  # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0
    Hd = 0
    if H < W:
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1

    v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
    v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)

    for l in range(1, L + 1):
        wl = math.floor(2 * w / (l + 1))
        wl2 = math.floor(wl / 2 - 1)

        if l + Wd == 1:
            b = 0
        else:
            b = (W - wl) / (l + Wd - 1)
        cenW = torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b) - wl2  # center coordinates
        if l + Hd == 1:
            b = 0
        else:
            b = (H - wl) / (l + Hd - 1)
        cenH = torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b) - wl2  # center coordinates

        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                R = x[:, :, (int(i_) + torch.Tensor(range(wl)).long()).tolist(), :]
                R = R[:, :, :, (int(j_) + torch.Tensor(range(wl)).long()).tolist()]
                vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt)
                v += vt

    return v


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def plot_pred_hist(pos_preds, neg_preds, bins=100, title='Pred Histogram', savepath='pred_dist'):
    plt.figure(figsize=(10, 10))
    plt.hist(sigmoid(pos_preds), alpha=0.5, bins=bins, color='g')
    plt.hist(sigmoid(neg_preds), alpha=0.5, bins=bins, color='r')


    lines = [Line2D([0], [0], color="g", lw=4),
             Line2D([0], [0], color="r", lw=4)]

    plt.legend(lines, ['Positive', 'Negative'])
    plt.title(title)

    plt.savefig(savepath)
    plt.close('all')
