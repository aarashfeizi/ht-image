import argparse
import datetime
import json
import logging
import math
import multiprocessing
import os
import sys
import time
import pickle

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

import faiss

import metrics

matplotlib.rc('font', size=24)

MERGE_METHODS = ['sim', 'diff', 'diff-sim', 'diff-sim-con',
                 'concat', 'diff-sim-con-att', 'concat-mid',
                 'diff-sim-con-complete', 'diff-sim-con-att-add',
                 'local-attention', 'local-ds-attention', 'local-diff-sim-concat-unequaldim',
                 'local-diff-sim-add-unequaldim', 'local-diff-sim-mult-unequaldim',
                 'channel-attention']

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
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4),
                 scale=[0.5, 1.0]):
        # hotels v5 train small mean: tensor([0.5791, 0.5231, 0.4664])
        # hotels v5 train small std: tensor([0.2512, 0.2581, 0.2698])

        # hotels v5 train mean: tensor([0.5805, 0.5247, 0.4683])
        # hotels v5 train std: tensor([0.2508, 0.2580, 0.2701])

        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
        self.rotate = rotate
        self.normalize = transforms.Normalize(**self.normalize_param)
        self.scale = scale
        self.random_erase_prob = 0.0

    def parse_transform(self, transform_type):
        # if transform_type == 'ImageJitter':
        #     method = add_transforms.ImageJitter(self.jitter_param)
        #     return method
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomResizedCrop':
            return method(self.image_size, scale=self.scale, ratio=[1.0, 1.0])
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Resize':
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        elif transform_type == 'RandomRotation':
            return method(self.rotate)
        elif transform_type == 'ColorJitter':
            return method(brightness=0.5, hue=0.5, contrast=0.5, saturation=0.5)
        elif transform_type == 'RandomErasing':
            return method(p=self.random_erase_prob, scale=(0.1, 0.75), ratio=(0.3, 3.3)) # TODO RANDOM ERASE!!!

        else:
            return method()

    def get_composed_transform(self, aug=False,
                               random_crop=False,
                               for_network=True,
                               color_jitter=False,
                               random_erase=0.0):
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

        if color_jitter:
            transform_list.extend(['ColorJitter'])

        if for_network:
            transform_list.extend(['ToTensor'])
            if random_erase > 0.0:
                self.random_erase_prob = random_erase
                transform_list.extend(['RandomErasing'])


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

@MY_DEC
def get_logger(logname, env):
    if env == 'hlr' or env == 'local':
        logging.basicConfig(filename=os.path.join('logs', logname + '.log'),
                            filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)
    else:
        logging.basicConfig(stream=sys.stdout,
                            filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)
    return logging.getLogger()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cuda', '--cuda', default=False, action='store_true')
    parser.add_argument('-gpu', '--gpu_ids', default='', help="gpu ids used to train")  # before: default="0,1,2,3"
    parser.add_argument('-env', '--env', default='local',
                        help="where the code is being run, e.g. local, beluga, graham")  # before: default="0,1,2,3"
    parser.add_argument('-on', '--overfit_num', default=0, type=int)
    parser.add_argument('-dsn', '--dataset_name', default='hotels',
                        choices=['omniglot', 'cub', 'cub_eval', 'hotels', 'new_hotels_small', 'new_hotels', 'hotels_dummy', 'cars', 'cars_eval', 'sop'])
    parser.add_argument('-dsp', '--dataset_path', default='')
    parser.add_argument('-por', '--portion', default=0, type=int)
    parser.add_argument('-ls', '--limit_samples', default=0, type=int, help="Limit samples per class for val and test")
    parser.add_argument('-nor', '--number_of_runs', default=1, type=int, help="Number of times to sample for k@n")
    parser.add_argument('-roc_num', '--roc_num', default=1, type=int,
                        help="Multiply number of pairs chosen by a coefficient")

    parser.add_argument('-sp', '--save_path', default='savedmodels/', help="path to store model")
    parser.add_argument('-np', '--negative_path', default='', help="path to store best negative "
                                                                   "images, should be a "
                                                                   "dictionary that maps each "
                                                                   "image paths to its best "
                                                                   "negative image paths")  # 'negatives/negatives.pkl'
    parser.add_argument('-lp', '--log_path', default='logs/', help="path to log")
    parser.add_argument('-tbp', '--tb_path', default='tensorboard/', help="path for tensorboard")
    parser.add_argument('-a', '--aug', default=False, action='store_true')
    # parser.add_argument('-m', '--mask', default=False, action='store_true')
    parser.add_argument('-r', '--rotate', default=0.0, type=float)
    parser.add_argument('-random_erase', '--random_erase', default=0.0, type=float)

    parser.add_argument('-mn', '--pretrained_model_name', default='')
    parser.add_argument('-pmd', '--pretrained_model_dir', default='')
    parser.add_argument('-ev', '--eval_mode', default='fewshot', choices=['fewshot', 'simple'])
    parser.add_argument('-fe', '--feat_extractor', default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'vgg16', 'deit16_224', 'deit16_small_224'])
    parser.add_argument('--pretrained_model', default='', choices=['swav', 'simclr', 'byol', 'dino'])
    parser.add_argument('-pool', '--pooling', default='spoc',
                        choices=['spoc', 'gem', 'mac', 'rmac'])
    parser.add_argument('-fr', '--freeze_ext', default=False, action='store_true')
    parser.add_argument('-cl', '--classifier_layer', default=0, type=int,
                        help="Number of layers for classifier")
    parser.add_argument('-pl', '--projection_layer', default=0, type=int,
                        help="Number of layers for projection after feature extractor")

    parser.add_argument('-nn', '--no_negative', default=1, type=int)
    parser.add_argument('-en', '--extra_name', default='')

    parser.add_argument('-s', '--seed', default=402, type=int, help="random seed")
    parser.add_argument('-w', '--way', default=20, type=int, help="how much way one-shot learning")
    parser.add_argument('-tk', '--test_k', default=4, type=int, help="how many images per class for teseting")

    parser.add_argument('-t', '--times', default=1000, type=int, help="number of samples to test accuracy")
    parser.add_argument('-wr', '--workers', default=4, type=int, help="number of dataLoader workers")
    parser.add_argument('-pim', '--pin_memory', default=False, action='store_true')
    parser.add_argument('-fbw', '--find_best_workers', default=False, action='store_true')
    parser.add_argument('-bs', '--batch_size', default=128, type=int, help="number of batch size")
    parser.add_argument('-dbb', '--db_batch', default=128, type=int, help="number of batch size for db")
    parser.add_argument('-lrs', '--lr_new', default=1e-3, type=float, help="siamese learning rate")
    parser.add_argument('-lrr', '--lr_resnet', default=1e-6, type=float, help="resnet learning rate")
    parser.add_argument('-warm', '--warm', default=False, action='store_true', help='Warmup learning rates')
    parser.add_argument('-lf', '--log_freq', default=10, type=int, help="show result after each show_every iter.")
    parser.add_argument('-sf', '--save_freq', default=100, type=int, help="save model after each save_every iter.")
    parser.add_argument('-tf', '--test_freq', default=100, type=int, help="test model after each test_every iter.")
    # parser.add_argument('-ms', '--max_steps', default=50000, type=int, help="number of steps before stopping")
    parser.add_argument('-ep', '--epochs', default=1, type=int, help="number of epochs before stopping")
    parser.add_argument('-es', '--early_stopping', default=20, type=int, help="number of tol for validation acc")
    parser.add_argument('-tst', '--test', default=False, action='store_true')
    parser.add_argument('-store_features_knn', '--store_features_knn', default=False, action='store_true')
    parser.add_argument('-katn', '--katn', default=False, action='store_true')
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('-sr', '--sampled_results', default=False, action='store_true')
    parser.add_argument('-pcr', '--per_class_results', default=True, action='store_true')
    parser.add_argument('-ptb', '--project_tb', default=False, action='store_true')

    parser.add_argument('-mg', '--margin', default=0.0, type=float, help="margin for triplet loss")
    parser.add_argument('-lss', '--loss', default='bce',
                        choices=['bce', 'trpl', 'maxmargin', 'batchhard', 'batchallgen', 'contrastive', 'stopgrad'])
    parser.add_argument('-soft', '--softmargin', default=False, action='store_true')
    parser.add_argument('-mm', '--merge_method', default='sim', choices=MERGE_METHODS)
    parser.add_argument('-bco', '--bcecoefficient', default=1.0, type=float, help="BCE loss weight")
    parser.add_argument('-tco', '--trplcoefficient', default=1.0, type=float, help="TRPL loss weight")
    parser.add_argument('-wd', '--weight_decay', default=1e-4, type=float, help="Decoupled Weight Decay Regularization")
    parser.add_argument('-reg_lambda', '--reg_lambda', default=0.0, type=float,
                        help="KoLeo Regularizer Lambda for Contrastive loss")

    parser.add_argument('-gamma', '--gamma', default=1.0, type=float, help="Learning Rate Scheduler")
    parser.add_argument('-gamma_step', '--gamma_step', default=1, type=int, help="Learning Rate Scheduler Step")
    parser.add_argument('-lr_tol', '--lr_tol', default=3, type=int, help="Adaptive Learning Rate Scheduler Tolerance")
    parser.add_argument('-lr_adp_loss', '--lr_adaptive_loss', default=False, action='store_true')

    parser.add_argument('-kbm', '--k_best_maps', nargs='+', help="list of k best activation maps")
    parser.add_argument('-fml', '--feature_map_layers', nargs='+', default=[],
                        help="feature maps for local merge")  # 1, 2, 3, 4

    parser.add_argument('-merge_global', '--merge_global', default=False, action='store_true')
    parser.add_argument('-no_global', '--no_global', default=False, action='store_true')

    parser.add_argument('-my_dist', '--my_dist', default=False, action='store_true')

    parser.add_argument('-hparams', '--hparams', default=False, action='store_true')
    parser.add_argument('-n', '--normalize', default=False, action='store_true')
    parser.add_argument('-dg', '--debug_grad', default=False, action='store_true')
    parser.add_argument('-dl', '--drop_last', default=False, action='store_true')
    parser.add_argument('-cam', '--cam', default=False, action='store_true')
    parser.add_argument('-dat', '--draw_all_thresh', default=32, type=int, help="threshold for drawing all heatmaps")
    parser.add_argument('-p', '--bh_P', default=18, type=int, help="number of classes for batchhard")
    parser.add_argument('-k', '--bh_K', default=4, type=int, help="number of imgs per class for batchhard")
    parser.add_argument('-m', '--aug_mask', default=False, action='store_true')
    parser.add_argument('-cm', '--colored_mask', default=False, action='store_true')
    parser.add_argument('-fs', '--from_scratch', default=False, action='store_true')
    parser.add_argument('-fd', '--fourth_dim', default=False, action='store_true')
    parser.add_argument('-camp', '--cam_path', default='cam_info_hotels.txt')

    parser.add_argument('--new_hotel_split_train', default='new_split_train.csv')
    parser.add_argument('--new_hotel_split_val', default='new_split_val.csv')
    parser.add_argument('--new_hotel_split_test', default='new_split_test.csv')
    # parser.add_argument('--new_hotel_split_query',  nargs='+', default=['new_split_query1.csv'])
    # parser.add_argument('--new_hotel_split_index',  nargs='+', default=['new_split_index1.csv'])


    parser.add_argument('--valsets',  nargs='+', default=[])
    parser.add_argument('--testsets', nargs='+', default=[])

    parser.add_argument('-qi', '--query_index', default=False, action='store_true')
    parser.add_argument('--queries', nargs='+', default=[])
    parser.add_argument('--indices', nargs='+', default=[])
    parser.add_argument('-ciq', '--classes_in_query', default=0, type=int, help="number of classes in each query/index run") # 2200



    parser.add_argument('-tqi', '--test_query_index', default=False, action='store_true')
    parser.add_argument('--t_queries', nargs='+', default=[])
    parser.add_argument('--t_indices', nargs='+', default=[])



    parser.add_argument('--train_folder_name', default='train')
    parser.add_argument('--vs_folder_name', default='val_seen')
    parser.add_argument('--vu_folder_name', default='val_unseen')
    parser.add_argument('--ts_folder_name', default='test_seen')
    parser.add_argument('--tu_folder_name', default='test_unseen')
    parser.add_argument('-ppth', '--project_path',
                        default='/home/aarash/projects/def-rrabba/aarash/ht-image-twoloss/ht-image/')
    parser.add_argument('-lpth', '--local_path',
                        default='/home/aarash/projects/def-rrabba/aarash/ht-image-twoloss/ht-image/')
    parser.add_argument('-jid', '--job_id', default='')
    parser.add_argument('-ss', '--static_size', default=0, type=int, help="number of neurons in classifier network")
    parser.add_argument('-dr', '--dim_reduction', default=0, type=int, help="dim reduction after feature extractor")

    parser.add_argument('-btf', '--bcotco_freq', default=0, type=int, help="frequency for bco dand tripl scheduler")
    parser.add_argument('-bcob', '--bco_base', default=1.0, type=float,
                        help="bco divide by ... every bcotco_freq epochs")
    parser.add_argument('-tcob', '--tco_base', default=1.0, type=float,
                        help="tco divide by ... every bcotco_freq epochs")

    parser.add_argument('-trf', '--train_fewshot', default=False, action='store_true')
    parser.add_argument('-tdp', '--train_diff_plot', default=False, action='store_true')

    parser.add_argument('-bnbc', '--bn_before_classifier', default=False, action='store_true')
    parser.add_argument('-leaky', '--leaky_relu', default=False, action='store_true')
    parser.add_argument('-draw_top_k_results', '--draw_top_k_results', default=5, type=int)

    parser.add_argument('-att', '--attention', default=False, action='store_true')

    parser.add_argument('-dp_type', '--dp_type', default='both',
                        choices=['query', 'key', 'both'])
    parser.add_argument('-add_local_features', '--add_local_features', default=False, action='store_true')

    parser.add_argument('-l2l', '--local_to_local', default=False, action='store_true')

    parser.add_argument('-att_mode_sc', '--att_mode_sc', default='spatial', choices=['spatial',
                                                                                     'channel',
                                                                                     'both',
                                                                                     'glb-both',
                                                                                     'unet-att',
                                                                                     'dot-product',
                                                                                     'dot-product-add'])

    parser.add_argument('-att_weight_init', '--att_weight_init', default=None, type=float, help="initialize glb-both att")

    parser.add_argument('-att_on_all', '--att_on_all', default=False, action='store_true')

    parser.add_argument('-cross_attention', '--cross_attention', default=False, action='store_true')
    parser.add_argument('-spp', '--same_pic_prob', default=0.0, type=float, help="Probability of choosing the same "
                                                                                 "image for positive pair")

    parser.add_argument('-aet', '--att_extra_layer', default=2, type=int, help="number of ")

    parser.add_argument('-smds', '--softmax_diff_sim', default=False, action='store_true')

    parser.add_argument('-spatial_projection', '--spatial_projection', default=False, action='store_true')
    parser.add_argument('--small_and_big', default=False, action='store_true')

    args = parser.parse_args()

    return args


# https://www.overleaf.com/5846741514dywtdjdpmxwn
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


def get_file_name(file_path):
    temp = file_path[file_path.rfind('/') + 1:]
    ret = temp[:temp.rfind('.')]
    return ret


def get_number_of_parameters(net):
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_params


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


def get_val_loaders(args, val_sets, workers, pin_memory, batch_size=None):
    val_loaders = []

    if not val_sets:
        return None

    if not batch_size:
        batch_size = args.way

    for val_set in val_sets:
        val_loaders.append(
            DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers,
                       pin_memory=pin_memory, drop_last=args.drop_last))

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
                     even_sampled=True, per_class=False, save_path='', mode='',
                     sim_matrix=None, metric='cosine', query_index=False, extra_name=''):
    if query_index and len(img_lbls) == 2 and len(img_feats) == 2:
        logger.info('K@N unsampled class')
        # unsampled_total = _get_per_class_distance_qi(args, img_feats[0], img_feats[1], img_lbls[0], img_lbls[1], logger, mode,
        #                                               sim_matrix=sim_matrix, metric=metric, extra_name=extra_name)
        unsampled_total, _, _ = _get_sampled_distance_qi(args, img_feats[0], img_feats[1], img_lbls[0], img_lbls[1],
                                                      logger, limit,
                                                      run_number, mode,
                                                      sim_matrix=sim_matrix, metric=metric, extra_name=extra_name,
                                                      sampled=False) # unsampled

        unsampled_total.to_csv(os.path.join(save_path, f'{args.dataset_name}_{mode}_{extra_name}_UNsampled_avg_k@n.csv'),
                     header=True, index=False)

        kavg = pd.DataFrame()
        if sampled:
            all_run_avgs = None
            for i in range(args.number_of_runs):
                logger.info(f'K@N sample {i} class')
                kavg_i, kruns, total = _get_sampled_distance_qi(args, img_feats[0], img_feats[1], img_lbls[0], img_lbls[1], logger, limit,
                                      run_number, mode,
                                      sim_matrix=sim_matrix, metric=metric, extra_name=extra_name)

                if all_run_avgs is None:
                    all_run_avgs = kavg_i
                else:
                    all_run_avgs = pd.concat([all_run_avgs, kavg_i])
                # kavg.to_csv(os.path.join(save_path, f'{args.dataset_name}_{mode}_{extra_name}_per_class_total_avg_k@n_r{i}.csv'), header=True,
                #              index=False)
            all_run_avgs = all_run_avgs.reset_index()
            all_run_avgs.to_csv(
                os.path.join(save_path, f'{args.dataset_name}_{mode}_{extra_name}_all_sampled_avg_k@n.csv'),
                header=True,
                index=False)

            kavg = pd.DataFrame()
            for c in all_run_avgs.columns:
                all_mean = np.array(all_run_avgs[c]).mean()
                kavg[c] = [all_mean]

            kavg.to_csv(
                os.path.join(save_path, f'{args.dataset_name}_{mode}_{extra_name}_sampled_avg_k@n.csv'),
                header=True,
                index=False)

    else:
        if per_class:
            logger.info('K@N per class')
            unsampled_total, seen, unseen = _get_per_class_distance(args, img_feats, img_lbls, seen_list, logger, mode,
                                                          sim_matrix=sim_matrix, metric=metric)
            unsampled_total.to_csv(os.path.join(save_path, f'{args.dataset_name}_{mode}_per_class_total_avg_k@n.csv'), header=True,
                         index=False)
            seen.to_csv(os.path.join(save_path, f'{args.dataset_name}_{mode}_per_class_seen_avg_k@n.csv'), header=True,
                        index=False)
            unseen.to_csv(os.path.join(save_path, f'{args.dataset_name}_{mode}_per_class_unseen_avg_k@n.csv'), header=True,
                          index=False)

        if sampled:
            logger.info('K@N for sampled')
            kavg, kruns, total, seen, unseen = _get_sampled_distance(args, img_feats, img_lbls, seen_list, logger, limit,
                                                                     run_number, mode, even_sampled=even_sampled,
                                                                     sim_matrix=sim_matrix, metric=metric)
            kavg.to_csv(os.path.join(save_path, f'{args.dataset_name}_{mode}_sampled_avg_k@n.csv'), header=True,
                        index=False)
            kruns.to_csv(os.path.join(save_path, f'{args.dataset_name}_{mode}_sampled_runs_k@n.csv'), header=True,
                         index=False)
            total.to_csv(os.path.join(save_path, f'{args.dataset_name}_{mode}_sampled_per_class_total_avg_k@n.csv'),
                         header=True, index=False)
            seen.to_csv(os.path.join(save_path, f'{args.dataset_name}_{mode}_sampled_per_class_seen_avg_k@n.csv'),
                        header=True, index=False)
            unseen.to_csv(os.path.join(save_path, f'{args.dataset_name}_{mode}_sampled_per_class_unseen_avg_k@n.csv'),
                          header=True, index=False)

    return kavg, unsampled_total

def _get_per_class_distance_qi(args, img_feats_q, img_feats_i, img_lbls_q, img_lbls_i, logger, mode, sim_matrix=None, metric='cosine', extra_name=''):
    all_lbls = np.unique(img_lbls_q)
    num = img_lbls_q.shape[0]

    k_max = min(1000, img_lbls_q.shape[0])

    if sim_matrix is None:
        _, I = get_faiss_query_index(img_feats_q, img_feats_i, k=k_max, gpu=True, metric=metric)
    else:
        I = (-sim_matrix).argsort()[:, :int(k_max)]

    metric_total = metrics.Accuracy_At_K(classes=np.array(all_lbls))

    for idx, lbl in enumerate(img_lbls_q):

        ret_lbls = img_lbls_i[I[idx]]

        metric_total.update(lbl, ret_lbls)

    total = metric_total.get_per_class_metrics()

    logger.info(f'%%%%%%%%%%%%%%%%%%% {extra_name}')
    logger.info(f'{args.dataset_name}_{mode}')
    logger.info('Without sampling Total: ' + str(metric_total.n))
    logger.info(metric_total)

    logger.info(f'{args.dataset_name}_{mode}')
    _log_per_class(logger, total, split_kind='Total')

    return total


def _get_per_class_distance(args, img_feats, img_lbls, seen_list, logger, mode, sim_matrix=None, metric='cosine'):
    all_lbls = np.unique(img_lbls)
    seen_lbls = np.unique(img_lbls[seen_list == 1])
    unseen_lbls = np.unique(img_lbls[seen_list == 0])
    num = img_lbls.shape[0]

    k_max = min(1000, img_lbls.shape[0])

    if sim_matrix is None:
        _, I, self_D = get_faiss_knn(img_feats, k=k_max, gpu=True, metric=metric)
    else:
        minval = np.min(sim_matrix) - 1.
        self_D = -(np.diag(sim_matrix))
        sim_matrix -= np.diag(np.diag(sim_matrix))
        sim_matrix += np.diag(np.ones(num) * minval)
        I = (-sim_matrix).argsort()[:, :-1]

    metric_total = metrics.Accuracy_At_K(classes=np.array(all_lbls))
    metric_seen = metrics.Accuracy_At_K(classes=np.array(seen_lbls))
    metric_unseen = metrics.Accuracy_At_K(classes=np.array(unseen_lbls))

    for idx, (lbl, seen) in enumerate(zip(img_lbls, seen_list)):

        ret_seens = seen_list[I[idx]]
        ret_lbls = img_lbls[I[idx]]

        metric_total.update(lbl, ret_lbls)

        if seen == 1:
            metric_seen.update(lbl, ret_lbls[ret_seens == 1])
        else:
            metric_unseen.update(lbl, ret_lbls[ret_seens == 0])

    total = metric_total.get_per_class_metrics()
    seen = metric_seen.get_per_class_metrics()
    unseen = metric_unseen.get_per_class_metrics()

    logger.info(f'{args.dataset_name}_{mode}')
    logger.info('Without sampling Total: ' + str(metric_total.n))
    logger.info(metric_total)

    logger.info(f'{args.dataset_name}_{mode}')
    _log_per_class(logger, total, split_kind='Total')

    logger.info(f'{args.dataset_name}_{mode}')
    logger.info('Without sampling Seen: ' + str(metric_seen.n))
    logger.info(metric_seen)

    logger.info(f'{args.dataset_name}_{mode}')
    _log_per_class(logger, seen, split_kind='Seen')

    logger.info(f'{args.dataset_name}_{mode}')
    logger.info('Without sampling Unseen: ' + str(metric_unseen.n))
    logger.info(metric_unseen)

    logger.info(f'{args.dataset_name}_{mode}')
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
    logger.info(f'k@2 per class average: {np.array(df["k@2"]).mean()}')
    logger.info(f'k@4 per class average: {np.array(df["k@4"]).mean()}')
    logger.info(f'k@5 per class average: {np.array(df["k@5"]).mean()}')
    logger.info(f'k@8 per class average: {np.array(df["k@8"]).mean()}')
    logger.info(f'k@10 per class average: {np.array(df["k@10"]).mean()}')
    logger.info(f'k@100 per class average: {np.array(df["k@100"]).mean()}\n')

def _get_sampled_distance_qi(args, img_feats_q, img_feats_i, img_lbls_q, img_lbls_i, logger, limit=0, run_number=0, mode='',
                             sim_matrix=None, metric='cosine', extra_name='', sampled=True):
    all_lbls = np.unique(img_lbls_q)

    if sampled:
        img_feats_q_sampled, img_feats_i_sampled, img_lbls_q_sampled, img_lbls_i_sampled = get_sampled_query_index(img_feats_q,
                                                                                                                   img_feats_i,
                                                                                                                   img_lbls_q,
                                                                                                                   img_lbls_i,
                                                                                                                   classes=args.classes_in_query)
    else:
        img_feats_q_sampled, img_feats_i_sampled, img_lbls_q_sampled, img_lbls_i_sampled = img_feats_q, img_feats_i, img_lbls_q, img_lbls_i

        # sim_mat = cosine_similarity(chosen_img_feats)
    print(f'$$$$ {extra_name}: Total of {len(img_lbls_q_sampled)} queries and {len(img_lbls_i_sampled)} indices')

    k_max = min(1000, img_lbls_q_sampled.shape[0])

    if sim_matrix is not None:
        I = (-sim_matrix).argsort()[:, :-1]
    else:
        _, I = get_faiss_query_index(img_feats_q_sampled, img_feats_i_sampled, k=k_max, gpu=True, metric=metric)

    metric_total = metrics.Accuracy_At_K(classes=all_lbls)

    for idx, lbl in enumerate(img_lbls_q_sampled):

        ret_lbls = img_lbls_i_sampled[I[idx]]

        metric_total.update(lbl, ret_lbls)


    total = metric_total.get_per_class_metrics()

    logger.info('Total: ' + str(metric_total.n))
    logger.info(metric_total)
    k1, k2, k4, k5, k8, k10, k100 = metric_total.get_tot_metrics()
    logger.info("*" * 50)


    _log_per_class(logger, total, split_kind='Total')

    total = metric_total.get_per_class_metrics()

    logger.info(f'%%%%%%%%%%%%%%%%%%% {extra_name}')
    logger.info('Avg Total: ' + str(metric_total.n))
    logger.info('k@1: ' + str(np.round(k1, decimals=3)))
    logger.info('k@2: ' + str(np.round(k2, decimals=3)))
    logger.info('k@4: ' + str(np.round(k4, decimals=3)))
    logger.info('k@5: ' + str(np.round(k5, decimals=3)))
    logger.info('k@8: ' + str(np.round(k8, decimals=3)))
    logger.info('k@10: ' + str(np.round(k10, decimals=3)))
    logger.info('k@100: ' + str(np.round(k100, decimals=3)))
    logger.info("*" * 50)

    d = {'run': [i for i in range(run_number)],
         'kAT1': [k1],
         'kAT2': [k2],
         'kAT4': [k4],
         'kAT5': [k5],
         'kAT8': [k8],
         'kAT10': [k10],
         'kAT100': [k100]}

    average_tot = pd.DataFrame(data={'avg_kAT1': [np.round(k1, decimals=3)],
                                     'avg_kAT2': [np.round(k2, decimals=3)],
                                     'avg_kAT4': [np.round(k4, decimals=3)],
                                     'avg_kAT5': [np.round(k5, decimals=3)],
                                     'avg_kAT8': [np.round(k8, decimals=3)],
                                     'avg_kAT10': [np.round(k10, decimals=3)],
                                     'avg_kAT100': [np.round(k100, decimals=3)]})

    return average_tot, None, total


def _get_sampled_distance(args, img_feats, img_lbls, seen_list, logger, limit=0, run_number=0, mode='',
                          even_sampled=False, sim_matrix=None, metric='cosine'):
    all_lbls = np.unique(img_lbls)
    seen_lbls = np.unique(img_lbls[seen_list == 1])
    unseen_lbls = np.unique(img_lbls[seen_list == 0])

    k1s = []
    k2s = []
    k4s = []
    k5s = []
    k8s = []
    k10s = []
    k100s = []

    k1s_s = []
    k2s_s = []
    k4s_s = []
    k5s_s = []
    k8s_s = []
    k10s_s = []
    k100s_s = []

    k1s_u = []
    k2s_u = []
    k4s_u = []
    k5s_u = []
    k8s_u = []
    k10s_u = []
    k100s_u = []

    if args.dataset_name.startswith('hotels') and even_sampled:
        sampled_indices_all = pd.read_csv(
            os.path.join(args.project_path, 'sample_index_por' + str(args.portion) + '.csv'))
        sampled_label_all = pd.read_csv(
            os.path.join(args.project_path, 'sample_label_por' + str(args.portion) + '.csv'))

    if not even_sampled:
        logger.info(f'### K@N only once. Not even samples, so no randomization.')
        run_number = 1

    for run in range(run_number):
        column_name = f'run{run}'
        if args.dataset_name.startswith('hotels') and even_sampled:
            sampled_indices = np.array(sampled_indices_all[column_name]).astype(int)
            sampled_labels = np.array(sampled_label_all[column_name]).astype(int)

            logger.info(f'{args.dataset_name}_{mode}')
            logger.info('### Run ' + str(run) + "...")
            chosen_img_feats = img_feats[sampled_indices]
            chosen_img_lbls = img_lbls[sampled_indices]
            chosen_seen_list = seen_list[sampled_indices]

            assert np.array_equal(sampled_labels, chosen_img_lbls)

        else:
            logger.info(f'### Run {run} with NOT even samples...')

            chosen_img_feats = img_feats
            chosen_img_lbls = img_lbls
            chosen_seen_list = seen_list

        # sim_mat = cosine_similarity(chosen_img_feats)
        k_max = min(1000, img_lbls.shape[0])

        if sim_matrix is not None:
            num = img_lbls.shape[0]

            minval = np.min(sim_matrix) - 1.
            self_D = -(np.diag(sim_matrix))
            sim_matrix -= np.diag(np.diag(sim_matrix))
            sim_matrix += np.diag(np.ones(num) * minval)
            I = (-sim_matrix).argsort()[:, :-1]
        else:
            _, I, self_D = get_faiss_knn(chosen_img_feats, k=k_max, gpu=True, metric=metric)

        metric_total = metrics.Accuracy_At_K(classes=all_lbls)
        metric_seen = metrics.Accuracy_At_K(classes=seen_lbls)
        metric_unseen = metrics.Accuracy_At_K(classes=unseen_lbls)

        for idx, (lbl, seen) in enumerate(zip(chosen_img_lbls, chosen_seen_list)):

            ret_seens = chosen_seen_list[I[idx]]
            ret_lbls = chosen_img_lbls[I[idx]]

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
        k1, k2, k4, k5, k8, k10, k100 = metric_total.get_tot_metrics()
        k1s.append(k1)
        k2s.append(k2)
        k4s.append(k4)
        k5s.append(k5)
        k8s.append(k8)
        k10s.append(k10)
        k100s.append(k100)
        logger.info("*" * 50)

        logger.info('Seen: ' + str(metric_seen.n))
        logger.info(metric_seen)
        k1, k2, k4, k5, k8, k10, k100 = metric_seen.get_tot_metrics()
        k1s_s.append(k1)
        k2s_s.append(k2)
        k4s_s.append(k4)
        k5s_s.append(k5)
        k8s_s.append(k8)
        k10s_s.append(k10)
        k100s_s.append(k100)
        logger.info("*" * 50)

        logger.info('Unseen: ' + str(metric_unseen.n))
        logger.info(metric_unseen)
        k1, k2, k4, k5, k8, k10, k100 = metric_unseen.get_tot_metrics()
        k1s_u.append(k1)
        k2s_u.append(k2)
        k4s_u.append(k4)
        k5s_u.append(k5)
        k8s_u.append(k8)
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
    logger.info('k@1: ' + str(np.round(np.array(k1s).mean(), decimals=3)))
    logger.info('k@2: ' + str(np.round(np.array(k2s).mean(), decimals=3)))
    logger.info('k@4: ' + str(np.round(np.array(k4s).mean(), decimals=3)))
    logger.info('k@5: ' + str(np.round(np.array(k5s).mean(), decimals=3)))
    logger.info('k@8: ' + str(np.round(np.array(k8s).mean(), decimals=3)))
    logger.info('k@10: ' + str(np.round(np.array(k10s).mean(), decimals=3)))
    logger.info('k@100: ' + str(np.round(np.array(k100s).mean(), decimals=3)))
    logger.info("*" * 50)

    logger.info('Avg Seen: ' + str(metric_seen.n))
    logger.info('k@1: ' + str(np.round(np.array(k1s_s).mean(), decimals=3)))
    logger.info('k@2: ' + str(np.round(np.array(k2s_s).mean(), decimals=3)))
    logger.info('k@4: ' + str(np.round(np.array(k4s_s).mean(), decimals=3)))
    logger.info('k@5: ' + str(np.round(np.array(k5s_s).mean(), decimals=3)))
    logger.info('k@8: ' + str(np.round(np.array(k8s_s).mean(), decimals=3)))
    logger.info('k@10: ' + str(np.round(np.array(k10s_s).mean(), decimals=3)))
    logger.info('k@100: ' + str(np.round(np.array(k100s_s).mean(), decimals=3)))
    logger.info("*" * 50)

    logger.info('Avg Unseen: ' + str(metric_unseen.n))
    logger.info('k@1: ' + str(np.round(np.array(k1s_u).mean(), decimals=3)))
    logger.info('k@2: ' + str(np.round(np.array(k2s_u).mean(), decimals=3)))
    logger.info('k@4: ' + str(np.round(np.array(k4s_u).mean(), decimals=3)))
    logger.info('k@5: ' + str(np.round(np.array(k5s_u).mean(), decimals=3)))
    logger.info('k@8: ' + str(np.round(np.array(k8s_u).mean(), decimals=3)))
    logger.info('k@10: ' + str(np.round(np.array(k10s_u).mean(), decimals=3)))
    logger.info('k@100: ' + str(np.round(np.array(k100s_u).mean(), decimals=3)))
    logger.info("*" * 50)

    d = {'run': [i for i in range(run_number)],
         'kAT1': k1s,
         'kAT2': k2s,
         'kAT4': k4s,
         'kAT5': k5s,
         'kAT8': k8s,
         'kAT10': k10s,
         'kAT100': k100s,
         'kAT1_seen': k1s_s,
         'kAT2_seen': k2s_s,
         'kAT4_seen': k4s_s,
         'kAT5_seen': k5s_s,
         'kAT8_seen': k8s_s,
         'kAT10_seen': k10s_s,
         'kAT100_seen': k100s_s,
         'kAT1_unseen': k1s_u,
         'kAT2_unseen': k2s_u,
         'kAT4_unseen': k4s_u,
         'kAT5_unseen': k5s_u,
         'kAT8_unseen': k8s_u,
         'kAT10_unseen': k10s_u,
         'kAT100_unseen': k100s_u}

    average_tot = pd.DataFrame(data={'avg_kAT1': [np.round(np.array(k1s).mean(), decimals=3)],
                                     'avg_kAT2': [np.round(np.array(k2s).mean(), decimals=3)],
                                     'avg_kAT4': [np.round(np.array(k4s).mean(), decimals=3)],
                                     'avg_kAT5': [np.round(np.array(k5s).mean(), decimals=3)],
                                     'avg_kAT8': [np.round(np.array(k8s).mean(), decimals=3)],
                                     'avg_kAT10': [np.round(np.array(k10s).mean(), decimals=3)],
                                     'avg_kAT100': [np.round(np.array(k100s).mean(), decimals=3)],
                                     'avg_kAT1_seen': [np.round(np.array(k1s_s).mean(), decimals=3)],
                                     'avg_kAT2_seen': [np.round(np.array(k2s_s).mean(), decimals=3)],
                                     'avg_kAT4_seen': [np.round(np.array(k4s_s).mean(), decimals=3)],
                                     'avg_kAT5_seen': [np.round(np.array(k5s_s).mean(), decimals=3)],
                                     'avg_kAT8_seen': [np.round(np.array(k8s_s).mean(), decimals=3)],
                                     'avg_kAT10_seen': [np.round(np.array(k10s_s).mean(), decimals=3)],
                                     'avg_kAT100_seen': [np.round(np.array(k100s_s).mean(), decimals=3)],
                                     'avg_kAT1_unseen': [np.round(np.array(k1s_u).mean(), decimals=3)],
                                     'avg_kAT2_unseen': [np.round(np.array(k2s_u).mean(), decimals=3)],
                                     'avg_kAT4_unseen': [np.round(np.array(k4s_u).mean(), decimals=3)],
                                     'avg_kAT5_unseen': [np.round(np.array(k5s_u).mean(), decimals=3)],
                                     'avg_kAT8_unseen': [np.round(np.array(k8s_u).mean(), decimals=3)],
                                     'avg_kAT10_unseen': [np.round(np.array(k10s_u).mean(), decimals=3)],
                                     'avg_kAT100_unseen': [np.round(np.array(k100s_u).mean(), decimals=3)]})

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
            ls = [(lbl, value, bl) for value, bl in value_list]
        else:
            ls = [(lbl, value) for value in value_list]

        data.extend(ls)

    if shuffle:
        np.random.shuffle(data)

    return data


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return True


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


def _read_new_split(dataset_path, file_name):  # mode = [test_seen, val_seen, train, test_unseen, test_unseen]

    file = pd.read_csv(os.path.join(dataset_path, file_name))
    image_labels = np.array(file.label)
    image_path = np.array(file.image)

    return image_path, image_labels


def _get_imgs_labels(dataset_path, extensions):
    classes = [d.name for d in os.scandir(dataset_path) if d.is_dir()]
    classes.sort()
    mode = dataset_path.split('/')[-1]
    try:
        class_to_idx = {cls_name: int(cls_name) for _, cls_name in enumerate(classes)}
    except ValueError:
        print('** Relabeling classes to integers')
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    image_path = []
    image_labels = []
    for c in classes:
        imgs = [os.path.join(mode, c, d.name) for d in os.scandir(os.path.join(dataset_path, c)) if
                d.name.lower().endswith(extensions)]
        image_path.extend(imgs)
        image_labels.extend([class_to_idx[c] for _ in range(len(imgs))])

    image_path = np.array(image_path)
    image_labels = np.array(image_labels)
    return image_path, image_labels


def loadDataToMem_2(dataPath, dataset_name, mode='train',
                    portion=0, return_bg=True,
                    extensions=('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')):
    dataset_path = os.path.join(dataPath, dataset_name)

    return_bg = return_bg and (mode.startswith('train'))

    background_datasets = {'val_seen': 'val_unseen',
                           'val_unseen': 'val_seen',
                           'test_seen': 'test_unseen',
                           'test_unseen': 'test_seen',
                           'train_seen': 'train_seen'}

    print("begin loading dataset to memory")
    datas = {}
    datas_bg = {}  # in case of mode == val/test_seen/unseen

    image_path, image_labels = _get_imgs_labels(os.path.join(dataset_path, mode), extensions)

    if return_bg:
        image_path_bg, image_labels_bg = _get_imgs_labels(os.path.join(dataset_path, background_datasets[mode]),
                                                          extensions)

    if portion > 0:
        image_path = image_path[image_labels < portion]
        image_labels = image_labels[image_labels < portion]

        if return_bg:
            image_path_bg = image_path_bg[image_labels_bg < portion]
            image_labels_bg = image_labels_bg[image_labels_bg < portion]

    print(f'{dataset_name}_{mode} number of imgs:', len(image_labels))
    print(f'{dataset_name}_{mode} number of labels:', len(np.unique(image_labels)))

    if return_bg:
        print(f'{dataset_name}_{mode} number of bg imgs:', len(image_labels_bg))
        print(f'{dataset_name}_{mode} number of bg lbls:', len(np.unique(image_labels_bg)))
    else:
        print(f'Just {dataset_name}_{mode}, background not required.')

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
    print(f'Number of labels in {dataset_name}_{mode}: ', len(labels))

    if return_bg:
        all_labels = np.unique(np.concatenate((image_labels, image_labels_bg)))
        print(f'Number of all labels (bg + fg) in {dataset_name}_{mode} and {background_datasets[mode]}: ',
              len(all_labels))

    if not return_bg:
        datas_bg = datas

    print(f'finish loading {dataset_name}_{mode} dataset to memory')
    return datas, num_classes, num_instances, labels, datas_bg


def load_Data_ToMem(dataPath,  dataset_folder, mode='train',
                    portion=0,
                    extensions=('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'),
                    split_file_path=''):
    dataset_path = os.path.join(dataPath, dataset_folder)

    print("begin loading dataset to memory")
    datas = {}

    if mode.endswith('.csv'):
        image_path, image_labels = _read_new_split(split_file_path, mode)
    else:
        image_path, image_labels = _get_imgs_labels(os.path.join(dataset_path, mode), extensions)


    if portion > 0:
        image_path = image_path[image_labels < portion]
        image_labels = image_labels[image_labels < portion]


    print(f'{dataset_folder}_{mode} number of imgs:', len(image_labels))
    print(f'{dataset_folder}_{mode} number of labels:', len(np.unique(image_labels)))


    num_instances = len(image_labels)

    num_classes = len(np.unique(image_labels))

    for idx, path in zip(image_labels, image_path):
        if idx not in datas.keys():
            datas[idx] = []

        datas[idx].append(os.path.join(dataset_path, path))

    labels = np.unique(image_labels)
    print(f'Number of labels in {dataset_folder}_{mode}: ', len(labels))

    print(f'finish loading {dataset_folder}_{mode} dataset to memory')

    return datas, num_classes, num_instances, labels


def load_hotelData_ToMem(dataPath, dataset_name, mode='train', split_file_path='',
                         portion=0, return_bg=True, dataset_folder='', hotels_splits=''):
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

    if hotels_splits == '':
        file_name = f'{dataset_name}_' + mode + '.csv'
    else:
        file_name = hotels_splits

    print(f'{mode}', file_name)
    image_path, image_labels = _read_new_split(split_file_path, file_name)
    if return_bg:
        file_name = f'{dataset_name}_' + background_datasets[mode] + '.csv'
        image_path_bg, image_labels_bg = _read_new_split(split_file_path, file_name)

    if portion > 0:
        image_path = image_path[image_labels < portion]
        image_labels = image_labels[image_labels < portion]

        if return_bg:
            image_path_bg = image_path_bg[image_labels_bg < portion]
            image_labels_bg = image_labels_bg[image_labels_bg < portion]

    print(f'{dataset_name}_{mode} number of imgs:', len(image_labels))
    print(f'{dataset_name}_{mode} number of labels:', len(np.unique(image_labels)))

    if return_bg:
        print(f'{dataset_name}_{mode} number of bg imgs:', len(image_labels_bg))
        print(f'{dataset_name}_{mode} number of bg lbls:', len(np.unique(image_labels_bg)))
    else:
        print(f'Just {dataset_name}_{mode}, background not required.')

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
    print(f'Number of labels in {dataset_name}_{mode}: ', len(labels))

    if return_bg:
        all_labels = np.unique(np.concatenate((image_labels, image_labels_bg)))
        print(f'Number of all labels (bg + fg) in {dataset_name}_{mode} and {background_datasets[mode]}: ',
              len(all_labels))

    if not return_bg:
        datas_bg = datas

    print(f'finish loading {dataset_name}_{mode} dataset to memory')


    return datas, num_classes, num_instances, labels, datas_bg

def load_splits(dataPath, dataset_name, main_split, mode='train', split_file_path='',
                         portion=0, dataset_folder='', backgroud_splits=[]):
    print(split_file_path, '!!!!!!!!')
    dataset_path = os.path.join(dataPath, dataset_folder)


    print("begin loading dataset to memory")
    datas = {}
    datas_bg = {}  # in case of mode == val/test_seen/unseen


    file_name = f'{dataset_name}_' + main_split


    print(f'{mode}', file_name)
    image_path, image_labels = _read_new_split(split_file_path, file_name)

    if backgroud_splits != []:
        if len(backgroud_splits) == 1:
            file_name = f'{dataset_name}_' + backgroud_splits[0]
            image_path_bg, image_labels_bg = _read_new_split(split_file_path, file_name)
        else:
            image_path_bg, image_labels_bg = [], []
            for bs in backgroud_splits:
                file_name = f'{dataset_name}_' + bs
                image_path_bg_temp, image_labels_bg_temp = _read_new_split(split_file_path, file_name)
                image_path_bg.append(image_path_bg_temp)
                image_labels_bg.append(image_labels_bg_temp)

            image_path_bg = pd.concat(image_path_bg, ignore_index=True)
            image_labels_bg = pd.concat(image_labels_bg, ignore_index=True)


    if portion > 0:
        image_path = image_path[image_labels < portion]
        image_labels = image_labels[image_labels < portion]

        if backgroud_splits != []:
            image_path_bg = image_path_bg[image_labels_bg < portion]
            image_labels_bg = image_labels_bg[image_labels_bg < portion]

    print(f'{dataset_name}_{mode} number of imgs:', len(image_labels))
    print(f'{dataset_name}_{mode} number of labels:', len(np.unique(image_labels)))

    if backgroud_splits != []:
        print(f'{dataset_name}_{mode} number of bg imgs:', len(image_labels_bg))
        print(f'{dataset_name}_{mode} number of bg lbls:', len(np.unique(image_labels_bg)))
    else:
        print(f'Just {dataset_name}_{mode}, background not required.')

    num_instances = len(image_labels)

    num_classes = len(np.unique(image_labels))

    for idx, path in zip(image_labels, image_path):
        if idx not in datas.keys():
            datas[idx] = []
            if backgroud_splits != []:
                datas_bg[idx] = []

        datas[idx].append(os.path.join(dataset_path, path))
        if backgroud_splits != []:
            datas_bg[idx].append((os.path.join(dataset_path, path), True))

    if backgroud_splits != []:
        for idx, path in zip(image_labels_bg, image_path_bg):
            if idx not in datas_bg.keys():
                datas_bg[idx] = []
            if (os.path.join(dataset_path, path), False) not in datas_bg[idx] and \
                    (os.path.join(dataset_path, path), True) not in datas_bg[idx]:
                datas_bg[idx].append((os.path.join(dataset_path, path), False))

    labels = np.unique(image_labels)
    print(f'Number of labels in {dataset_name}_{mode}: ', len(labels))

    if backgroud_splits != []:
        all_labels = np.unique(np.concatenate((image_labels, image_labels_bg)))
        print(f'Number of all labels (bg + fg) in {dataset_name}_{mode} and {backgroud_splits}: ',
              len(all_labels))

    if backgroud_splits == []:
        datas_bg = datas

    print(f'finish loading {dataset_name}_{mode} dataset to memory')


    return datas, num_classes, num_instances, labels, datas_bg

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
        logger.info(
            f'Save directory {path} already exists, but how?? id_str = {id_str}')  # almost impossible if not pretrained


def read_masks(path):
    # create mask csv
    # read mask csv and paths
    masks = pd.read_csv(path)
    return masks


def get_overfit(data, labels, anchors=1, neg_per_pos=1, batchhard=[0, 0]):
    if batchhard != [0, 0]: # not batchhard
        anch_class = np.random.choice(labels, 1)
        neg_class = anch_class
        while neg_class == anch_class:
            neg_class = np.random.choice(labels, 1)

        to_return = []

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

                to_return.append({'anch': anch_path, 'pos': pos_path, 'neg': negs})
    else: #batchhard

        to_return = []
        bh_P = batchhard[0]
        bh_K = batchhard[1]

        for _ in range(anchors):
            batch = {}
            final_labels = {}
            for p in range(bh_P):
                label_idx = np.random.choice(labels, 1)
                label = labels[label_idx]
                if len(data[label]) >= bh_K:
                    random_paths = np.random.choice(data[label], size=bh_K, replace=False)
                else:
                    random_paths = np.random.choice(data[label], size=bh_K, replace=True)
                batch[p] = list(random_paths)
                final_labels[p] = list([label for _ in range(bh_K)])
            to_return.append({'batch': batch, 'labels': final_labels})

    return to_return


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


def activation_reduction(activations, method='avg'):  # avg or max
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


def get_heatmaps(activations, shape, save_path=None, label=None, normalize=[], method='avg', classifier_weights=None,
                 attention=False):
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
    # cpu_activations = np.array(list(map(activation_reduction,
    #                                     cpu_activations, method_list)))
    cpu_activations = [act.squeeze().data.cpu().numpy() for act in cpu_activations]

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf

    # heatmap = heatmap
    final_heatmaps = []

    if attention:
        for heatmap in cpu_activations:
            heatmap = heatmap / np.max(heatmap)
            final_heatmaps.append(__post_create_heatmap(heatmap, shape))
    else:
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


def vector_merge_function(v1, v2, method='sim', normalize=True, softmax=False):
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
    elif 'diff-sim' in method:
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

        if softmax:
            diff_merged = F.softmax(diff_merged, dim=1)
            sim_merged = F.softmax(sim_merged, dim=1)

        return torch.cat([diff_merged, sim_merged], dim=1)
    elif method.startswith('concat'):
        merged = torch.cat([v1, v2], dim=1)
        return merged
    # elif method == 'diff-sim-con':
    #
    #     first_merged = torch.cat([v1, v2], dim=1)
    #
    #     diff_merged = torch.pow((v1 - v2), 2)
    #     diff_merged = F.normalize(diff_merged, p=2, dim=1)
    #     sim_merged = F.normalize(v1, p=2, dim=1) * F.normalize(v2, p=2, dim=1)
    #     second_merged = torch.cat([diff_merged, sim_merged], dim=1)
    #
    #     return torch.cat([first_merged, second_merged], dim=1)

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
def apply_grad_heatmaps(grads, activations, img_dict, label, id, path, plot_title, tb_path, epoch, writer):
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

        drew_plot = Image.open(path_)

        writer.add_image(tb_path + f'/{method}', np.array(drew_plot)[:, :, :3], epoch, dataformats='HWC')

        writer.flush()

    # for pic, path in zip(pics, paths):
    #     cv2.imwrite(path, pic)


@MY_DEC
def apply_forward_heatmap(acts, img_list, id, heatmap_path, overall_title,
                          titles=[''], histogram_path='',
                          merge_method='sim', classifier_weights=None, softmax=False, tb_path=None, epoch=None,
                          writer=None):
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

    acts.append(vector_merge_function(acts[0], acts[1], method=merge_method, softmax=softmax))  # anch_pos_subtraction
    titles.append(f'anch_pos_{merge_method}')

    acts.append(vector_merge_function(acts[0], acts[2], method=merge_method, softmax=softmax))  # anch_neg_subtraction
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
                                     classifier_weights=classifier_weights))  # seperated for normalization, heatmaps WITH classifier weights
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

        drew_plot = Image.open(heatmap_path[method])

        writer.add_image(tb_path + f'/{method}', np.array(drew_plot)[:, :, :3], epoch, dataformats='HWC')
        writer.flush()

        os.rmdir(heatmap_path[method])
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


def seperate_pos_neg(ts):
    pos_ts = []
    neg_ts = []
    for t in ts:
        pos_ts.append(t.relu())
        neg_ts.append((-t).relu())

    return pos_ts, neg_ts


def apply_attention_heatmap(atts, img_list, id, heatmap_path, overall_title,
                            titles=[''], tb_path=None, epoch=None, writer=None):
    """

    :param atts: [anch_activation_p,
                    pos_activation,
                    anch_activation_n,
                    neg_activation]
    :param img_list: [(anch_lbl, anch_img),
                        (pos_lbl, pos_img),
                        (neg_lbl, neg_img)]
    :param id:
    :param individual_paths:
    :param pair_paths:
    :return:
    """

    # classifier_weights = classifier_weights.cpu().numpy()
    shape = img_list[0][1].shape[0:2]

    for idx, (att, (title, im)) in enumerate(zip(atts, img_list)):
        # import pdb
        # pdb.set_trace()
        pos_att, neg_att = seperate_pos_neg(att)

        equal = True
        for a, ap in zip(att, pos_att):
            if not torch.equal(a, ap):
                equal = False
                break

        if not equal:  # activations have negatives and positives

            for (a, v) in [(pos_att, 'POS'), (neg_att, 'NEG')]:
                # print(a[0].shape)
                # import pdb
                # pdb.set_trace()
                heatmaps = get_heatmaps(a, shape=shape, classifier_weights=None,
                                        attention=True)  # seperated for normalization, heatmaps withOUT classifier weights
                pics = []
                for h in heatmaps:
                    pics.append(merge_heatmap_img(im, h))

                for layer, pic in enumerate(pics, 1):
                    writer.add_image(tb_path + '_' + v + f'/{title}_layer{layer}', pic, epoch, dataformats='HWC')

        else:
            heatmaps = get_heatmaps(att, shape=shape, classifier_weights=None,
                                    attention=True)  # seperated for normalization, heatmaps withOUT classifier weights
            pics = []
            for h in heatmaps:
                pics.append(merge_heatmap_img(im, h))

            for layer, pic in enumerate(pics, 1):
                writer.add_image(tb_path + f'/{title}_layer{layer}', pic, epoch, dataformats='HWC')

        # pics = [merge_heatmap_img(im, heatmaps[0]),
        #         merge_heatmap_img(im, heatmaps[1]),
        #         merge_heatmap_img(img_list[2][1], heatmaps[2])
        #         ]
    writer.flush()


def get_distances(dists, img_classes):
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

        acts_pos = np.maximum(acts, 0)  # todo fucking it up
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


def get_logname(args):
    id_str = str(datetime.datetime.now()).replace(' ', '_').replace(':', '')
    id_str = '-time_' + id_str.replace('.', '-')

    if args.job_id != '':
        id_str += '_' + args.job_id

    # name = 'model-betteraug-distmlp-' + self.model

    name = f'model-bs{args.batch_size}'

    if args.cuda and args.gpu_ids != '':
        gpus_num = len(args.gpu_ids.split(','))
        gpu_info = f'-{gpus_num}gpus'
    else:
        gpu_info = f'-cpu'

    name += gpu_info

    if args.extra_name != '':
        name += '-' + args.extra_name

    name_replace_dict = {'dataset_name': 'dsn',
                         'batch_size': 'bs',
                         'lr_new': 'lrs',
                         'lr_resnet': 'lrr',
                         # 'early_stopping',
                         'feat_extractor': 'fe',
                         'classifier_layer': 'cl',
                         'projection_layer': 'pl',
                         'normalize': 'normalClas',
                         'number_of_runs': 'nor',
                         'no_negative': 'nn',
                         'margin': 'm',
                         'loss': 'loss',
                         'overfit_num': 'on',
                         'bcecoefficient': 'bco',
                         'trplcoefficient': 'tco',
                         'debug_grad': 'dg',
                         'aug_mask': 'am',
                         'colored_mask': 'colmask',
                         'from_scratch': 'fs',
                         'fourth_dim': 'fd',
                         'image_size': 'igsz',
                         'pooling': 'pool',
                         'merge_method': 'mm',
                         'softmargin': 'softm',
                         'static_size': 'fcsize',
                         'dim_reduction': 'dr',
                         'leaky_relu': 'lrel',
                         'bn_before_classifier': 'bnbc',
                         'weight_decay': 'decay',
                         'drop_last': 'dl',
                         'softmax_diff_sim': 'smds',
                         'feature_map_layers': 'fml',
                         'gamma': 'gamma',
                         'merge_global': 'merge_global',
                         'no_global': 'no_global',
                         'spatial_projection': 'spatial_projection',
                         'attention': 'att',
                         'same_pic_prob': 'spp',
                         'query_index': 'qi',
                         'test_query_index': 'tqi',
                         'classes_in_query': 'ciq',
                         'small_and_big': 'SB',
                         'warm': 'w',
                         'random_erase': 're',
                         'dot-product': 'dp',
                         'dot-product-add': 'dpa'}

    important_args = ['dataset_name',
                      'batch_size',
                      'lr_new',
                      'lr_resnet',
                      # 'early_stopping',
                      'feat_extractor',
                      'classifier_layer',
                      'projection_layer',
                      'normalize',
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
                      'colored_mask',
                      'softmargin',
                      'bn_before_classifier',
                      'leaky_relu',
                      'weight_decay',
                      'drop_last',
                      'softmax_diff_sim',
                      'gamma',
                      'merge_global',
                      'no_global',
                      'dim_reduction',
                      'spatial_projection',
                      'attention',
                      'same_pic_prob',
                      'query_index',
                      'test_query_index',
                      'classes_in_query',
                      'small_and_big',
                      'warm',
                      'random_erase']

    arg_booleans = ['spatial_projection',
                    'attention',
                    'merge_global',
                    'no_global',
                    'drop_last',
                    'normalize',
                    'bn_before_classifier',
                    'leaky_relu',
                    'softmargin',
                    'colored_mask',
                    'fourth_dim',
                    'softmax_diff_sim',
                    'aug_mask',
                    'debug_grad',
                    'from_scratch',
                    'query_index',
                    'test_query_index',
                    'small_and_big',
                    'warm']

    args_shouldnt_be_zero = ['overfit_num',
                             'gamma',
                             'dim_reduction',
                             'same_pic_prob',
                             'random_erase']

    if args.loss != 'bce' and args.loss != 'stopgrad':
        if args.loss == 'contrastive':
            important_args.extend(['margin'])
        else:
            important_args.extend(['trplcoefficient',
                                   'margin',
                                   'classes_in_query'])

    if args.loss != 'bce' and args.loss != 'stopgrad':
        important_args.extend(['margin'])

    for arg in vars(args):
        if str(arg) in important_args:
            if str(arg) in arg_booleans and not getattr(args, arg):
                continue
            elif str(arg) in args_shouldnt_be_zero and getattr(args, arg) == 0:
                continue
            elif str(arg) == 'bcecoefficient' and getattr(args, arg) == 1.0:
                continue


            if type(getattr(args, arg)) is not bool:
                name += '-' + name_replace_dict[str(arg)] + '_' + str(getattr(args, arg))
            else:
                name += '-' + name_replace_dict[str(arg)]

            if str(arg) == 'bcecoefficient':
                name += f'#{args.bcotco_freq}-{args.bco_base}#'
            if str(arg) == 'trplcoefficient':
                name += f'#{args.bcotco_freq}-{args.tco_base}#'

            if str(arg) == 'loss' and getattr(args, arg).startswith('contrastive') and args.reg_lambda != 0.0:
                name += f'-lbd{args.reg_lambda}'

            if str(arg) == 'merge_method' and getattr(args, arg).startswith('local'):
                lays = 'L' + ''.join(args.feature_map_layers)
                att_type = args.att_mode_sc
                if args.att_on_all:
                    att_type += '-all'
                name += f'-{lays}-{att_type}'

            if str(arg) == 'merge_method' and (
                    getattr(args, arg).startswith('diff') or getattr(args, arg).startswith('sim')) and \
                    args.attention:

                lays = 'L' + ''.join(args.feature_map_layers)
                name += f'-att-{lays}'
                if args.add_local_features:
                    name += 'ADD'
                else:
                    name += 'CONC'

            if str(arg) == 'merge_method' and (
                    getattr(args, arg).startswith('diff') or getattr(args, arg).startswith(
                'sim')) and args.att_mode_sc.startswith('dot-product'):
                name += f'-{name_replace_dict[args.att_mode_sc]}'
                if args.att_mode_sc == 'dot-product':
                    name += f'-{args.dp_type}'

            if str(arg) == 'merge_method' and getattr(args, arg).startswith('channel-attention'):
                lays = 'L' + ''.join(args.feature_map_layers)
                att_type = ''
                if args.cross_attention:
                    att_type += '-CA'
                name += f'-{lays}{att_type}'

            if str(arg) == 'gamma':
                name += f'-step{args.gamma_step}'

                if args.gamma_step == 0:
                    if args.lr_adaptive_loss:
                        name += f'LOSS'
                    else:
                        name += f'VAL'
                    name += f'-tol{args.lr_tol}'

    if args.pretrained_model_dir != '':
        name = args.pretrained_model_dir + '_PTRN'
        id_str = ''


    if args.loss == 'batchhard' or args.loss == 'contrastive':
        name += f'-p_{args.bh_P}-k_{args.bh_K}'

    # if args.pretrained_model != '':  # for running baselines and feature extractors
    #     name = f'{args.feat_extractor}_{args.pretrained_model}_{args.extra_name}'

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
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)
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
    return 1 / (1 + np.exp(-x))


def plot_pred_hist(pos_preds, neg_preds, bins=100, title='Pred Histogram', savepath='pred_dist', normalizefactor=9):
    pos_preds = np.repeat(pos_preds, normalizefactor)  # normalize

    b = np.arange(0, 1.01, 1 / bins)

    plt.figure(figsize=(10, 10))
    plt.hist(sigmoid(pos_preds), alpha=0.5, bins=b, color='g')

    lines = [Line2D([0], [0], color="g", lw=4)]

    plt.legend(lines, ['Positive'])
    plt.title(title + 'POS')

    plt.savefig(savepath + '_pos')
    plt.close('all')

    plt.figure(figsize=(10, 10))

    plt.hist(sigmoid(neg_preds), alpha=0.5, bins=b, color='r')

    lines = [Line2D([0], [0], color="r", lw=4)]

    plt.legend(lines, ['Negative'])
    plt.title(title + 'NEG')

    plt.savefig(savepath + '_neg')
    plt.close('all')

    plt.figure(figsize=(10, 10))

    plt.hist(sigmoid(neg_preds), alpha=0.5, bins=b, color='r')
    plt.hist(sigmoid(pos_preds), alpha=0.5, bins=b, color='g')

    lines = [Line2D([0], [0], color="g", lw=4),
             Line2D([0], [0], color="r", lw=4)]

    plt.legend(lines, ['Positive', 'Negative'])
    plt.xlim(-0.01, 1.01)
    plt.title(title)

    plt.savefig(savepath)
    plt.close('all')

def save_representation_hists(representation, savepath):
    representation = representation.cpu().detach().numpy()
    if os.path.exists(savepath):
        previous_reps = np.load(savepath)
        new_reps = np.concatenate([previous_reps, representation], axis=0)
    else:
        new_reps = representation

    np.save(new_reps, new_reps)



def get_pos_neg_preds(file_path, pos_freq=10):
    preds = np.load(file_path)['arr_0']
    pos_mask = np.zeros_like(preds, dtype=bool)
    pos_mask[::pos_freq] = True
    neg_mask = np.logical_not(pos_mask)

    pos_preds = preds[pos_mask]
    neg_preds = preds[neg_mask]

    return pos_preds, neg_preds


def squared_pairwise_distances(embeddings, sqrt=False):
    """
    get dot product (batch_size, batch_size)
    ||a-b||^2 = |a|^2 - 2*<a,b> + |b|^2
    :param embeddings:
    :param squared:
    :return:
    """

    dot_product = embeddings.mm(embeddings.t())

    # a vector
    square_sum = dot_product.diag()

    distances = square_sum.unsqueeze(1) - 2 * dot_product + square_sum.unsqueeze(0)

    distances = distances.clamp(min=0)

    if sqrt:
        distances = (distances + 1e-7).sqrt()

    return distances


def get_valid_positive_mask(labels, gpu=False):
    """
    To be a valid positive pair (a,p),
        - a and p are different embeddings
        - a and p have the same label
    """
    indices_equal = torch.eye(labels.size(0)).byte()
    indices_not_equal = torch.logical_not(indices_equal)

    label_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))

    if gpu:
        indices_not_equal = indices_not_equal.cuda()

    mask = torch.logical_and(indices_not_equal, label_equal)

    return mask


def get_valid_negative_mask(labels, gpu=False):
    """
    To be a valid negative pair (a,n),
        - a and n are different embeddings
        - a and n have the different label
    """
    indices_equal = torch.eye(labels.size(0)).byte()
    indices_not_equal = torch.logical_not(indices_equal)

    label_not_equal = torch.ne(labels.unsqueeze(1), labels.unsqueeze(0))

    if gpu:
        indices_not_equal = indices_not_equal.cuda()

    mask = torch.logical_and(indices_not_equal, label_not_equal)

    return mask


# def get_valid_triplets_mask(labels):
#     """
#     To be valid, a triplet (a,p,n) has to satisfy:
#         - a,p,n are distinct embeddings
#         - a and p have the same label, while a and n have different label
#     """
#     indices_equal = torch.eye(labels.size(0)).byte()
#     indices_not_equal = ~indices_equal
#     i_ne_j = indices_not_equal.unsqueeze(2)
#     i_ne_k = indices_not_equal.unsqueeze(1)
#     j_ne_k = indices_not_equal.unsqueeze(0)
#     distinct_indices = i_ne_j & i_ne_k & j_ne_k
#
#     label_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
#     i_eq_j = label_equal.unsqueeze(2)
#     i_eq_k = label_equal.unsqueeze(1)
#     i_ne_k = torch.logical_not(i_eq_k)
#     valid_labels = torch.logical_and(i_eq_j, i_ne_k)
#
#     mask = torch.logical_and(distinct_indices, valid_labels)
#     return mask
def get_resnet(args, model_name):
    from torchvision.models import resnet50

    pretrained_path = os.path.join(args.project_path, f'models/pretrained_{model_name}.pt')
    state_dict = None
    if os.path.exists(pretrained_path):
        print(f'loading {model_name} from pretrained')
        state_dict = torch.load(pretrained_path)['model_state_dict']
    else:
        raise Exception("Baseline model not found in models/ dir")

    model = resnet50(pretrained=False)
    model.load_state_dict(state_dict)
    model.fc = torch.nn.Linear(2048, 256)

    return model


def plot_class_dist(datas, plottitle, path):
    # import pdb
    # pdb.set_trace()
    labels_unique = []
    labels_count = []
    count_dist_path = path[:path.rfind('.')] + 'count_dist' + path[path.rfind('.'):]

    if not (os.path.exists(path) and os.path.exists(count_dist_path)):
        for k, v in datas.items():
            labels_unique.append(k)
            labels_count.append(len(v))

        labels_count = np.array(labels_count)
        # labels_unique, labels_count = np.unique(labels, return_counts=True)

        # plt.rcParams['figure.dpi'] = 600
        plt.rcParams['figure.figsize'] = (20, 20)
        # x = [i for i in range(len(labels_unique))]
        x = labels_unique
        plt.bar(x, labels_count)
        plt.title(plottitle)
        plt.savefig(path)
        plt.close()

        # plt.rcParams['figure.dpi'] = 600
        plt.rcParams['figure.figsize'] = (20, 20)
        plt.hist(labels_count, bins=(labels_count.max() - labels_count.min()) + 1)
        plt.title(plottitle + ' count dist')
        plt.savefig(count_dist_path)
        plt.close()


# softtriplet loss code
def evaluation(args, X, Y, superY, ids, writer, loader, Kset, split, path, gpu=False, k=5, path_to_lbl2chain='', tb_draw=False,
               metric='cosine', dist_matrix=None, create_best_negatives=True, create_too_close_negatvies=True):
    if superY is None:
        superY = np.array([-1 for _ in Y])

    num = X.shape[0]
    classN = np.max(Y) + 1
    kmax = min(np.max(Kset), num)
    print(f'kmax = {kmax}')
    recallK = np.zeros(len(Kset))
    # compute NMI
    # kmeans = KMeans(n_clusters=classN).fit(X)
    # nmi = normalized_mutual_info_score(Y, kmeans.labels_, average_method='arithmetic')
    # compute Recall@K
    # sim = X.dot(X.T)
    # minval = np.min(sim) - 1.
    # sim -= np.diag(np.diag(sim))
    # sim += np.diag(np.ones(num) * minval)
    start = time.time()
    print(f'**** Evaluation, calculating rank dist: {metric}')
    if dist_matrix is not None:
        num = Y.shape[0]

        minval = np.min(dist_matrix) - 1.
        self_distance = -(np.diag(dist_matrix))
        dist_matrix -= np.diag(np.diag(dist_matrix))
        dist_matrix += np.diag(np.ones(num) * minval)
        indices = (-dist_matrix).argsort()[:, :-1]
        distances = (-dist_matrix)[indices]
    else:
        distances, indices, self_distance = get_faiss_knn(X, k=int(kmax), gpu=gpu, metric=metric)

    print(f'**** Evaluation, calculating dist rank DONE. Took {time.time() - start}s')

    lbl2chain = None
    if create_best_negatives:
        if path_to_lbl2chain != '' and args.negative_path != '':
            super_labels = pd.read_csv(path_to_lbl2chain)
            lbl2chain = {k: v for k, v, in zip(list(super_labels.label), list(super_labels.chain))}
            best_negatives = {}
            for i, (i_row) in enumerate(indices):
                query_label = Y[i]
                query_superlabel = superY[i]
                for j in i_row:
                    negative_idx = j
                    negative_label = Y[j]
                    negative_superlabel = superY[i]
                    if lbl2chain[query_label] != lbl2chain[negative_label]:
                        break
                best_negatives[loader.dataset.all_shuffled_data[i][1]] = (
                    loader.dataset.all_shuffled_data[negative_idx][1], negative_label)

            with open(args.negative_path, 'wb') as f:
                print('new negative set creeated')
                pickle.dump(best_negatives, f)

    label_to_simlabels = {}
    if create_too_close_negatvies:
        for i, (d_row, i_row) in enumerate(zip(distances, indices)):
            leq_indx = i_row[d_row <= self_distance[i]]
            leq_dist = d_row[d_row <= self_distance[i]]
            label_to_simlabels[Y[i]] = {'i': [Y[j] for j in leq_indx if j != i],
                                        'd': leq_dist}

        with open(os.path.join(path, split + '_too_close_otherlabels.pkl'), 'wb') as f:
            pickle.dump(label_to_simlabels, f)

    YNN = Y[indices]
    superYNN = superY[indices]
    idxNN = ids[indices]
    counter = 0
    r1_counter = 0
    r10_counter = 0
    for i in range(0, len(Kset)):
        pos = 0.
        for j in range(0, num):
            if Y[j] in YNN[j, :Kset[i]]:
                pos += 1.
            if tb_draw:
                if Y[j] == YNN[j, 0]:
                    if r1_counter < k:
                        plot_images(ids[j], Y[j], superY[j], idxNN[j, :10], YNN[j, :10], superYNN[j, :10], writer, loader,
                                    f'r@1_{r1_counter}_{split}')
                        r1_counter += 1
                        print('r1_counter = ', r1_counter)
                elif Y[j] in YNN[j, :10]:
                    if r10_counter < k:
                        plot_images(ids[j], Y[j], superY[j], idxNN[j, :10], YNN[j, :10], superYNN[j, :10], writer, loader,
                                    f'r@10_{r10_counter}_{split}')
                        r10_counter += 1
                        print('r10_counter = ', r10_counter)
                elif counter < k:
                    plot_images(ids[j], Y[j], superY[j], idxNN[j, :10], YNN[j, :10], superYNN[j, :10], writer, loader,
                                f'{counter}_{split}')
                    counter += 1
                    print('counter = ', counter)

        recallK[i] = pos / num
    return recallK


def evaluation_qi(args, X_q, X_i, Y_q, Y_i, superY_q, superY_i, ids_q, ids_i, writer, loaders, Kset, split, path, gpu=False, k=5, tb_draw=False,
                  metric='cosine', dist_matrix=None):
    if superY_q is None:
        superY_q = np.array([-1 for _ in Y_q])
        superY_i = np.array([-1 for _ in Y_i])

    num = X_i.shape[0]

    kmax = min(np.max(Kset), num)
    print(f'kmax = {kmax}')
    recallK = np.zeros(len(Kset))

    start = time.time()
    print(f'**** Evaluation, calculating rank dist: {metric}')
    if dist_matrix is not None:

        indices = (-dist_matrix).argsort()[:, :int(kmax)]
        distances = (-dist_matrix)[indices]
    else:
        distances, indices = get_faiss_query_index(X_q, X_i, k=int(kmax), gpu=gpu, metric=metric)

    print(f'**** Evaluation, calculating dist rank DONE. Took {time.time() - start}s')


    YNN = Y_i[indices]
    superYNN = superY_i[indices]
    idxNN = ids_i[indices]
    counter = 0
    r1_counter = 0
    r10_counter = 0
    for i in range(0, len(Kset)):
        pos = 0.
        for j in range(0, Y_q.shape[0]):
            if Y_q[j] in YNN[j, :Kset[i]]:
                pos += 1.
            if tb_draw:
                if Y_q[j] == YNN[j, 0]:
                    if r1_counter < k:
                        plot_images(ids_q[j], Y_q[j], superY_q[j], idxNN[j, :10], YNN[j, :10], superYNN[j, :10], writer, loaders[0],
                                    f'r@1_{r1_counter}_{split}', loader_i=loaders[1])
                        r1_counter += 1
                        print('r1_counter = ', r1_counter)
                elif Y_q[j] in YNN[j, :10]:
                    if r10_counter < k:
                        plot_images(ids_q[j], Y_q[j], superY_q[j], idxNN[j, :10], YNN[j, :10], superYNN[j, :10], writer, loaders[0],
                                    f'r@10_{r10_counter}_{split}', loader_i=loaders[1])
                        r10_counter += 1
                        print('r10_counter = ', r10_counter)
                elif counter < k:
                    plot_images(ids_q[j], Y_q[j], superY_q[j], idxNN[j, :10], YNN[j, :10], superYNN[j, :10], writer, loaders[0],
                                f'{counter}_{split}', loader_i=loaders[1])
                    counter += 1
                    print('counter = ', counter)

        recallK[i] = pos / num
    return recallK



def plot_images(org_idx, org_lbl, sup_lbl, top_10_indx, top_10_lbl, top_10_super_lbl, writer, loader_q, tb_label, loader_i=None):
    if loader_i is None:
        loader_i = loader_q

    writer.add_image(tb_label + f'/0_q_class_C{org_lbl}_{sup_lbl}',
                     get_image_from_dataloader(loader_q, org_idx),
                     0, dataformats='CHW')

    for i in range(len(top_10_lbl)):
        writer.add_image(tb_label + f'/{i + 1}_class_C{top_10_lbl[i]}_{top_10_super_lbl[i]}',
                         get_image_from_dataloader(loader_i, top_10_indx[i]),
                         0, dataformats='CHW')

    writer.flush()


# testing merge
def get_image_from_dataloader(loader, index):
    img = Image.open(loader.dataset.all_shuffled_data[int(index)][1]).convert('RGB')
    img = loader.dataset.transform(img).numpy()
    return img


def get_attention_normalized(reps, chunks):
    rep_chunks = np.split(reps, chunks, axis=1)
    final_reps = []
    for rep in rep_chunks:
        rep = np.array(rep, order='C')
        faiss.normalize_L2(rep)
        final_reps.append(rep)

    reps = np.concatenate(final_reps, axis=1)
    return reps

def get_faiss_query_index(reps_q, reps_i, k=1000, gpu=False, metric='cosine'):  # method "cosine" or "euclidean"
    assert reps_q.dtype == np.float32
    assert reps_i.dtype == np.float32

    print(f'get_faiss_knn metric is: {metric}')

    d = reps_q.shape[1]
    if metric == 'euclidean':
        index_function = faiss.IndexFlatL2
    elif metric == 'cosine':
        index_function = faiss.IndexFlatIP
    else:
        index_function = None
        raise Exception(f'get_faiss_knn unsupported method {metric}')

    if gpu:
        try:
            index_flat = index_function(d)
            res = faiss.StandardGpuResources()
            index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
            index_flat.add(reps_i)  # add vectors to the index
            print('Using GPU for KNN!!'
                  ' Thanks FAISS!')
        except:
            print('Didn\'t fit it GPU, No gpus for faiss! :( ')
            index_flat = index_function(d)
            index_flat.add(reps_i)  # add vectors to the index
    else:
        print('No gpus for faiss! :( ')
        index_flat = index_function(d)
        index_flat.add(reps_i)  # add vectors to the index

    assert (index_flat.ntotal == reps_i.shape[0])

    D, I = index_flat.search(reps_q, k)


    return D, I

def get_faiss_knn(reps, k=1000, gpu=False, metric='cosine'):  # method "cosine" or "euclidean"
    assert reps.dtype == np.float32

    print(f'get_faiss_knn metric is: {metric}')

    d = reps.shape[1]
    if metric == 'euclidean':
        index_function = faiss.IndexFlatL2
    elif metric == 'cosine':
        index_function = faiss.IndexFlatIP
    else:
        index_function = None
        raise Exception(f'get_faiss_knn unsupported method {metric}')

    if gpu:
        try:
            index_flat = index_function(d)
            res = faiss.StandardGpuResources()
            index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
            index_flat.add(reps)  # add vectors to the index
            print('Using GPU for KNN!!'
                  ' Thanks FAISS!')
        except:
            print('Didn\'t fit it GPU, No gpus for faiss! :( ')
            index_flat = index_function(d)
            index_flat.add(reps)  # add vectors to the index
    else:
        print('No gpus for faiss! :( ')
        index_flat = index_function(d)
        index_flat.add(reps)  # add vectors to the index

    assert (index_flat.ntotal == reps.shape[0])

    D, I = index_flat.search(reps, k)

    D_notself = []
    I_notself = []

    self_distance = []

    start = time.time()
    for i, (i_row, d_row) in enumerate(zip(I, D)):
        self_distance.append(d_row[np.where(i_row == i)])
        I_notself.append(np.delete(i_row, np.where(i_row == i)))
        D_notself.append(np.delete(d_row, np.where(i_row == i)))
    end = time.time()

    self_D = np.array(self_distance)
    D = np.array(D_notself)
    I = np.array(I_notself)

    print(f'D and I cleaning time: {end - start}')

    return D, I, self_D


def save_knn(embbeddings, path, gpu=False, metric='cosine'):
    import pickle
    make_dirs(path=path)
    distances, indicies, _ = get_faiss_knn(embbeddings, gpu=gpu, metric=metric)
    with open(os.path.join(f'{path}', 'indicies.pkl'), 'wb') as f:
        pickle.dump(indicies, f)

    with open(os.path.join(f'{path}', 'embeddings.pkl'), 'wb') as f:
        pickle.dump(indicies, f)

    return


def calc_custom_cosine_sim(feat1, feat2, agg='mean', weights=[]):
    sims = torch.zeros(size=(feat1[0].shape[0], len(feat1)))
    for idx, (f1, f2) in enumerate(zip(feat1, feat2)):
        sims[:, idx] = torch.nn.functional.cosine_similarity(f1, f2)

    return sims.mean(dim=1)


def calc_custom_euc(feat, chunks=4):
    feats = np.split(feat, chunks, axis=1)
    eucs = []
    for f in feats:
        eucs.append(euclidean_distances(f))
    return sum(eucs)


def draw_top_results(args, embeddings, labels, superlabels, ids, seens, data_loader, tb_writer, save_path, metric='cosine', k=5,
                     dist_matrix=None, best_negative=False, too_close_negative=False):
    unique_seens = np.unique(seens)
    if len(unique_seens) == 2:
        superlabels_0 = superlabels[seens == 0] if superlabels is not None else None
        superlabels_1 = superlabels[seens == 1] if superlabels is not None else None
        seen_res = evaluation(args, embeddings[seens == 1], labels[seens == 1], superlabels_1,
                              ids[seens == 1], tb_writer, data_loader,
                              Kset=[1, 2, 4, 5, 8, 10, 100, 1000], split='seen', path=save_path, k=k,
                              gpu=args.cuda, metric=metric, dist_matrix=dist_matrix, tb_draw=True,
                              create_best_negatives=best_negative, create_too_close_negatvies=too_close_negative)
        print(f'Seen length: {len(labels[seens == 1])}')
        print(f'K@1, K@2, K@4, K@5, K@8, K@10, K@100, K@1000')
        print(seen_res)
        unseen_res = evaluation(args, embeddings[seens == 0], labels[seens == 0], superlabels_0,
                                ids[seens == 0], tb_writer, data_loader,
                                Kset=[1, 2, 4, 5, 8, 10, 100, 1000], split='unseen', path=save_path, k=k,
                                gpu=args.cuda, metric=metric, dist_matrix=dist_matrix, tb_draw=True,
                                create_best_negatives=best_negative, create_too_close_negatvies=too_close_negative)
        print(f'Unseen length: {len(labels[seens == 0])}')
        print(f'K@1, K@2, K@4, K@5, K@8, K@10, K@100, K@1000')
        print(unseen_res)

        res = evaluation(args, embeddings, labels, superlabels, ids, tb_writer,
                         data_loader, Kset=[1, 2, 4, 5, 8, 10, 100, 1000], split='total', path=save_path, k=k,
                         gpu=args.cuda, metric=metric, dist_matrix=dist_matrix, tb_draw=True,
                         create_best_negatives=best_negative, create_too_close_negatvies=too_close_negative)
        print(f'Total length: {len(labels)}')
        print(f'K@1, K@2, K@4, K@5, K@8, K@10, K@100, K@1000')
        print(res)

    elif len(unique_seens) == 1:
        res = evaluation(args, embeddings, labels, superlabels, ids, tb_writer,
                         data_loader, Kset=[1, 2, 4, 5, 8, 10, 100, 1000], split='total', path=save_path, k=k,
                         gpu=args.cuda, metric=metric, dist_matrix=dist_matrix,
                         path_to_lbl2chain=os.path.join(args.splits_file_path, 'label2chain.csv'),
                         tb_draw=True, create_best_negatives=best_negative,
                         create_too_close_negatvies=too_close_negative)
        print(f'Total length: {len(labels)}')
        print(f'K@1, K@2, K@4, K@5, K@8, K@10, K@100, K@1000')
        print(res)
    else:
        raise Exception(f"More than 2 values in 'seens'. len(unique_seens) = {len(unique_seens)}")

def draw_top_results_qi(args, embeddings, labels, superlabels, ids, seens, data_loaders, tb_writer, save_path, metric='cosine', k=5,
                     dist_matrix=None):

    res = evaluation_qi(args, embeddings[0], embeddings[1], labels[0], labels[1], superlabels[0], superlabels[1], ids[0], ids[1], tb_writer,
                     data_loaders, Kset=[1, 2, 4, 5, 8, 10, 100, 1000], split=data_loaders[0].dataset.name, path=save_path, k=k,
                     gpu=args.cuda, metric=metric, dist_matrix=dist_matrix,
                     tb_draw=True)
    print(f'Total length: {len(labels)}')
    print(f'K@1, K@2, K@4, K@5, K@8, K@10, K@100, K@1000')
    print(res)


def get_sampled_query_index(query_feats, index_feats, query_labels, index_labels, thresh=6, min_index_perclass=3, classes=0):
    ilbls, ilbls_c = np.unique(index_labels, return_counts=True)

    if ilbls_c.mean() > thresh:
        above_thresh_lbls = ilbls[ilbls_c >= thresh]
    else:
        above_thresh_lbls = ilbls

    # df_query_above_thresh = query_labels[query_labels.isin(above_thresh_lbls)]
    # df_index_above_thresh = index_labels[index_labels.label.isin(above_thresh_lbls)]

    imask_on_all_abovethresh = [i for i in range(len(index_labels))]

    qmask_on_all_abovethresh = [i for i in range(len(query_labels))]

    sampled_index_masks = np.array([])
    sampled_query_masks = np.array([])

    if classes != 0:
        above_thresh_lbls = np.random.choice(above_thresh_lbls, classes, replace=False) # should raise error if classes != 0 and more than the number

    for l in above_thresh_lbls:
        i_relevant_masks = np.array(imask_on_all_abovethresh)[index_labels == l]
        q_relevant_masks = np.array(qmask_on_all_abovethresh)[query_labels == l]

        sampled_index_masks = np.append(sampled_index_masks,
                                        np.random.choice(i_relevant_masks, min_index_perclass, replace=False))
        sampled_query_masks = np.append(sampled_query_masks,
                                        np.random.choice(q_relevant_masks, (min_index_perclass // 3), replace=False))

    sampled_index_masks = sampled_index_masks.astype(int)
    sampled_index = index_feats[sampled_index_masks, :]
    sampled_index_lbls = index_labels[sampled_index_masks]

    sampled_query_masks = sampled_query_masks.astype(int)
    sampled_query = query_feats[sampled_query_masks, :]
    sampled_query_lbls = query_labels[sampled_query_masks]


    return sampled_query, sampled_index, sampled_query_lbls, sampled_index_lbls

def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        for param_group in optimizer.param_groups:
            if param_group['new']:
                warmup_to = args.lr_new
                warmup_from = args.lr_new * 0.01
            else:
                warmup_to = args.lr_resnet
                warmup_from = args.lr_resnet * 0.01

            p = ((batch_id - 1) + (epoch - 1) * total_batches) / \
                (args.warm_epochs * total_batches)
            lr = warmup_from + p * (warmup_to - warmup_from)

            param_group['lr'] = lr