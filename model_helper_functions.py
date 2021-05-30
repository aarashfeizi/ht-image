import collections
import os
import time
from collections import deque

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from torch.autograd import Variable
from tqdm import tqdm

import metrics
import models.top_model
import utils
# from torch.utils.tensorboard import SummaryWriter
from Tensorboard_Writer import SummaryWriter
# def to_numpy_axis_order_change(t):
#     t = t.numpy()
#     t = np.moveaxis(t.squeeze(), 0, -1)
#     return t
from losses import TripletLoss

EVAL_SET_NAMES = {1: ['total'],
                  2: ['seen', 'unseen']}


class Adaptive_Scheduler:
    def __init__(self, opt, gamma, tol=3, logger=None, val=True, loss=False, min_lr=1e-6):
        if val == loss:
            raise Exception("val and loss can't be true (or false) together")
        self.opt = opt
        self.gamma = gamma
        self.metric = -1
        self.max_tol = tol
        self.level = 0
        self.logger = logger
        self.min_lr = min_lr

        self.val_mode = val
        self.loss_mode = loss

    def reduce_lr(self, lr):
        lr *= self.gamma
        if lr >= self.min_lr:
            return lr
        else:
            if self.logger is not None:
                self.logger.info(f'min_lr reached = {self.min_lr}')
            return self.min_lr

    def step(self, current_loss, current_val):
        if self.metric == -1:
            self.metric = current_loss if self.loss_mode else current_val
            return

        if self.loss_mode:  # metric is a loss value
            if current_loss > self.metric or np.abs(current_loss - self.metric) <= self.metric * 0.005:
                if self.logger is not None:
                    self.logger.info(
                        f"level = {self.level}, last loss = {self.metric}, current loss = {current_loss}, eps = {self.metric * 0.005}")
                self.level += 1
            elif current_loss < self.metric and np.abs(current_loss - self.metric) > self.metric * 0.005:
                self.metric = current_loss
                self.level = 0

        elif self.val_mode:
            if current_val < self.metric:
                if self.logger is not None:
                    self.logger.info(
                        f"level = {self.level}, last total val = {self.metric}, current total val = {current_val}")
                self.level += 1
            elif current_val >= self.metric:
                self.metric = current_val
                self.level = 0
        else:
            raise Exception('Error in adaptive learning rate decay')

        if self.level == self.max_tol:
            for p in self.opt.param_groups:
                if self.logger is not None:
                    self.logger.info(
                        f"Tol = {self.max_tol} and previous loss = {self.metric} Decaying learning rate from {p['lr']} to {p['lr'] * self.gamma}")

                p['lr'] = self.reduce_lr(p['lr'])

            self.level = 0
            self.metric = current_loss if self.loss_mode else current_val


class ModelMethods:

    def __init__(self, args, logger, model='top', cam_images_len=-1, model_name='', id_str=''):  # res or top

        self.merge_global = args.merge_global
        self.model = model

        self.model_name = model_name
        self.colored_mask = args.colored_mask

        self.bcotco_freq = args.bcotco_freq
        self.bco_base = args.bco_base
        self.tco_base = args.tco_base

        self.no_negative = args.no_negative

        weight_sum = args.bcecoefficient + args.trplcoefficient

        self.bce_weight = args.bcecoefficient / weight_sum
        self.trpl_weight = args.trplcoefficient / weight_sum

        self.draw_all_thresh = args.draw_all_thresh

        utils.make_dirs(os.path.join(args.local_path, args.tb_path))

        self.tensorboard_path = os.path.join(args.local_path, args.tb_path, self.model_name)
        self.logger = logger
        self.writer = SummaryWriter(self.tensorboard_path)

        if args.pretrained_model_dir == '':
            utils.make_dirs(os.path.join(args.local_path, args.save_path))
            self.save_path = os.path.join(args.local_path, args.save_path, self.model_name)
            utils.create_save_path(self.save_path, id_str, self.logger)
        else:
            self.logger.info(f"Using pretrained path... \nargs.pretrained_model_dir: {args.pretrained_model_dir}")
            self.save_path = os.path.join(args.save_path, args.pretrained_model_dir)

        self.logger.info("** Save path: " + self.save_path)
        self.logger.info("** Tensorboard path: " + self.tensorboard_path)

        self.merge_method = args.merge_method
        self.logger.info(f'Merging with {self.merge_method}')

        if 'attention' in self.merge_method:
            self.metric = 'cosine'
        else:
            self.metric = 'euclidean'
        self.logger.info(f'Metric is {self.metric}')

        if args.debug_grad:
            self.draw_grad = True
            self.plt_save_path = f'{self.save_path}/loss_plts/'
            utils.make_dirs(self.plt_save_path)
        else:
            self.draw_grad = False
            self.plt_save_path = ''

        self.created_image_heatmap_path = False

        self.gen_plot_path = f'{self.save_path}/plots/'

        self.hparams_metric = {}

        utils.make_dirs(self.gen_plot_path)
        utils.make_dirs(os.path.join(self.gen_plot_path, f'{args.dataset_name}_train'))
        utils.make_dirs(os.path.join(self.gen_plot_path, f'{args.dataset_name}_val'))

        if args.negative_path != '':
            args.negative_path = os.path.join(self.save_path, args.negative_path)
            utils.make_dirs(os.path.split(args.negative_path)[0])

        if args.cam:
            utils.make_dirs(f'{self.save_path}/heatmap/')

            self.cam_all = 0
            self.cam_neg = np.array([0 for _ in range(cam_images_len)])
            self.cam_pos = np.array([0 for _ in range(cam_images_len)])

        self.class_diffs = {'train':
                                {'between_class_average': [],
                                 'between_class_min': [],
                                 'between_class_max': [],
                                 'in_class_average': [],
                                 'in_class_min': [],
                                 'in_class_max': []},
                            'val':
                                {'between_class_average': [],
                                 'between_class_min': [],
                                 'between_class_max': [],
                                 'in_class_average': [],
                                 'in_class_min': [],
                                 'in_class_max': []},
                            'val_seen':
                                {'between_class_average': [],
                                 'between_class_min': [],
                                 'between_class_max': [],
                                 'in_class_average': [],
                                 'in_class_min': [],
                                 'in_class_max': []},
                            'val_unseen':
                                {'between_class_average': [],
                                 'between_class_min': [],
                                 'between_class_max': [],
                                 'in_class_average': [],
                                 'in_class_min': [],
                                 'in_class_max': []}}

        self.silhouette_scores = {'train': [],
                                  'val': [],
                                  'val_seen': [],
                                  'val_unseen': [],
                                  'test': []}

        self.aug_mask = args.aug_mask

        if self.aug_mask:
            masks = utils.get_masks(args.dataset_path, args.dataset_folder,
                                    os.path.join(args.project_path, args.mask_path))
            self.anch_mask = Image.open(masks[np.random.randint(len(masks))])
            self.pos_mask = Image.open(masks[np.random.randint(len(masks))])
            self.neg_mask = Image.open(masks[np.random.randint(len(masks))])
            self.anch_offsets = None
            self.anch_resizefactors = None
            self.pos_offsets = None
            self.pos_resizefactors = None
            self.neg_offsets = None
            self.neg_resizefactors = None

    def _tb_project_embeddings(self, args, net, loader, k):

        net.eval()
        # device = f'cuda:{net.device_ids[0]}'

        imgs, lbls = loader.dataset.get_k_samples(k)

        lbls = list(map(lambda x: x.argmax(), lbls))

        imgs = torch.stack(imgs)
        # lbls = torch.stack(lbls)
        self.logger.info(f'imgs.shape {imgs.shape}')
        if args.cuda:
            imgs_c = Variable(imgs.cuda())
        else:
            imgs_c = Variable(imgs)

        features, logits = net.forward(imgs_c, is_feat=True)
        feats = features[-1]

        self.logger.info(f'feats.shape {feats.shape}')

        self.writer.add_embedding(mat=feats.view(k, -1), metadata=lbls, label_img=imgs)
        self.writer.flush()

    def _tb_draw_histograms(self, args, net, epoch):

        for name, param in net.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(name, param.flatten(), epoch)

        self.writer.flush()

    @utils.MY_DEC
    def draw_heatmaps(self, net, loss_fn, bce_loss, args, cam_loader, transform_for_model=None,
                      transform_for_heatmap=None, epoch=0, count=1, draw_all_thresh=32):

        net.eval()
        # device = f'cuda:{net.device_ids[0]}'
        heatmap_path = f'{self.save_path}/heatmap/'
        heatmap_path_perepoch = os.path.join(heatmap_path, f'epoch_{epoch}/')

        utils.make_dirs(heatmap_path_perepoch)
        self.cam_all += 1
        if 'attention' in self.merge_method:
            if self.merge_method == 'local-ds-attention':
                sub_methods = []
                for i in args.feature_map_layers:
                    sub_methods += [f'Layer {i} diff']
                    sub_methods += [f'Layer {i} sim']
            else:
                sub_methods = [f'Layer {i}' for i in args.feature_map_layers]

        else:
            sub_methods = self.merge_method.split('-')

        classifier_weights = net.get_classifier_weights().data[0]
        classifier_dim = len(classifier_weights)
        classifier_histogram_path = os.path.join(heatmap_path_perepoch,
                                                 f'classifier_histogram_epoch{epoch}.png')

        self.plot_classifier_hist(classifier_weights.chunk(len(sub_methods), dim=-1), sub_methods,
                                  'Classifier weight distribution', classifier_histogram_path,
                                  f'classifier_weight', epoch)

        for id, (anch_path, pos_path, neg_path) in enumerate(cam_loader, 1):

            self.logger.info(f'Anch path: {anch_path}')
            self.logger.info(f'Pos path: {pos_path}')
            self.logger.info(f'Neg path: {neg_path}')

            heatmap_path_perepoch_id = os.path.join(heatmap_path_perepoch, f'triplet_{id}')

            utils.make_dirs(heatmap_path_perepoch_id)

            anch = Image.open(anch_path)
            pos = Image.open(pos_path)
            neg = Image.open(neg_path)

            if self.aug_mask:
                anch, masked_anch, anch_mask, param_anch = utils.add_mask(anch, self.anch_mask,
                                                                          offsets=self.anch_offsets,
                                                                          resize_factors=self.anch_resizefactors,
                                                                          colored=self.colored_mask)
                pos, masked_pos, pos_mask, param_pos = utils.add_mask(pos, self.pos_mask, offsets=self.pos_offsets,
                                                                      resize_factors=self.pos_resizefactors,
                                                                      colored=self.colored_mask)
                neg, masked_neg, neg_mask, param_neg = utils.add_mask(neg, self.neg_mask, offsets=self.neg_offsets,
                                                                      resize_factors=self.neg_resizefactors,
                                                                      colored=self.colored_mask)

                if not args.fourth_dim:
                    anch = masked_anch
                    pos = masked_pos
                    neg = masked_neg

                if self.anch_offsets is None:
                    self.anch_offsets = param_anch['offsets']
                    self.anch_resizefactors = param_anch['resize_factors']

                    self.pos_offsets = param_pos['offsets']
                    self.pos_resizefactors = param_pos['resize_factors']

                    self.neg_offsets = param_neg['offsets']
                    self.neg_resizefactors = param_neg['resize_factors']

            tl = utils.TransformLoader(228)

            anch = tl.transform_normalize(transform_for_model(anch))
            pos = tl.transform_normalize(transform_for_model(pos))
            neg = tl.transform_normalize(transform_for_model(neg))

            anch = anch.reshape(shape=(1, anch.shape[0], anch.shape[1], anch.shape[2]))
            pos = pos.reshape(shape=(1, pos.shape[0], pos.shape[1], pos.shape[2]))
            neg = neg.reshape(shape=(1, neg.shape[0], neg.shape[1], neg.shape[2]))

            if self.aug_mask:
                anch_org = np.asarray(transform_for_heatmap(masked_anch))
                pos_org = np.asarray(transform_for_heatmap(masked_pos))
                neg_org = np.asarray(transform_for_heatmap(masked_neg))
            else:
                anch_org = np.asarray(transform_for_heatmap(Image.open(anch_path)))
                pos_org = np.asarray(transform_for_heatmap(Image.open(pos_path)))
                neg_org = np.asarray(transform_for_heatmap(Image.open(neg_path)))

            zero_labels = torch.tensor([0], dtype=float)
            one_labels = torch.tensor([1], dtype=float)

            if args.cuda:
                anch, pos, neg, one_labels, zero_labels = Variable(anch.cuda()), \
                                                          Variable(pos.cuda()), \
                                                          Variable(neg.cuda()), \
                                                          Variable(one_labels.cuda()), \
                                                          Variable(zero_labels.cuda())
            else:
                anch, pos, neg, one_labels, zero_labels = Variable(anch), \
                                                          Variable(pos), \
                                                          Variable(neg), \
                                                          Variable(one_labels), \
                                                          Variable(zero_labels)

            class_loss = 0
            ext_loss = 0

            pos_pred, pos_dist, anch_feat, pos_feat, acts_anch_pos, anchp_att, pos_att = net.forward(anch, pos,
                                                                                                     feats=True,
                                                                                                     hook=True,
                                                                                                     return_att=True)

            map_shape = acts_anch_pos[0].shape
            classifier_weights_tensor = torch.repeat_interleave(classifier_weights, repeats=map_shape[2] * map_shape[3],
                                                                dim=0).view(map_shape[0], classifier_dim, map_shape[2],
                                                                            map_shape[3])

            # acts_anch_pos[0] *= classifier_weights
            # acts_anch_pos[1] *= classifier_weights
            #
            # acts_anch_pos[0] = np.maximum(acts_anch_pos[0], 0)
            # acts_anch_pos[1] = np.maximum(acts_anch_pos[1], 0)

            # self.logger.info(f'cam pos {id - 1}: ', torch.sigmoid(pos_pred).item())
            pos_pred_int = int(torch.sigmoid(pos_pred).item() >= 0.5)
            self.cam_pos[id - 1] += pos_pred_int
            pos_text = "Correct" if pos_pred_int == 1 else "Wrong"
            plot_title = f'Backward BCE heatmaps Anch Pos\nAnch-Pos: {pos_text}'
            if 'diff' in self.merge_method or 'sim' in self.merge_method:
                pos_class_loss = bce_loss(pos_pred.squeeze(), one_labels.squeeze())
                pos_class_loss.backward(retain_graph=True)
                class_loss = pos_class_loss

                utils.apply_grad_heatmaps(net.get_activations_gradient(),
                                          net.get_activations().detach(),
                                          {'anch': anch_org,
                                           'pos': pos_org}, 'bce_anch_pos', id, heatmap_path_perepoch_id,
                                          plot_title, f'triplet_{id}_anchpos_bce', epoch, self.writer)

            neg_pred, neg_dist, _, neg_feat, acts_anch_neg, anchn_att, neg_att = net.forward(anch, neg,
                                                                                             feats=True,
                                                                                             hook=True,
                                                                                             return_att=True)

            neg_pred_int = int(torch.sigmoid(neg_pred).item() < 0.5)
            self.cam_neg[id - 1] += neg_pred_int

            neg_text = "Correct" if neg_pred_int == 1 else "Wrong"
            plot_title = f'Backward BCE heatmaps Anch Neg\nAnch-Neg: {neg_text}'

            if 'diff' in self.merge_method or 'sim' in self.merge_method:
                neg_class_loss = bce_loss(neg_pred.squeeze(), zero_labels.squeeze())
                neg_class_loss.backward(retain_graph=True)
                class_loss += neg_class_loss
                utils.apply_grad_heatmaps(net.get_activations_gradient(),
                                          net.get_activations().detach(),
                                          {'anch': anch_org,
                                           'neg': neg_org}, 'bce_anch_neg', id, heatmap_path_perepoch_id,
                                          plot_title, f'triplet_{id}_anchneg_bce', epoch, self.writer)

            # utils.draw_all_heatmaps(acts_anch_pos[0], anch_org, 'Anch', all_heatmap_grid_anch_path)
            # utils.draw_all_heatmaps(acts_anch_pos[1], pos_org, 'Pos', all_heatmap_grid_pos_path)
            # utils.draw_all_heatmaps(acts_anch_neg[1], neg_org, 'Neg', all_heatmap_grid_neg_path)

            # all_heatmap_grid_path = os.path.join(heatmap_path_perepoch_id, f'triplet{id}_all_heatmaps.pdf')
            # utils.draw_all_heatmaps([acts_anch_pos[0],
            #                          acts_anch_pos[1],
            #                          acts_anch_neg[1]],
            #                         [anch_org, pos_org, neg_org],
            #                         ['Anch', 'Pos', 'Neg'],
            #                         all_heatmap_grid_path)

            # self.logger.info('neg_pred', torch.sigmoid(neg_pred))

            result_text = f'\nAnch-Pos: {pos_text}\nAnch-Neg: {neg_text}'

            if 'diff' in self.merge_method or 'sim' in self.merge_method:
                ks = list(map(lambda x: int(x), args.k_best_maps))

                value = ''
                if self.merge_method == 'diff':
                    value = 'different'
                elif self.merge_method == 'sim':
                    value = 'similar'
                elif self.merge_method == 'diff-sim':
                    value = 'different and similar'
                else:
                    raise Exception(f'Merge method not recognized {self.merge_method}')
                #
                # import pdb
                # pdb.set_trace()

                pos_dist_weighted = pos_dist * classifier_weights
                neg_dist_weighted = neg_dist * classifier_weights
                sub_method_dim = classifier_dim / len(sub_methods)

                for k in ks:
                    acts_tmp = []

                    for m_i, met in enumerate(sub_methods):

                        # import pdb
                        # pdb.set_trace()
                        offset = int(m_i * sub_method_dim)
                        begin_index = offset
                        end_index = int(offset + sub_method_dim)
                        pos_dist_weighted_temp = pos_dist_weighted[:, begin_index:end_index]
                        neg_dist_weighted_temp = neg_dist_weighted[:, begin_index:end_index]
                        pos_max_indices = torch.topk(pos_dist_weighted_temp, k=k).indices

                        print(f'offset = {offset}')

                        self.logger.info(
                            f'pos max indices {met}: {pos_max_indices}, {pos_dist_weighted_temp[0][pos_max_indices]}')
                        print(f'pos max indices {met}: {pos_max_indices}, {pos_dist_weighted_temp[0][pos_max_indices]}')

                        acts_tmp = []

                        acts_tmp.append(acts_anch_pos[0][:, pos_max_indices, :, :].squeeze(dim=0))
                        acts_tmp.append(acts_anch_pos[1][:, pos_max_indices, :, :].squeeze(dim=0))
                        acts_tmp.append(acts_anch_neg[1][:, pos_max_indices, :, :].squeeze(dim=0))

                        all_heatmap_path = {
                            'max': os.path.join(heatmap_path_perepoch_id,
                                                f'max_k_{k}_triplet{id}_best_anchpos_{met}.png'),
                            'avg': os.path.join(heatmap_path_perepoch_id,
                                                f'avg_k_{k}_triplet{id}_best_anchpos_{met}.png')}

                        histogram_path = os.path.join(heatmap_path_perepoch_id,
                                                      f'k_{k}_histogram_triplet{id}_best_anchpos_{met}.png')

                        plot_title_wo_weights = f"{k} most important {value} channels for Anch Pos (w/o weights mul {met})" + result_text
                        plot_title = f"{k} most important {value} channels for Anch Pos (with weights mul {met})" + result_text

                        if k < draw_all_thresh:
                            all_heatmap_grid_path = os.path.join(heatmap_path_perepoch_id,
                                                                 f'k_{k}_triplet{id}_all_heatmaps_best_anchpos_{met}.pdf')
                            self.logger.info('before')
                            self.logger.info(str(acts_tmp[0].min()))
                            self.logger.info(str(acts_tmp[1].min()))
                            self.logger.info(str(acts_tmp[2].min()))
                            utils.draw_all_heatmaps(acts_tmp,
                                                    [anch_org, pos_org, neg_org],
                                                    ['Anch', 'Pos', 'Neg'],
                                                    all_heatmap_grid_path,
                                                    plot_title_wo_weights)
                            self.logger.info('after')
                            self.logger.info(str(acts_tmp[0].min()))
                            self.logger.info(str(acts_tmp[1].min()))
                            self.logger.info(str(acts_tmp[2].min()))
                            self.logger.info('-------------')

                        self.logger.info('before forward drawing')
                        self.logger.info(f'min: {(acts_tmp[0].min())}')
                        self.logger.info(f'min: {(acts_tmp[1].min())}')
                        self.logger.info(f'min: {(acts_tmp[2].min())}')

                        self.logger.info(f'max: {(acts_tmp[0].max())}')
                        self.logger.info(f'max: {(acts_tmp[1].max())}')
                        self.logger.info(f'max: {(acts_tmp[2].max())}')

                        utils.apply_forward_heatmap(acts_tmp,
                                                    [('anch', anch_org), ('pos', pos_org), ('neg', neg_org)],
                                                    id,
                                                    all_heatmap_path,
                                                    overall_title=plot_title,
                                                    # individual_paths=[anch_hm_file_path,
                                                    #                   pos_hm_file_path],
                                                    # pair_paths=[anchpos_anch_hm_file_path, anchpos_pos_hm_file_path],
                                                    titles=['Anch', 'Pos', 'Neg'],
                                                    histogram_path=histogram_path,
                                                    merge_method=met,
                                                    classifier_weights=classifier_weights_tensor[:,
                                                                       offset + pos_max_indices, :,
                                                                       :].squeeze(dim=0),
                                                    tb_path=f'triplet_{id}_anch_pos_forward', epoch=epoch,
                                                    writer=self.writer)

                        self.logger.info('after forward drawing')
                        self.logger.info(f'min: {(acts_tmp[0].min())}')
                        self.logger.info(f'min: {(acts_tmp[1].min())}')
                        self.logger.info(f'min: {(acts_tmp[2].min())}')

                        self.logger.info(f'max: {(acts_tmp[0].max())}')
                        self.logger.info(f'max: {(acts_tmp[1].max())}')
                        self.logger.info(f'max: {(acts_tmp[2].max())}')

                        self.logger.info('-------------')
                        # import pdb
                        # pdb.set_trace()

                        acts_tmp = []

                        neg_max_indices = torch.topk(neg_dist_weighted_temp, k=k).indices

                        all_heatmap_path = {
                            'max': os.path.join(heatmap_path_perepoch_id,
                                                f'max_k_{k}_triplet{id}_best_anchneg_{met}.png'),
                            'avg': os.path.join(heatmap_path_perepoch_id,
                                                f'avg_k_{k}_triplet{id}_best_anchneg_{met}.png')}
                        histogram_path = os.path.join(heatmap_path_perepoch_id,
                                                      f'k_{k}_histogram_triplet{id}_best_anchneg_{met}.png')

                        acts_tmp.append(acts_anch_pos[0][:, neg_max_indices, :, :].squeeze(dim=0))
                        acts_tmp.append(acts_anch_pos[1][:, neg_max_indices, :, :].squeeze(dim=0))
                        acts_tmp.append(acts_anch_neg[1][:, neg_max_indices, :, :].squeeze(dim=0))

                        plot_title_wo_weights = f"{k} most important {value} channels for Anch Neg (w/o weights mul {met})" + result_text
                        plot_title = f"{k} most important {value} channels for Anch Neg (with weights mul {met})" + result_text

                        if k < draw_all_thresh:
                            all_heatmap_grid_path = os.path.join(heatmap_path_perepoch_id,
                                                                 f'k_{k}_triplet{id}_all_heatmaps_best_anchneg_{met}.pdf')
                            utils.draw_all_heatmaps(acts_tmp,
                                                    [anch_org, pos_org, neg_org],
                                                    ['Anch', 'Pos', 'Neg'],
                                                    all_heatmap_grid_path,
                                                    plot_title_wo_weights)

                        utils.apply_forward_heatmap(acts_tmp,
                                                    [('anch', anch_org), ('pos', pos_org), ('neg', neg_org)],
                                                    id,
                                                    all_heatmap_path,
                                                    overall_title=plot_title,
                                                    # individual_paths=[anch_hm_file_path,
                                                    #                   neg_hm_file_path],
                                                    # pair_paths=[anchneg_anch_hm_file_path, anchneg_neg_hm_file_path],
                                                    titles=['Anch', 'Pos', 'Neg'],
                                                    histogram_path=histogram_path,
                                                    merge_method=met,
                                                    classifier_weights=classifier_weights_tensor[:,
                                                                       offset + neg_max_indices, :,
                                                                       :].squeeze(dim=0),
                                                    tb_path=f'triplet_{id}_anch_neg_forward', epoch=epoch,
                                                    writer=self.writer)
            elif 'attention' in self.merge_method:
                att_heatmap_path = os.path.join(heatmap_path_perepoch_id, f'triplet{id}_att.png')
                anch_name = utils.get_file_name(anch_path)
                pos_name = utils.get_file_name(pos_path)
                neg_name = utils.get_file_name(neg_path)
                if args.local_to_local or self.merge_global:

                    utils.apply_attention_heatmap([anchp_att, pos_att, neg_att],
                                                  [(anch_name, anch_org), (pos_name, pos_org)],
                                                  id,
                                                  att_heatmap_path,
                                                  overall_title=plot_title,
                                                  # individual_paths=[anch_hm_file_path,
                                                  #                   neg_hm_file_path],
                                                  # pair_paths=[anchneg_anch_hm_file_path, anchneg_neg_hm_file_path],
                                                  tb_path=f'triplet_{id}_anchpos_attention',
                                                  epoch=epoch,
                                                  writer=self.writer)

                    utils.apply_attention_heatmap([anchn_att, neg_att],
                                                  [(anch_name, anch_org), (neg_name, neg_org)],
                                                  id,
                                                  att_heatmap_path,
                                                  overall_title=plot_title,
                                                  # individual_paths=[anch_hm_file_path,
                                                  #                   neg_hm_file_path],
                                                  # pair_paths=[anchneg_anch_hm_file_path, anchneg_neg_hm_file_path],
                                                  tb_path=f'triplet_{id}_anchneg_attention',
                                                  epoch=epoch,
                                                  writer=self.writer)
                else:
                    utils.apply_attention_heatmap([anchp_att, pos_att, neg_att],  # anchn_att and anchp_att are the same
                                                  [(anch_name, anch_org), (pos_name, pos_org), (neg_name, neg_org)],
                                                  id,
                                                  att_heatmap_path,
                                                  overall_title=plot_title,
                                                  # individual_paths=[anch_hm_file_path,
                                                  #                   neg_hm_file_path],
                                                  # pair_paths=[anchneg_anch_hm_file_path, anchneg_neg_hm_file_path],
                                                  tb_path=f'triplet_{id}_attention',
                                                  epoch=epoch,
                                                  writer=self.writer)

            if 'diff' in self.merge_method or 'sim' in self.merge_method:
                if loss_fn is not None:
                    ext_batch_loss, parts = self.get_loss_value(args, loss_fn, anch_feat, pos_feat, neg_feat)
                    ext_loss = ext_batch_loss

                    ext_loss.backward(retain_graph=True)
                    ext_loss /= self.no_negative

                    plot_title = f"Backward Triplet Loss" + result_text

                    utils.apply_grad_heatmaps(net.get_activations_gradient(),
                                              net.get_activations().detach(),
                                              {'anch': anch_org,
                                               'pos': pos_org,
                                               'neg': neg_org}, 'triplet', id, heatmap_path_perepoch_id,
                                              plot_title, f'triplet_{id}_anchpos_triplet', epoch, self.writer)

                    class_loss /= (self.no_negative + 1)

                    loss = self.trpl_weight * ext_loss + self.bce_weight * class_loss

                else:

                    loss = self.bce_weight * class_loss

                plot_title = f"Backward Total Loss" + result_text

                loss.backward()
                utils.apply_grad_heatmaps(net.get_activations_gradient(),
                                          net.get_activations().detach(),
                                          {'anch': anch_org,
                                           'pos': pos_org,
                                           'neg': neg_org}, 'all', id, heatmap_path_perepoch_id,
                                          plot_title, f'triplet_{id}_anchpos_total', epoch, self.writer)

        self.created_image_heatmap_path = True

        self.logger.info(f'CAM: anch-pos acc: {self.cam_pos / self.cam_all}')
        self.logger.info(f'CAM: anch-neg acc: {self.cam_neg / self.cam_all}')

    def train_metriclearning(self, net, loss_fn, bce_loss, args, train_loader, val_loaders, val_loaders_fewshot,
                             train_loader_fewshot, cam_args=None, db_loaders=None, val_loaders_edgepred=None):
        net.train()
        # device = f'cuda:{net.device_ids[0]}'
        val_tol = args.early_stopping
        train_db_loader = db_loaders[0]
        val_db_loader = db_loaders[1]

        multiple_gpu = len(args.gpu_ids.split(",")) > 1

        if args.cuda:
            print('current_device: ', torch.cuda.current_device())

        if multiple_gpu:  # todo local not supported
            if net.module.aug_mask:
                learnable_params = [{'params': net.module.sm_net.parameters()},
                                    {'params': net.module.ft_net.rest.parameters(), 'lr': args.lr_resnet},
                                    {'params': net.module.ft_net.conv1.parameters(), 'lr': args.lr_new}]

            else:
                learnable_params = [{'params': net.module.sm_net.parameters()},
                                    {'params': net.module.ft_net.rest.parameters(), 'lr': args.lr_resnet},
                                    {'params': net.module.ft_net.pool.parameters(), 'lr': args.lr_new}]
        else:
            if net.aug_mask:
                learnable_params = [{'params': net.sm_net.parameters()},
                                    {'params': net.ft_net.rest.parameters(),
                                     'lr': args.lr_resnet,
                                     'weight_decay': args.weight_decay},
                                    {'params': net.ft_net.conv1.parameters(),
                                     'lr': args.lr_new,
                                     'weight_decay': args.weight_decay}]
            else:
                learnable_params = [{'params': net.sm_net.parameters()},
                                    {'params': net.ft_net.rest.parameters(),
                                     'lr': args.lr_resnet,
                                     'weight_decay': args.weight_decay},
                                    {'params': net.ft_net.pool.parameters(),
                                     'lr': args.lr_new,
                                     'weight_decay': args.weight_decay}]

            if net.local_features:
                learnable_params += [{'params': net.local_features.parameters(),
                                      'lr': args.lr_new,
                                      'weight_decay': args.weight_decay}]

            if net.channel_attention:
                learnable_params += [{'params': net.channel_attention.parameters(),
                                      'lr': args.lr_new,
                                      'weight_decay': args.weight_decay}]


            if net.diffsim_fc_net:
                learnable_params += [{'params': net.diffsim_fc_net.parameters(),
                                      'lr': args.lr_new,
                                      'weight_decay': args.weight_decay}]

            if net.classifier:
                learnable_params += [{'params': net.classifier.parameters(),
                                      'lr': args.lr_new,
                                      'weight_decay': args.weight_decay}]

            if net.ft_net.last_conv is not None:
                learnable_params += [{'params': net.ft_net.last_conv.parameters(),
                                      'lr': args.lr_new,
                                      'weight_decay': args.weight_decay}]

        # net.ft_net.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        opt = torch.optim.Adam(learnable_params, lr=args.lr_new, weight_decay=args.weight_decay)
        opt.zero_grad()

        time_start = time.time()
        queue = deque(maxlen=20)

        max_epochs = args.epochs

        metric_ACC = metrics.Metric_Accuracy()

        max_val_acc = 0

        max_val_acc_knwn = 0
        max_val_acc_unknwn = 0
        val_acc = -1
        val_loss = -1
        val_rgt = 0
        val_err = 0
        best_model = ''

        max_val_between_epochs = -1

        drew_graph = multiple_gpu

        val_counter = 0

        self.important_hparams = self._tb_get_important_hparams(args)

        if args.gamma_step != 0:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=args.gamma_step, gamma=args.gamma)
            adaptive_scheduler = None
        else:
            scheduler = None
            adaptive_scheduler = Adaptive_Scheduler(opt, gamma=args.gamma,
                                                    logger=self.logger,
                                                    tol=args.lr_tol,
                                                    val=not args.lr_adaptive_loss,
                                                    loss=args.lr_adaptive_loss)

        for epoch in range(1, max_epochs + 1):

            epoch_start = time.time()

            if args.negative_path != '':
                self.save_best_negatives(args, net.ft_net, train_db_loader)
                train_loader.dataset.load_best_negatives(args.negative_path)

            models.top_model.A_SUM = [0, 0]

            with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{args.epochs}') as t:
                if self.draw_grad:
                    grad_save_path = os.path.join(self.plt_save_path, f'grads/epoch_{epoch}/')
                    # self.logger.info(grad_save_path)
                    utils.make_dirs(grad_save_path)
                else:
                    grad_save_path = None
                all_batches_start = time.time()

                utils.print_gpu_stuff(args.cuda, 'before train epoch')

                if args.loss == 'batchhard':
                    t, (train_loss, train_bce_loss, train_triplet_loss), (
                        _, _) = self.train_metriclearning_one_epoch_batchhard(args, t, net, opt, bce_loss,
                                                                              metric_ACC,
                                                                              loss_fn, train_loader, epoch,
                                                                              grad_save_path, drew_graph)

                else:
                    t, (train_loss, train_bce_loss, train_triplet_loss), (
                        pos_parts, neg_parts) = self.train_metriclearning_one_epoch(args, t, net, opt, bce_loss,
                                                                                    metric_ACC,
                                                                                    loss_fn, train_loader, epoch,
                                                                                    grad_save_path, drew_graph)
                utils.print_gpu_stuff(args.cuda, 'after train epoch')

                all_batches_end = time.time()
                if utils.MY_DEC.enabled:
                    self.logger.info(f'########### all batches time: {all_batches_end - all_batches_start}')

                with torch.no_grad():  # for evaluations
                    if args.loss == 'maxmargin':
                        plt.hist([np.array(pos_parts).flatten(), np.array(neg_parts).flatten()], bins=30, alpha=0.3,
                                 label=['pos', 'neg'])
                        plt.title(f'Losses Epoch {epoch}')
                        plt.legend(loc='upper right')
                        plt.savefig(f'{self.plt_save_path}/pos_part_{epoch}.png')
                        plt.close('all')

                    if bce_loss is None:
                        bce_loss = loss_fn

                    utils.print_gpu_stuff(args.cuda, 'before train few_shot')

                    if args.train_fewshot:
                        start = time.time()
                        train_fewshot_acc, train_fewshot_loss, train_fewshot_right, train_fewshot_error, train_fewshot_predictions = self.apply_fewshot_eval(
                            args, net, train_loader_fewshot, bce_loss)
                        end = time.time()

                        utils.print_gpu_stuff(args.cuda, 'after train few_shot')

                        if utils.MY_DEC.enabled:
                            self.logger.info(f'########### apply_fewshot_eval TRAIN time: {end - start}')

                        self.logger.info(
                            f'Train_Fewshot_Acc: {train_fewshot_acc}, Train_Fewshot_loss: {train_fewshot_loss},\n '
                            f'Train_Fewshot_Right: {train_fewshot_right}, Train_Fewshot_Error: {train_fewshot_error}')

                    self.writer.add_scalar('Train/Loss', train_loss / len(train_loader), epoch)
                    if loss_fn is not None:
                        self.writer.add_scalar('Train/Triplet_Loss', train_triplet_loss / len(train_loader), epoch)

                    self.writer.add_scalar('Train/BCE_Loss', train_bce_loss / len(train_loader), epoch)
                    self.writer.add_scalar('Train/Acc', metric_ACC.get_acc(), epoch)
                    # self.writer.add_hparams(self.important_hparams, {'Train_2/Acc': metric_ACC.get_acc()}, epoch)

                    if args.train_fewshot:
                        self.writer.add_scalar('Train/Fewshot_Loss', train_fewshot_loss / len(train_loader_fewshot),
                                               epoch)
                        self.writer.add_scalar('Train/Fewshot_Acc', train_fewshot_acc, epoch)

                    self.writer.flush()

                    if val_loaders is not None and (epoch % args.test_freq == 0 or epoch == max_epochs):
                        net.eval()
                        # device = f'cuda:{net.device_ids[0]}'
                        val_acc_unknwn, val_acc_knwn = -1, -1
                        results = {}
                        results_to_save = {}
                        if args.eval_mode == 'fewshot':

                            for fewshot_loader, loader, comm in zip(val_loaders_fewshot, val_loaders,
                                                                    EVAL_SET_NAMES[len(val_loaders)]):

                                utils.print_gpu_stuff(args.cuda, f'before test few_shot {comm}')
                                _, _, val_acc_fewshot, _ = self.test_fewshot(args, net,
                                                                             fewshot_loader,
                                                                             bce_loss,
                                                                             val=True,
                                                                             epoch=epoch,
                                                                             comment=comm)

                                utils.print_gpu_stuff(args.cuda, f'after test few_shot {comm} and before test_metric')

                                val_auc, val_acc, val_rgt_err, val_preds_pos_neg, val_loss = self.test_metric(
                                    args, net, loader,
                                    loss_fn, bce_loss, val=True,
                                    epoch=epoch, comment=comm)

                                if comm not in results.keys():
                                    results[comm] = {}
                                    results_to_save[comm] = {}
                                #
                                results[comm]['right'] = val_rgt_err['right']
                                results[comm]['wrong'] = val_rgt_err['wrong']
                                results[comm]['val_auc'] = val_auc
                                results[comm]['val_acc'] = val_acc
                                results[comm]['val_acc_fewshot'] = val_acc_fewshot
                                results[comm]['val_loss'] = val_loss
                                # val_err_knwn = val_rgt_err_knwn['wrong']
                                # val_rgt_knwn = val_rgt_err_knwn['right']

                                results_to_save[comm] = val_preds_pos_neg
                                # val_preds_knwn_pos = val_preds_knwn_pos_neg['pos']
                                # val_preds_knwn_neg = val_preds_knwn_pos_neg['neg']

                                utils.print_gpu_stuff(args.cuda, f'after test_metric {comm}')

                            # _, _, val_acc_unknwn_fewshot, _ = self.test_fewshot(args,
                            #                                                     net,
                            #                                                     val_loaders_fewshot[
                            #                                                         1],
                            #                                                     bce_loss,
                            #                                                     val=True,
                            #                                                     epoch=epoch,
                            #                                                     comment='unknown')
                            # utils.print_gpu_stuff(args.cuda, 'after test_fewshot 2 and before test_metric 2')
                            #
                            # unseen_val_auc, val_acc_unknwn, val_rgt_err_unknwn, val_preds_unknwn_pos_neg = self.test_metric(
                            #     args, net, val_loaders[1],
                            #     loss_fn, bce_loss, val=True,
                            #     epoch=epoch, comment='unknown')
                            #
                            # val_err_unknwn = val_rgt_err_unknwn['wrong']
                            # val_rgt_unknwn = val_rgt_err_unknwn['right']
                            #
                            # val_preds_unknwn_pos = val_preds_unknwn_pos_neg['pos']
                            # val_preds_unknwn_neg = val_preds_unknwn_pos_neg['neg']

                            utils.print_gpu_stuff(args.cuda, 'after all validation')

                        elif args.eval_mode == 'edgepred':
                            utils.print_gpu_stuff(args.cuda, 'before test few_shot 1')
                            val_rgt_knwn, val_err_knwn, val_acc_knwn, val_preds_knwn = self.test_edgepred(args, net,
                                                                                                          val_loaders_edgepred[
                                                                                                              0],
                                                                                                          bce_loss,
                                                                                                          val=True,
                                                                                                          epoch=epoch,
                                                                                                          comment='known')

                            utils.print_gpu_stuff(args.cuda, 'after test_edgepred 1 and before test_metric')

                            self.test_metric(args, net, val_loaders[0],
                                             loss_fn, bce_loss, val=True,
                                             epoch=epoch, comment='known')

                            utils.print_gpu_stuff(args.cuda, 'after test_metric 1 and before test_fewshot 2')

                            val_rgt_unknwn, val_err_unknwn, val_acc_unknwn, val_preds_unknwn = self.test_edgepred(args,
                                                                                                                  net,
                                                                                                                  val_loaders_edgepred[
                                                                                                                      1],
                                                                                                                  bce_loss,
                                                                                                                  val=True,
                                                                                                                  epoch=epoch,
                                                                                                                  comment='unknown')
                            utils.print_gpu_stuff(args.cuda, 'after test_edgepred 2 and before test_metric 2')

                            self.test_metric(args, net, val_loaders[1],
                                             loss_fn, bce_loss, val=True,
                                             epoch=epoch, comment='unknown')

                            utils.print_gpu_stuff(args.cuda, 'after all validation')

                        elif args.eval_mode == 'simple':  # todo not compatible with new data-splits
                            val_rgt, val_err, val_acc = self.test_simple(args, net, val_loaders, loss_fn, val=True,
                                                                         epoch=epoch)
                        else:
                            raise Exception('Unsupporeted eval mode')

                        val_acc_str = ''

                        for key, value in results.items():
                            val_acc_str += f'{key} val acc: {value["val_acc"]}, '

                        # self.logger.info('known val acc: [%f], unknown val acc [%f]' % (val_acc_knwn, val_acc_unknwn))
                        self.logger.info(val_acc_str)
                        self.logger.info('*' * 30)
                        if len(val_loaders) == 2:  # validation has seen and unseen sets
                            if results['seen']['val_acc'] > max_val_acc_knwn:
                                self.logger.info(
                                    'known val acc: [%f], beats previous max [%f]' % (
                                        results['seen']['val_acc'], max_val_acc_knwn))
                                self.logger.info('known rights: [%d], known errs [%d]' % (results['seen']['right'],
                                                                                          results['seen']['wrong']))
                                max_val_acc_knwn = results['seen']['val_acc']

                            if results['unseen']['val_acc'] > max_val_acc_unknwn:
                                self.logger.info(
                                    'unknown val acc: [%f], beats previous max [%f]' % (
                                        results['unseen']['val_acc'], max_val_acc_unknwn))
                                self.logger.info(
                                    'unknown rights: [%d], unknown errs [%d]' % (results['unseen']['right'],
                                                                                 results['unseen']['wrong']))
                                max_val_acc_unknwn = results['unseen']['val_acc']

                            val_rgt = (results['seen']['right'] + results['unseen']['right'])
                            val_err = (results['seen']['wrong'] + results['unseen']['wrong'])

                            val_acc = (val_rgt * 1.0) / (val_rgt + val_err)

                            val_loss = (results['seen']['val_loss'] + results['unseen']['val_loss']) / 2


                        else:  # no seen and unseen dataset for validation
                            val_loss = results['total']['val_loss']
                            val_acc = results['total']['val_acc']
                            val_rgt = results['total']['right']
                            val_err = results['total']['wrong']

                        # self.writer.add_scalar('Total_Val/Acc', val_acc, epoch)
                        # self.writer.add_hparams(self.important_hparams, {'Total_Val/Acc': val_acc}, epoch)
                        if args.hparams:
                            self.hparams_metric['Total_Val/Acc'] = val_acc
                        else:
                            self.writer.add_scalar('Total_Val/Acc', val_acc, epoch)

                        self.writer.add_scalar('Total_Val/Loss', val_loss, epoch)
                        self.writer.flush()

                        if val_acc >= max_val_acc or epoch == max_epochs:
                            utils.print_gpu_stuff(args.cuda, 'Before saving model')
                            val_counter = 0
                            if args.train_fewshot:
                                np.savez(os.path.join(self.save_path, f'train_preds_epoch{epoch}'),
                                         np.array(train_fewshot_predictions))

                            for key, value in results_to_save.items():
                                np.savez(os.path.join(self.save_path, f'val_preds_{key}_neg_epoch{epoch}'),
                                         np.array(value['neg']))
                                np.savez(os.path.join(self.save_path, f'val_preds_{key}_pos_epoch{epoch}'),
                                         np.array(value['pos']))

                            # np.savez(os.path.join(self.save_path, f'val_preds_unknwn_neg_epoch{epoch}'),
                            #          np.array(val_preds_unknwn_neg))
                            # np.savez(os.path.join(self.save_path, f'val_preds_unknwn_pos_epoch{epoch}'),
                            #          np.array(val_preds_unknwn_pos))

                            self.logger.info(
                                f'[epoch {epoch}] saving model... current val acc: [{val_acc}], previous val acc [{max_val_acc}]')
                            best_model = self.save_model(args, net, epoch, val_acc)
                            max_val_acc = val_acc
                            utils.print_gpu_stuff(args.cuda, 'Before saving model')

                            queue.append(val_rgt * 1.0 / (val_rgt + val_err))

                            if args.hparams:
                                self.writer.add_hparams(self.important_hparams, self.hparams_metric, epoch)
                                self.writer.flush()

                    elif (epoch) % args.test_freq == 0 or epoch == max_epochs:
                        self.logger.info(
                            f'[epoch {epoch}] saving model...')
                        best_model = self.save_model(args, net, epoch, 0.0)

            if models.top_model.A_SUM[1] != 0:
                self.logger.info(
                    f'Epoch {epoch} A_SUM = {models.top_model.A_SUM} and:\n{models.top_model.A_SUM[0] / models.top_model.A_SUM[1]}')
                models.top_model.A_SUM = [0, 0]

            epoch_end = time.time()
            if utils.MY_DEC.enabled:
                self.logger.info(f'########### one epoch (after batch loop) time: {epoch_end - epoch_start}')

            if args.train_diff_plot:
                self.logger.info('plotting train class diff plot...')
                if args.my_dist:
                    self.make_all_emb_dist_db(args, net, train_db_loader,
                                              eval_sampled=args.sampled_results,
                                              eval_per_class=args.per_class_results,
                                              batch_size=args.db_batch,
                                              mode='train',
                                              epoch=epoch,
                                              k_at_n=False)
                else:
                    self.make_emb_db(args, net, train_db_loader,
                                     eval_sampled=args.sampled_results,
                                     eval_per_class=args.per_class_results,
                                     newly_trained=True,
                                     batch_size=args.db_batch,
                                     mode='train',
                                     epoch=epoch,
                                     k_at_n=False)

            if val_db_loader:
                self.logger.info('plotting val class diff plot...')
                if args.my_dist:
                    self.make_all_emb_dist_db(args, net, val_db_loader,
                                              eval_sampled=args.sampled_results,
                                              eval_per_class=args.per_class_results,
                                              batch_size=args.db_batch,
                                              mode='val',
                                              epoch=epoch,
                                              k_at_n=args.katn)
                else:
                    self.make_emb_db(args, net, val_db_loader,
                                     eval_sampled=args.sampled_results,
                                     eval_per_class=args.per_class_results,
                                     newly_trained=True,
                                     batch_size=args.db_batch,
                                     mode='val',
                                     epoch=epoch,
                                     k_at_n=args.katn)

            if max_val_between_epochs <= max_val_acc or epoch == max_epochs:
                max_val_between_epochs = max_val_acc
                if args.cam:
                    print(f'Drawing heatmaps on epoch {epoch}...')
                    self.logger.info(f'Drawing heatmaps on epoch {epoch}...')
                    self.draw_heatmaps(net=net,
                                       loss_fn=loss_fn,
                                       bce_loss=bce_loss,
                                       args=args,
                                       cam_loader=cam_args[0],
                                       transform_for_model=cam_args[1],
                                       transform_for_heatmap=cam_args[2],
                                       epoch=epoch,
                                       count=1,
                                       draw_all_thresh=self.draw_all_thresh)

                    self.logger.info(f'DONE drawing heatmaps on epoch {epoch}!!!')

            self.update_bce_tco(epoch)

            self._tb_draw_histograms(args, net, epoch)

            epoch_end = time.time()
            if utils.MY_DEC.enabled:
                self.logger.info(f'########### one epoch (complete) time: {epoch_end - epoch_start}')

            if scheduler:
                scheduler.step()
            else:
                adaptive_scheduler.step(current_loss=val_loss, current_val=val_acc)

        # acc = 0.0
        # for d in queue:
        #     acc += d
        # self.logger.info("#" * 70)
        # self.logger.info(f'queue len: {len(queue)}')

        if args.project_tb:
            self.logger.info("Start projecting")
            # self._tb_project_embeddings(args, net.ft_net, train_loader, 1000)
            self.logger.info("Projecting done")

        return net, best_model

    def test_simple(self, args, net, data_loader, loss_fn, val=False, epoch=0):
        net.eval()
        # device = f'cuda:{net.device_ids[0]}'
        if val:
            prompt_text = f'VAL SIMPLE epoch {epoch}: \tcorrect:\t%d\terror:\t%d\tval_loss:%f\tval_acc:%f\tval_rec:%f\tval_negacc:%f\t'
            prompt_text_tb = 'Val'
        else:
            prompt_text = 'TEST SIMPLE:\tTest set\tcorrect:\t%d\terror:\t%d\ttest_loss:%f\ttest_acc:%f\ttest_rec:%f\ttest_negacc:%f\t'
            prompt_text_tb = 'Test'

        tests_right, tests_error = 0, 0

        fn = 0
        fp = 0
        tn = 0
        tp = 0

        for label, (test1, test2) in enumerate(data_loader, 1):
            if args.cuda:
                test1, test2 = test1.cuda(), test2.cuda()
            test1, test2 = Variable(test1), Variable(test2)

            output = net.forward(test1, test2)
            test_loss = loss_fn(output, label)
            output = output.data.cpu().numpy()
            pred = np.rint(output)

            tn_t, fp_t, fn_t, tp_t = confusion_matrix(label, pred).ravel()

            fn += fn_t
            tn += tn_t
            fp += fp_t
            tp += tp_t

        test_acc = ((tp + tn) * 1.0) / (tp + tn + fn + fp)
        test_recall = (tp * 0.1) / (tp + fn)
        test_negacc = (tn * 0.1) / (tn + fp)
        self.logger.info('$' * 70)
        self.logger.info(prompt_text % (tests_right, tests_error, test_loss, test_acc, test_recall, test_negacc))
        self.logger.info('$' * 70)

        self.writer.add_scalar(f'{prompt_text_tb}/Loss', test_loss, epoch)
        self.writer.add_scalar(f'{prompt_text_tb}/Acc', test_acc, epoch)
        self.writer.flush()

        return tests_right, tests_error, test_acc

    def test_metric(self, args, net, data_loader, loss_fn, bce_loss, val=False, epoch=0, comment=''):
        net.eval()
        # device = f'cuda:{net.device_ids[0]}'
        if val:
            prompt_text = comment + f' VAL METRIC LEARNING epoch {epoch}:\tcorrect:\t%d\terror:\t%d\tval_acc:%f\tval_loss:%f\t'
            prompt_text_tb = comment + '_Val'
        else:
            prompt_text = comment + ' TEST METRIC LEARNING:\tcorrect:\t%d\terror:\t%d\ttest_acc:%f\ttest_loss:%f\t'
            prompt_text_tb = comment + '_Test'

        tests_right, tests_error = 0, 0

        metric_ACC = metrics.Metric_Accuracy()

        metric_ACC.reset_acc()

        test_loss = 0
        test_bce_loss = 0
        test_triplet_loss = 0
        loss = 0
        true_label_auc = []
        pred_label_auc = []
        all_pos_predictions = []
        all_neg_predictions = []
        with tqdm(total=len(data_loader), desc=f'{prompt_text_tb}') as t:
            for _, (anch, pos, neg) in enumerate(data_loader, 1):

                one_labels = torch.tensor([1 for _ in range(anch.shape[0])], dtype=float).reshape(-1, 1)
                zero_labels = torch.tensor([0 for _ in range(anch.shape[0])], dtype=float).reshape(-1, 1)

                if args.cuda:
                    anch, pos, neg, one_labels, zero_labels = anch.cuda(), pos.cuda(), neg.cuda(), one_labels.cuda(), zero_labels.cuda()
                anch, pos, neg, one_labels, zero_labels = Variable(anch), Variable(pos), Variable(neg), Variable(
                    one_labels), Variable(zero_labels)

                ###
                pos_pred, pos_dist, anch_feat, pos_feat = net.forward(anch, pos, feats=True)
                class_loss = bce_loss(pos_pred.squeeze(axis=1), one_labels.squeeze(axis=1))

                pred_label_auc.extend(pos_pred.data.cpu().numpy())
                true_label_auc.extend(one_labels.data.cpu().numpy())
                all_pos_predictions.extend(pos_pred.data.cpu().numpy())
                metric_ACC.update_acc(pos_pred.squeeze(axis=1), one_labels.squeeze(axis=1))

                for neg_iter in range(self.no_negative):
                    # self.logger.info(anch.shape)
                    # self.logger.info(neg[:, neg_iter, :, :, :].squeeze(dim=1).shape)
                    neg_pred, neg_dist, _, neg_feat = net.forward(anch, neg[:, neg_iter, :, :, :].squeeze(dim=1),
                                                                  feats=True)

                    all_neg_predictions.extend(neg_pred.data.cpu().numpy())
                    pred_label_auc.extend(neg_pred.data.cpu().numpy())
                    true_label_auc.extend(zero_labels.data.cpu().numpy())

                    class_loss += bce_loss(neg_pred.squeeze(axis=1), zero_labels.squeeze(axis=1))
                    metric_ACC.update_acc(neg_pred.squeeze(axis=1), zero_labels.squeeze(axis=1))

                    if loss_fn is not None:
                        ext_batch_loss, parts = self.get_loss_value(args, loss_fn, anch_feat, pos_feat, neg_feat)

                        if neg_iter == 0:
                            ext_loss = ext_batch_loss
                        else:
                            ext_loss += ext_batch_loss

                class_loss /= (self.no_negative + 1)
                if loss_fn is not None:
                    ext_loss /= self.no_negative
                    test_triplet_loss += ext_loss.item()
                    loss = self.trpl_weight * ext_loss + self.bce_weight * class_loss
                else:
                    loss = self.bce_weight * class_loss

                test_loss += loss.item()

                test_bce_loss += class_loss.item()

                t.update()

        self.logger.info(f'Lnegth of true_label_auc for calculating is: {len(true_label_auc)}')
        roc_auc = roc_auc_score(true_label_auc, utils.sigmoid(np.array(pred_label_auc)))

        self.logger.info('$' * 70)

        # self.writer.add_scalar(f'{prompt_text_tb}/Triplet_Loss', test_loss / len(data_loader), epoch)
        self.logger.error(f'{prompt_text_tb}/Loss:  {test_loss / len(data_loader)}, epoch: {epoch}')
        self.writer.add_scalar(f'{prompt_text_tb}/Loss', test_loss / len(data_loader), epoch)
        if loss_fn is not None:
            self.logger.error(f'{prompt_text_tb}/Triplet_Loss: {test_triplet_loss / len(data_loader)}, epoch: {epoch}')
            self.writer.add_scalar(f'{prompt_text_tb}/Triplet_Loss', test_triplet_loss / len(data_loader), epoch)
        self.logger.error(f'{prompt_text_tb}/BCE_Loss: {test_bce_loss / len(data_loader)}, epoch: {epoch}')
        self.writer.add_scalar(f'{prompt_text_tb}/BCE_Loss', test_bce_loss / len(data_loader), epoch)

        self.logger.error(f'{prompt_text_tb}/ROC_AUC: {roc_auc}, epoch: {epoch}')
        self.writer.add_scalar(f'{prompt_text_tb}/ROC_AUC', roc_auc, epoch)

        self.logger.error(f'{prompt_text_tb}/Acc: {metric_ACC.get_acc()} epoch: {epoch}')
        if args.hparams:
            self.hparams_metric[f'{prompt_text_tb}/Acc'] = metric_ACC.get_acc()
        else:
            self.writer.add_scalar(f'{prompt_text_tb}/Acc', metric_ACC.get_acc(), epoch)

        # self.writer.add_scalar(f'{prompt_text_tb}/Acc', test_acc, epoch)
        self.writer.flush()

        return roc_auc, metric_ACC.get_acc(), metric_ACC.get_right_wrong(), {'pos': all_pos_predictions,
                                                                             'neg': all_neg_predictions}, (
                       test_loss / len(data_loader))

    def test_edgepred(self, args, net, data_loader, loss_fn, val=False, epoch=0, comment=''):
        net.eval()
        # device = f'cuda:{net.device_ids[0]}'
        if val:
            prompt_text = comment + f' VAL EDGE PRED epoch {epoch}:\tcorrect:\t%d\terror:\t%d\tval_auc:%f\tval_loss:%f\t'
            prompt_text_tb = comment + '_Val'
        else:
            prompt_text = comment + ' TEST EDGE PRED:\tcorrect:\t%d\terror:\t%d\ttest_auc:%f\ttest_loss:%f\t'
            prompt_text_tb = comment + '_Test'

        test_auc, test_loss, tests_right, tests_error, tests_predictions = self.apply_edgepred_eval(args, net,
                                                                                                    data_loader)

        self.logger.info('$' * 70)
        self.logger.info(prompt_text % (tests_right, tests_error, test_auc, test_loss))
        self.logger.info('$' * 70)

        self.writer.add_scalar(f'{prompt_text_tb}/Edgepred_Loss', test_loss, epoch)
        self.writer.add_scalar(f'{prompt_text_tb}/Edgepred_AUC', test_auc, epoch)
        self.writer.flush()

        return tests_right, tests_error, test_auc, tests_predictions

    def test_fewshot(self, args, net, data_loader, loss_fn, val=False, epoch=0, comment=''):
        net.eval()
        # device = f'cuda:{net.device_ids[0]}'
        if val:
            prompt_text = comment + f' VAL FEW SHOT epoch {epoch}:\tcorrect:\t%d\terror:\t%d\tval_acc:%f\tval_loss:%f\t'
            prompt_text_tb = comment + '_Val'
        else:
            prompt_text = comment + ' TEST FEW SHOT:\tcorrect:\t%d\terror:\t%d\ttest_acc:%f\ttest_loss:%f\t'
            prompt_text_tb = comment + '_Test'

        test_acc, test_loss, tests_right, tests_error, tests_predictions = self.apply_fewshot_eval(args, net,
                                                                                                   data_loader, loss_fn)

        self.logger.info('$' * 70)
        self.logger.info(prompt_text % (tests_right, tests_error, test_acc, test_loss))
        self.logger.info('$' * 70)

        self.writer.add_scalar(f'{prompt_text_tb}/Fewshot_Loss', test_loss, epoch)
        self.writer.add_scalar(f'{prompt_text_tb}/Fewshot_Acc', test_acc, epoch)
        self.writer.flush()

        return tests_right, tests_error, test_acc, tests_predictions

    def make_all_emb_dist_db(self, args, net, data_loader, eval_sampled, eval_per_class, batch_size=None,
                             mode='val', epoch=-1, k_at_n=True):
        """

        :param batch_size:
        :param eval_sampled:
        :param eval_per_class:
        :param newly_trained:
        :param mode:
        :param args: utils args
        :param net: trained top_model network
        :param data_loader: DataLoader object
        :param epoch: epoch we're in
        :param k_at_n: Do k at n
        :return: None
        """

        return_bg = (mode.startswith('val') and
                     args.vu_folder_name != 'none') \
                    or \
                    (mode.startswith('test') and
                     args.tu_folder_name != 'none')

        if (not os.path.exists(os.path.join(self.save_path, f'{args.dataset_name}_{mode}Sim.h5'))):
            net.eval()
            # device = f'cuda:{net.device_ids[0]}'
            if batch_size is None:
                batch_size = args.batch_size

            test_sim = -1 * np.ones(shape=len(data_loader.dataset))

            lbls, seen = data_loader.dataset.get_info()
            test_classes = np.array(lbls)
            test_seen = np.array(seen, dtype=int)

            # test_paths = np.array([i for i in range(len(lbls))])

            chunks = len(args.feature_map_layers)

            with tqdm(total=len(data_loader), desc=f'Getting embeddings for {mode}') as t:

                for idx, tpl in enumerate(data_loader):

                    end_dist = min((idx + 1) * batch_size, len(test_sim))

                    if return_bg and mode != 'train':
                        (img1, img2, lbl1, lbl2, seen1, seen2, path1, path2) = tpl
                    else:
                        (img1, lbl1, seen1, path1) = tpl

                    if args.cuda:
                        img1 = img1.cuda()
                        img2 = img2.cuda()

                    img1 = Variable(img1)
                    img2 = Variable(img2)

                    _, _, img1_feat, img2_feat = net.forward(img1, img2, feats=True)
                    output = utils.calc_custom_cosine_sim(img1_feat.chunk(chunks=chunks, dim=1),
                                                          img2_feat.chunk(chunks=chunks, dim=1))
                    output = output.data.cpu().numpy().flatten()

                    test_sim[idx * batch_size:end_dist] = output

                    t.update()

            test_sim = test_sim.reshape((len(test_classes), len(test_classes)))
            # utils.save_h5(f'{args.dataset_name}_{mode}_ids', test_paths, 'S20',
            #               os.path.join(self.save_path, f'{args.dataset_name}_{mode}Ids.h5'))
            utils.save_h5(f'{args.dataset_name}_{mode}_classes', test_classes, 'i8',
                          os.path.join(self.save_path, f'{args.dataset_name}_{mode}Classes.h5'))
            utils.save_h5(f'{args.dataset_name}_{mode}_sim', test_sim, 'f',
                          os.path.join(self.save_path, f'{args.dataset_name}_{mode}Sim.h5'))

            if return_bg and mode != 'train':
                utils.save_h5(f'{args.dataset_name}_{mode}_seen', test_seen, 'i2',
                              os.path.join(self.save_path, f'{args.dataset_name}_{mode}Seen.h5'))

        test_seen = np.zeros(((len(data_loader.dataset))))
        test_sim = utils.load_h5(f'{args.dataset_name}_{mode}_sim',
                                 os.path.join(self.save_path, f'{args.dataset_name}_{mode}Sim.h5'))
        test_classes = utils.load_h5(f'{args.dataset_name}_{mode}_classes',
                                     os.path.join(self.save_path, f'{args.dataset_name}_{mode}Classes.h5'))
        if return_bg and mode != 'train':
            test_seen = utils.load_h5(f'{args.dataset_name}_{mode}_seen',
                                      os.path.join(self.save_path, f'{args.dataset_name}_{mode}Seen.h5'))

        # pca_path = os.path.join(self.scatter_plot_path, f'pca_{epoch}.png')

        # self.draw_dim_reduced(test_feats, test_classes, method='pca', title="on epoch " + str(epoch), path=pca_path)

        ##  for drawing tsne plot
        # tsne_path = os.path.join(self.gen_plot_path, f'{mode}/tsne_{epoch}.png')
        # self.draw_dim_reduced(test_feats, test_classes, method='tsne', title=f"{mode}, epoch: " + str(epoch),
        #                       path=tsne_path)

        if mode != 'test':

            if mode == 'val':
                tb_tag = 'Val'
            elif mode == 'train':
                tb_tag = 'Train'
            else:
                tb_tag = 'Other'

            # self.plot_silhouette_score(test_feats, test_classes, epoch, mode, silhouette_path,
            #                            f'Total_{tb_tag}')

        # import pdb
        # pdb.set_trace()
        if k_at_n:
            utils.calculate_k_at_n(args, None, test_classes, test_seen, logger=self.logger,
                                   limit=args.limit_samples,
                                   run_number=args.number_of_runs,
                                   save_path=self.save_path,
                                   sampled=eval_sampled,
                                   even_sampled=False,
                                   per_class=eval_per_class,
                                   mode=mode,
                                   sim_matrix=test_sim)

            self.logger.info('results at: ' + self.save_path)

    def make_emb_db(self, args, net, data_loader, eval_sampled, eval_per_class, newly_trained=True, batch_size=None,
                    mode='val', epoch=-1, k_at_n=True):
        """

        :param batch_size:
        :param eval_sampled:
        :param eval_per_class:
        :param newly_trained:
        :param mode:
        :param args: utils args
        :param net: trained top_model network
        :param data_loader: DataLoader object
        :param epoch: epoch we're in
        :param k_at_n: Do k at n
        :return: None
        """

        has_attention = 'attention' in args.merge_method

        return_bg = (mode.startswith('val') and
                     args.vu_folder_name != 'none') \
                    or \
                    (mode.startswith('test') and
                     args.tu_folder_name != 'none')

        if newly_trained or \
                (not os.path.exists(os.path.join(self.save_path, f'{args.dataset_name}_{mode}Feats.h5'))):
            net.eval()
            # device = f'cuda:{net.device_ids[0]}'
            if batch_size is None:
                batch_size = args.batch_size

            steps = int(np.ceil(len(data_loader) / batch_size))

            test_classes = np.zeros(((len(data_loader.dataset))))
            test_seen = np.zeros(((len(data_loader.dataset))))
            test_paths = np.empty(dtype='S50', shape=((len(data_loader.dataset))))

            if self.merge_method == 'local-attention' or self.merge_method == 'local-ds-attention':
                coeff = len(args.feature_map_layers)
            else:
                coeff = 1

            if args.feat_extractor == 'resnet50':
                test_feats = np.zeros((len(data_loader.dataset), 2048 * coeff))
            elif args.feat_extractor == 'resnet18':
                test_feats = np.zeros((len(data_loader.dataset), 512 * coeff))
            elif args.feat_extractor == 'vgg16':
                test_feats = np.zeros((len(data_loader.dataset), 4096 * coeff))
            else:
                raise Exception('Not handled feature extractor')

            if args.dim_reduction != 0:
                test_feats = np.zeros((len(data_loader.dataset), args.dim_reduction * coeff))

            with tqdm(total=len(data_loader), desc=f'Getting embeddings for {mode}') as t:
                for idx, tpl in enumerate(data_loader):

                    end = min((idx + 1) * batch_size, len(test_feats))

                    if return_bg and mode != 'train':
                        (img, lbl, seen, path) = tpl
                    else:
                        (img, lbl, path) = tpl

                    if args.cuda:
                        img = img.cuda()

                    img = Variable(img)

                    output = net.forward(img, None, single=True)
                    output = output.data.cpu().numpy()

                    test_feats[idx * batch_size:end, :] = output
                    test_classes[idx * batch_size:end] = lbl
                    test_paths[idx * batch_size:end] = path

                    if return_bg and mode != 'train':  # todo 1. seen is zeros -> res under unseen? 2. Seen is weird
                        test_seen[idx * batch_size:end] = seen.to(int)

                    t.update()

            chunks = len(args.feature_map_layers)

            # if has_attention:
            #     if test_feats.dtype != np.float32:
            #         test_feats = test_feats.astype(np.float32)
            #     test_feats = utils.get_attention_normalized(test_feats, chunks=chunks)

            utils.save_h5(f'{args.dataset_name}_{mode}_ids', test_paths, 'S20',
                          os.path.join(self.save_path, f'{args.dataset_name}_{mode}Ids.h5'))
            utils.save_h5(f'{args.dataset_name}_{mode}_classes', test_classes, 'i8',
                          os.path.join(self.save_path, f'{args.dataset_name}_{mode}Classes.h5'))
            utils.save_h5(f'{args.dataset_name}_{mode}_feats', test_feats, 'f',
                          os.path.join(self.save_path, f'{args.dataset_name}_{mode}Feats.h5'))
            if return_bg and mode != 'train':
                utils.save_h5(f'{args.dataset_name}_{mode}_seen', test_seen, 'i2',
                              os.path.join(self.save_path, f'{args.dataset_name}_{mode}Seen.h5'))

        test_seen = np.zeros(((len(data_loader.dataset))))
        test_feats = utils.load_h5(f'{args.dataset_name}_{mode}_feats',
                                   os.path.join(self.save_path, f'{args.dataset_name}_{mode}Feats.h5'))
        test_classes = utils.load_h5(f'{args.dataset_name}_{mode}_classes',
                                     os.path.join(self.save_path, f'{args.dataset_name}_{mode}Classes.h5'))
        if return_bg and mode != 'train':
            test_seen = utils.load_h5(f'{args.dataset_name}_{mode}_seen',
                                      os.path.join(self.save_path, f'{args.dataset_name}_{mode}Seen.h5'))

        # pca_path = os.path.join(self.scatter_plot_path, f'pca_{epoch}.png')

        # self.draw_dim_reduced(test_feats, test_classes, method='pca', title="on epoch " + str(epoch), path=pca_path)

        ##  for drawing tsne plot
        # tsne_path = os.path.join(self.gen_plot_path, f'{mode}/tsne_{epoch}.png')
        # self.draw_dim_reduced(test_feats, test_classes, method='tsne', title=f"{mode}, epoch: " + str(epoch),
        #                       path=tsne_path)

        if epoch != -1:
            diff_class_path = os.path.join(self.gen_plot_path, f'{args.dataset_name}_{mode}/class_diff_plot.png')
            if return_bg and mode != 'train':
                self.plot_class_diff_plots(test_feats, test_classes,
                                           epoch=epoch,
                                           mode=mode,
                                           path=diff_class_path,
                                           img_seen=test_seen, attention=has_attention)
            else:
                self.plot_class_diff_plots(test_feats, test_classes,
                                           epoch=epoch,
                                           mode=mode,
                                           path=diff_class_path, attention=has_attention)

        silhouette_path = ['', '']
        silhouette_path[0] = os.path.join(self.gen_plot_path, f'{args.dataset_name}_{mode}/silhouette_scores_plot.png')
        silhouette_path[1] = os.path.join(self.gen_plot_path,
                                          f'{args.dataset_name}_{mode}/silhouette_scores_dist_plot_{epoch}.png')

        if mode != 'test':

            if mode == 'val':
                tb_tag = 'Val'
            elif mode == 'train':
                tb_tag = 'Train'
            else:
                tb_tag = 'Other'

            self.plot_silhouette_score(test_feats, test_classes, epoch, mode, silhouette_path,
                                       f'Total_{tb_tag}', attention=has_attention)

            if return_bg and mode == 'val':
                silhouette_path[0] = os.path.join(self.gen_plot_path,
                                                  f'{args.dataset_name}_{mode}/silhouette_scores_plot_seen.png')
                silhouette_path[1] = os.path.join(self.gen_plot_path,
                                                  f'{args.dataset_name}_{mode}/silhouette_scores_dist_plot_{epoch}_seen.png')
                self.plot_silhouette_score(test_feats[test_seen == 1], test_classes[test_seen == 1], epoch,
                                           mode + '_seen', silhouette_path,
                                           f'seen_{tb_tag}', attention=has_attention)

                silhouette_path[0] = os.path.join(self.gen_plot_path,
                                                  f'{args.dataset_name}_{mode}/silhouette_scores_plot_unseen.png')
                silhouette_path[1] = os.path.join(self.gen_plot_path,
                                                  f'{args.dataset_name}_{mode}/silhouette_scores_dist_plot_{epoch}_unseen.png')

                self.plot_silhouette_score(test_feats[test_seen == 0], test_classes[test_seen == 0], epoch,
                                           mode + '_unseen', silhouette_path,
                                           f'unseen_{tb_tag}', attention=has_attention)

        # import pdb
        # pdb.set_trace()
        if k_at_n:
            kavg = utils.calculate_k_at_n(args, test_feats, test_classes, test_seen, logger=self.logger,
                                          limit=args.limit_samples,
                                          run_number=args.number_of_runs,
                                          save_path=self.save_path,
                                          sampled=True,
                                          even_sampled=False,
                                          per_class=eval_per_class,
                                          mode=mode,
                                          metric=self.metric)

            if epoch != -1:
                for c in list(kavg.columns): # plot tb
                    if 'kAT' in c:
                        tb_tag = c.replace('AT', '@')
                        cmode = mode[0].upper() + mode[1:]  # capitalize

                        if return_bg and mode == 'val':
                            if 'unseen' in c:
                                self.writer.add_scalar(f'unseen_{cmode}/{tb_tag}', kavg[c][0], epoch)
                            elif 'seen' in c:
                                self.writer.add_scalar(f'seen_{cmode}/{tb_tag}', kavg[c][0], epoch)
                        if 'seen' not in c:
                            self.writer.add_scalar(f'Total_{cmode}/{tb_tag}', kavg[c][0], epoch)

                self.writer.flush()
        self.logger.info('results at: ' + self.save_path)

    def load_model(self, args, net, best_model):
        if args.cuda:
            checkpoint = torch.load(os.path.join(self.save_path, best_model))
        else:
            checkpoint = torch.load(os.path.join(self.save_path, best_model), map_location=torch.device('cpu'))
        self.logger.info('Loading model %s from epoch [%d]' % (best_model, checkpoint['epoch']))
        o_dic = checkpoint['model_state_dict']
        exp = True
        counter = 1
        exp_msg = ''
        while exp and counter < 4:
            try:
                net.load_state_dict(o_dic)
                exp = False
            except Exception as e:
                exp_msg = e
                counter += 1
                self.logger.info(str(exp))
                new_o_dic = collections.OrderedDict()
                for k, v in o_dic.items():
                    new_o_dic[k[7:]] = v
                o_dic = new_o_dic
        if exp:
            raise Exception(exp_msg)
        return net

    def draw_dim_reduced(self, features, labels, title, path, method='pca'):
        """

        :param features:
        :param labels:
        :param method: pce or tsne
        :return: None
        """

        if method == 'pca':
            from sklearn.decomposition import PCA
            model = PCA(n_components=2, random_state=0)

        elif method == 'tsne':
            from sklearn.manifold import TSNE
            model = TSNE(n_components=2, random_state=0)
        else:
            raise Exception('Not acceptable method')

        feats_reduced = model.fit_transform(features)

        tsne_data = np.vstack((feats_reduced.T, labels)).T

        df = pd.DataFrame(data=tsne_data, columns=('dim1', 'dim2', 'label'))

        df.plot.scatter(x='dim1',
                        y='dim2',
                        c='label',
                        colormap='plasma')

        plt.title(method + " " + title)
        plt.savefig(path)
        plt.close('all')

    def save_model(self, args, net, epoch, val_acc):
        best_model = 'model-epoch-' + str(epoch) + '-val-acc-' + str(val_acc) + '.pt'
        torch.save({'epoch': epoch, 'model_state_dict': net.state_dict()},
                   self.save_path + '/' + best_model)
        return best_model

    def getBack(self, var_grad_fn):
        self.logger.info(str(var_grad_fn))
        for n in var_grad_fn.next_functions:
            if n[0]:
                try:
                    tensor = getattr(n[0], 'variable')
                    self.logger.info(str(n[0]))
                    self.logger.info(f'Tensor with grad found: {tensor}')
                    self.logger.info(f' - gradient: {tensor.grad}')
                    self.logger.info('\n')
                except AttributeError as e:
                    self.getBack(n[0])

    def plot_grad_flow(self, named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                if p.grad is None:
                    self.logger.info(f'{n}, {p}')
                    continue
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

        return plt

    # extra, for having an extra linear classifier
    def linear_classifier(self, emb, classifier, metric, trained=False):

        if not trained:
            classifier.fit(emb[0], emb[1])

        preds = classifier.predict(emb[0])
        metric.update_acc(preds, emb[1])
        return classifier, metric

    # extra, for having an extra linear classifier
    def get_embeddings(self, args, net, data_loader, batch_size=None):

        net.eval()
        # device = f'cuda:{net.device_ids[0]}'
        if batch_size is None:
            batch_size = args.db_batch

        if args.dim_reduction != 0:
            embs = np.zeros((len(data_loader.dataset), args.dim_reduction), dtype=np.float32)
        elif args.feat_extractor == 'resnet50':
            embs = np.zeros((len(data_loader.dataset), 2048), dtype=np.float32)
        elif args.feat_extractor == 'resnet18' or args.feat_extractor == 'vgg16':
            embs = np.zeros((len(data_loader.dataset), 512), dtype=np.float32)
        else:
            raise Exception('Arch not handled for "get_embeddings" function')

        labels = np.zeros((len(data_loader.dataset)))
        seens = np.zeros((len(data_loader.dataset)))
        ids = np.zeros((len(data_loader.dataset)))

        with tqdm(total=len(data_loader), desc='Storing embeddings...') as t:
            for idx, tpl in enumerate(data_loader):

                if len(tpl) == 4:
                    img, lbl, seen, id = tpl
                else:
                    img, lbl, id = tpl
                    seen = -1

                if args.cuda:
                    img = img.cuda()
                img = Variable(img)

                output = net.forward(img)  # for only a resnet (not custom to my implementation)
                output = output.data.cpu().numpy()

                end = min((idx + 1) * batch_size, len(embs))

                embs[idx * batch_size:end, :] = output
                labels[idx * batch_size:end] = lbl
                ids[idx * batch_size:end] = id

                if len(tpl) == 4:
                    seens[idx * batch_size:end] = seen.to(int)
                t.update()

        return embs, labels, seens, ids

    def apply_edgepred_eval(self, args, net, data_loader):

        right, error = 0, 0
        net.eval()
        # device = f'cuda:{net.device_ids[0]}'
        true_label = torch.Tensor([[i for _ in range(args.test_k)] for i in range(args.way)]).flatten()
        loss = 0
        if args.cuda:
            true_label = Variable(true_label.cuda())
        else:
            true_label = Variable(true_label)
        all_predictions = []

        for _, (img, labels) in enumerate(data_loader, 1):
            if args.cuda:
                img = img.cuda()
            img = Variable(img)

            features = net.forward(img, None, single=True)

            # loss += loss_fn(pred_vector.reshape((-1,)), label.reshape((-1,))).item()

            true_edge_probs = []

            pred_edge_probs = []

            pred_vector = pred_vector.reshape((-1,)).data.cpu().numpy()
            all_predictions.extend(pred_vector)
            # high_confidence_false_positives_idxs = utils.sigmoid(pred_vector) > 0.8
            pred = np.argmax(pred_vector)
            if pred == 0:
                right += 1
            else:
                error += 1

        acc = right * 1.0 / (right + error)

        return acc, loss, right, error, all_predictions

    def apply_fewshot_eval(self, args, net, data_loader, loss_fn):

        right, error = 0, 0
        net.eval()
        # device = f'cuda:{net.device_ids[0]}'
        label = np.zeros(shape=args.way, dtype=np.float32)
        label[0] = 1
        label = torch.from_numpy(label)
        loss = 0
        if args.cuda:
            label = Variable(label.cuda())
        else:
            label = Variable(label)
        all_predictions = []
        for _, (img1, img2) in enumerate(data_loader, 1):
            if args.cuda:
                img1, img2 = img1.cuda(), img2.cuda()
            img1, img2 = Variable(img1), Variable(img2)
            pred_vector, dist = net.forward(img1, img2)
            loss += loss_fn(pred_vector.reshape((-1,)), label.reshape((-1,))).item()
            pred_vector = pred_vector.reshape((-1,)).data.cpu().numpy()
            all_predictions.extend(pred_vector)
            # high_confidence_false_positives_idxs = utils.sigmoid(pred_vector) > 0.8
            pred = np.argmax(pred_vector)
            if pred == 0:
                right += 1
            else:
                error += 1

        acc = right * 1.0 / (right + error)
        loss /= len(data_loader)

        return acc, loss, right, error, all_predictions

    def get_loss_value(self, args, loss_fn, anch_feat, pos_feat, neg_feat):

        pos_dist = torch.pow((anch_feat - pos_feat), 2)
        neg_dist = torch.pow((anch_feat - neg_feat), 2)

        if args.loss == 'trpl':
            loss = loss_fn(pos_dist, neg_dist)
            parts = []
        elif args.loss == 'maxmargin':
            loss, parts = loss_fn(pos_dist, neg_dist)
        elif args.loss == 'batchhard':
            loss_fn_trpl = TripletLoss(margin=args.margin, args=args, soft=args.softmargin)
            loss = loss_fn_trpl(pos_dist, neg_dist)
            parts = []
        else:
            raise Exception('Loss function not supported in get_loss_value() method')

        return loss, parts

    # todo make customized dataloader for cam
    # todo easy cases?
    def plot_class_diff_plots(self, img_feats, img_classes, epoch, mode, path, img_seen=None, attention=False):

        if self.metric == 'cosine':
            sims = img_feats.dot(img_feats.T)
            max_sim = np.max(sims)
            dists = -sims
            dists += max_sim
            np.fill_diagonal(dists, 0)

            # dists = utils.calc_custom_euc(img_feats, chunks=4)  # todo hardcoded

        elif self.metric == 'euclidean':
            dists = euclidean_distances(img_feats)
        else:
            raise Exception(f'NoT SuPpoRTeD metric: {self.metric}')

        res = utils.get_distances(dists, img_classes)

        reses = [res]
        modes = [mode]
        paths = [path]

        if img_seen is not None:
            res_seen = utils.get_distances(dists[img_seen == 1, :][:, img_seen == 1], img_classes[img_seen == 1])
            res_unseen = utils.get_distances(dists[img_seen == 0, :][:, img_seen == 0], img_classes[img_seen == 0])

            reses.append(res_seen)
            modes.append(f'{mode}_seen')
            paths.append(path[:path.rfind('.')] + f'_{mode}_seen' + path[path.rfind('.'):])

            reses.append(res_unseen)
            modes.append(f'{mode}_unseen')
            paths.append(path[:path.rfind('.')] + f'_{mode}_unseen' + path[path.rfind('.'):])

        for m, r, p in zip(modes, reses, paths):
            for k, v in self.class_diffs[m].items():
                v.append(r[k])

            colors = ['r', 'b', 'y', 'g', 'c', 'm']
            epochs = [i for i in range(1, epoch + 1)]
            legends = []
            colors_reordered = []

            plt.figure(figsize=(10, 10))
            for (k, v), c in zip(self.class_diffs[m].items(), colors):
                if len(v) > 1:
                    plt.plot(epochs, v, color=c, linewidth=2, markersize=12)
                else:
                    plt.scatter(epochs, v, color=c)
                legends.append(k)
                colors_reordered.append(c)

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Euclidean Distance')
            plt.xlim(left=0, right=epoch + 5)
            plt.legend([Line2D([0], [0], color=colors_reordered[0], lw=4),
                        Line2D([0], [0], color=colors_reordered[1], lw=4),
                        Line2D([0], [0], color=colors_reordered[2], lw=4),
                        Line2D([0], [0], color=colors_reordered[3], lw=4),
                        Line2D([0], [0], color=colors_reordered[4], lw=4),
                        Line2D([0], [0], color=colors_reordered[5], lw=4)], legends)

            plt.title(f'{m} class diffs')

            plt.savefig(p)
            plt.close('all')

    def plot_silhouette_score(self, X, labels, epoch, mode, path, tb_tag, attention=False):

        if self.metric == 'cosine':
            sims = X.dot(X.T)
            max_sim = np.max(sims)
            dists = -sims
            dists += max_sim
            np.fill_diagonal(dists, 0)

            # dists = utils.calc_custom_euc(X, chunks=4)  # todo chunks hardcoded

        elif self.metric == 'euclidean':
            dists = euclidean_distances(X)
        else:
            raise Exception(f'Metric {self.metric} not supported')

        last_silh_score = silhouette_score(dists, labels, metric='precomputed')
        self.silhouette_scores[mode].append(last_silh_score)

        samples_silhouette = silhouette_samples(X, labels)

        if epoch != -1:
            self.writer.add_scalar(tb_tag + '/Silhouette_Score', last_silh_score, epoch)
            self.writer.add_histogram('Silhouette_Scores/' + tb_tag, samples_silhouette, epoch)
        else:
            self.writer.add_scalar(tb_tag + '/Final_Silhouette_Score', last_silh_score, epoch)
            self.writer.add_histogram('Final_Silhouette_Scores/' + tb_tag, samples_silhouette, epoch)
        self.writer.flush()

        if epoch != -1:
            epochs = [i for i in range(1, epoch + 1)]

            plt.figure(figsize=(10, 10))
            if len(self.silhouette_scores[mode]) > 1:
                plt.plot(epochs, self.silhouette_scores[mode], linewidth=2, markersize=12)
            else:
                plt.scatter(epochs, self.silhouette_scores[mode])

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel(f'Silhouette Score')
            plt.xlim(left=0, right=epoch + 5)

            plt.title(f'Silhouette Scores for {mode} set')

            plt.savefig(path[0])

        plt.close('all')

        plt.figure(figsize=(10, 10))

        plt.hist(samples_silhouette, bins=40)

        plt.grid(True)
        plt.xlabel('Silhouette Score')
        plt.ylabel(f'Freq')
        plt.xlim(left=-1.1, right=1.1)

        plt.title(f'Silhouette Scores Distribution on {mode} set')

        plt.savefig(path[1])
        plt.close('all')

    def train_metriclearning_one_epoch(self, args, t, net, opt, bce_loss, metric_ACC, loss_fn, train_loader, epoch,
                                       grad_save_path, drew_graph):
        train_loss = 0
        train_bce_loss = 0
        train_triplet_loss = 0
        pos_parts = []
        neg_parts = []

        metric_ACC.reset_acc()

        merged_vectors = {}

        pos_all_merged_vectors = None
        neg_all_merged_vectors = None

        for batch_id, (anch, pos, neg) in enumerate(train_loader, 1):
            start = time.time()
            # self.logger.info('input: ', img1.size())

            debug_grad = self.draw_grad and (batch_id == 1 or batch_id == len(train_loader))

            one_labels = torch.tensor([1 for _ in range(anch.shape[0])], dtype=float)
            zero_labels = torch.tensor([0 for _ in range(anch.shape[0])], dtype=float)

            if args.cuda:
                anch, pos, neg, one_labels, zero_labels = Variable(anch.cuda()), \
                                                          Variable(pos.cuda()), \
                                                          Variable(neg.cuda()), \
                                                          Variable(one_labels.cuda()), \
                                                          Variable(zero_labels.cuda())
            else:
                anch, pos, neg, one_labels, zero_labels = Variable(anch), \
                                                          Variable(pos), \
                                                          Variable(neg), \
                                                          Variable(one_labels), \
                                                          Variable(zero_labels)

            if not drew_graph:
                self.writer.add_graph(net, (anch.detach(), pos.detach()), verbose=True)
                self.writer.flush()
                drew_graph = True

            net.train()
            # device = f'cuda:{net.device_ids[0]}'
            opt.zero_grad()
            forward_start = time.time()
            pos_pred, pos_dist, anch_feat, pos_feat = net.forward(anch, pos, feats=True)
            forward_end = time.time()

            if pos_all_merged_vectors is None:
                pos_all_merged_vectors = pos_dist.data.cpu()
            else:
                pos_all_merged_vectors = torch.cat([pos_all_merged_vectors, pos_dist.data.cpu()], dim=0)

            if utils.MY_DEC.enabled:
                self.logger.info(f'########### anch pos forward time: {forward_end - forward_start}')

            # if args.verbose:
            #     self.logger.info(f'norm pos: {pos_dist}')
            class_loss = bce_loss(pos_pred.squeeze(), one_labels.squeeze())
            metric_ACC.update_acc(pos_pred.squeeze(), one_labels.squeeze())  # zero dist means similar

            for neg_iter in range(self.no_negative):
                forward_start = time.time()
                neg_pred, neg_dist, _, neg_feat = net.forward(anch, neg[:, neg_iter, :, :, :].squeeze(dim=1),
                                                              feats=True)

                if neg_all_merged_vectors is None:
                    neg_all_merged_vectors = neg_dist.data.cpu()
                else:
                    neg_all_merged_vectors = torch.cat([neg_all_merged_vectors, neg_dist.data.cpu()], dim=0)

                forward_end = time.time()
                if utils.MY_DEC.enabled:
                    self.logger.info(f'########### anch-neg forward time: {forward_end - forward_start}')
                # neg_dist.register_hook(lambda x: self.logger.info(f'neg_dist grad:{x}'))
                # neg_pred.register_hook(lambda x: self.logger.info(f'neg_pred grad:{x}'))

                # if args.verbose:
                #     self.logger.info(f'norm neg {neg_iter}: {neg_dist}')

                metric_ACC.update_acc(neg_pred.squeeze(), zero_labels.squeeze())  # 1 dist means different

                class_loss += bce_loss(neg_pred.squeeze(), zero_labels.squeeze())
                if loss_fn is not None:
                    ext_batch_loss, parts = self.get_loss_value(args, loss_fn, anch_feat, pos_feat, neg_feat)

                    if neg_iter == 0:
                        ext_loss = ext_batch_loss
                    else:
                        ext_loss += ext_batch_loss

                    if args.loss == 'maxmargin':
                        if neg_iter == 0:
                            pos_parts.extend(parts[0].tolist())
                        neg_parts.extend(parts[1].tolist())

            class_loss /= (self.no_negative + 1)

            if loss_fn is not None:
                ext_loss /= self.no_negative
                loss = self.trpl_weight * ext_loss + self.bce_weight * class_loss
                train_triplet_loss += ext_loss.item()

                if debug_grad:
                    ext_loss.backward(retain_graph=True)
                    triplet_loss_named_parameters = net.named_parameters()

                    trpl_ave_grads = []
                    trpl_max_grads = []
                    layers = []
                    for n, p in net.named_parameters():
                        if (p.requires_grad) and ("bias" not in n):
                            if n == 'ft_net.fc.weight':
                                continue
                            if p.grad is None:
                                trpl_ave_grads.append(torch.Tensor([0.0]))
                                trpl_max_grads.append(torch.Tensor([0.0]))
                            else:
                                trpl_ave_grads.append(p.grad.abs().mean())
                                trpl_max_grads.append(p.grad.abs().max())

                            layers.append(n)

                    self.logger.info('got triplet loss grads')

                    # utils.line_plot_grad_flow(args, net.named_parameters(), 'TRIPLETLOSS', batch_id, epoch,
                    #                           grad_save_path)
                    opt.zero_grad()

            else:
                loss = self.bce_weight * class_loss

            train_loss += loss.item()
            train_bce_loss += class_loss.item()

            if debug_grad:
                lambda_class_loss = self.bce_weight * class_loss
                lambda_class_loss.backward(retain_graph=True)

                bce_named_parameters = net.named_parameters()
                bce_named_parameters = {k: v for k, v in bce_named_parameters}

                bce_ave_grads = []
                bce_max_grads = []
                for n, p in net.named_parameters():
                    if (p.requires_grad) and ("bias" not in n):
                        if n == 'ft_net.fc.weight':
                            continue
                        if p.grad is None:
                            continue

                        bce_ave_grads.append(p.grad.abs().mean())
                        bce_max_grads.append(p.grad.abs().max())

                # utils.bar_plot_grad_flow(args, [trpl_ave_grads, trpl_max_grads, layers], 'TRIPLETLOSS', batch_id,
                #                          epoch, grad_save_path)
                #
                # utils.bar_plot_grad_flow(args, [bce_ave_grads, bce_max_grads, layers], 'BCE', batch_id, epoch,
                #                          grad_save_path)

                self.logger.info('got bce grads')

                if loss_fn is None:
                    utils.bar_plot_grad_flow(args, net.named_parameters(), 'BCE', batch_id, epoch,
                                             grad_save_path)
                    utils.line_plot_grad_flow(args, net.named_parameters(), 'BCE', batch_id, epoch,
                                              grad_save_path)
                else:
                    # utils.bar_plot_grad_flow(args, triplet_loss_named_parameters,
                    #                          'TRIPLET', batch_id, epoch, grad_save_path)
                    # utils.bar_plot_grad_flow(args, bce_named_parameters,
                    #                          'BCE', batch_id, epoch, grad_save_path)
                    utils.two_line_plot_grad_flow(args, [trpl_ave_grads, trpl_max_grads, layers],
                                                  [bce_ave_grads, bce_max_grads, layers],
                                                  'BOTH', batch_id, epoch, grad_save_path)
                    # import pdb
                    # pdb.set_trace()
                    utils.two_bar_plot_grad_flow(args, [trpl_ave_grads, trpl_max_grads, layers],
                                                 [bce_ave_grads, bce_max_grads, layers],
                                                 'BOTH', batch_id, epoch, grad_save_path)

                opt.zero_grad()

            loss.backward()  # training with triplet loss

            # if debug_grad:
            #     utils.bar_plot_grad_flow(args, net.named_parameters(), 'total', batch_id, epoch, grad_save_path)
            #     utils.line_plot_grad_flow(args, net.named_parameters(), 'total', batch_id, epoch,
            #                               grad_save_path)

            opt.step()

            if loss_fn is not None:
                t.set_postfix(loss=f'{train_loss / (batch_id) :.4f}',
                              bce_loss=f'{train_bce_loss / batch_id:.4f}',
                              triplet_loss=f'{train_triplet_loss / batch_id:.4f}',
                              train_acc=f'{metric_ACC.get_acc():.4f}'
                              )
            else:
                t.set_postfix(loss=f'{train_loss / (batch_id) :.4f}',
                              bce_loss=f'{train_bce_loss / batch_id:.4f}',
                              train_acc=f'{metric_ACC.get_acc():.4f}'
                              )

            t.update()
            end = time.time()
            if utils.MY_DEC.enabled:
                self.logger.info(f'########### one batch time: {end - start}')

        if self.merge_method == 'diff-sim':

            merged_vectors['pos-diff'] = pos_all_merged_vectors[:, :(pos_all_merged_vectors.shape[1] // 2)]
            merged_vectors['pos-sim'] = pos_all_merged_vectors[:, (pos_all_merged_vectors.shape[1] // 2):]

            merged_vectors['neg-diff'] = neg_all_merged_vectors[:, :(neg_all_merged_vectors.shape[1] // 2)]
            merged_vectors['neg-sim'] = neg_all_merged_vectors[:, (neg_all_merged_vectors.shape[1] // 2):]
        else:
            merged_vectors[f'pos-{self.merge_method}'] = pos_all_merged_vectors
            merged_vectors[f'neg-{self.merge_method}'] = neg_all_merged_vectors

        for name, param in merged_vectors.items():
            self.writer.add_histogram(name, param.flatten(), epoch)
            self.writer.flush()

        return t, (train_loss, train_bce_loss, train_triplet_loss), (pos_parts, neg_parts)

    def train_metriclearning_one_epoch_batchhard(self, args, t, net, opt, bce_loss, metric_ACC, loss_fn, train_loader,
                                                 epoch,
                                                 grad_save_path, drew_graph):
        train_loss = 0
        train_bce_loss = 0
        train_batchhard_loss = 0

        metric_ACC.reset_acc()

        merged_vectors = {}

        pos_all_merged_vectors = None
        neg_all_merged_vectors = None

        labels = torch.Tensor([[i for _ in range(args.bh_K)] for i in range(args.bh_P)]).flatten()
        if args.cuda:
            labels = Variable(labels.cuda())
        else:
            labels = Variable(labels)

        for batch_id, imgs in enumerate(train_loader, 1):
            imgs = imgs.reshape(-1, imgs.shape[2], imgs.shape[3], imgs.shape[4])
            start = time.time()
            # self.logger.info('input: ', img1.size())

            debug_grad = self.draw_grad and (batch_id == 1 or batch_id == len(train_loader))

            one_labels = torch.tensor([1 for _ in range(imgs.shape[0])], dtype=float)
            zero_labels = torch.tensor([0 for _ in range(imgs.shape[0])], dtype=float)

            if args.cuda:
                imgs, one_labels, zero_labels = Variable(imgs.cuda()), \
                                                Variable(one_labels.cuda()), \
                                                Variable(zero_labels.cuda())
            else:
                imgs, one_labels, zero_labels = Variable(imgs), \
                                                Variable(one_labels), \
                                                Variable(zero_labels)

            if not drew_graph:
                self.writer.add_graph(net, (imgs.detach(), imgs.detach()), verbose=True)
                self.writer.flush()
                drew_graph = True

            net.train()
            # device = f'cuda:{net.device_ids[0]}'
            opt.zero_grad()
            forward_start = time.time()
            imgs_f, imgs_l = net.ft_net(imgs, is_feat=True)
            forward_end = time.time()

            imgs_f = imgs_f.view(imgs_f.size()[0], -1)

            ext_loss, (hard_pos_idx, hard_neg_idx) = loss_fn(imgs_f, labels, get_idx=True)

            pos_f = imgs_f[hard_pos_idx, :]
            neg_f = imgs_f[hard_neg_idx, :]

            pos_f = pos_f.view(pos_f.size()[0], -1)
            neg_f = neg_f.view(neg_f.size()[0], -1)

            pos_pred, pos_dist = net.sm_net(imgs_f, pos_f)

            neg_pred, neg_dist = net.sm_net(imgs_f, neg_f)

            if pos_all_merged_vectors is None:
                pos_all_merged_vectors = pos_dist.data.cpu()
            else:
                pos_all_merged_vectors = torch.cat([pos_all_merged_vectors, pos_dist.data.cpu()], dim=0)

            if utils.MY_DEC.enabled:
                self.logger.info(f'########### anch pos forward time: {forward_end - forward_start}')

            # if args.verbose:
            #     self.logger.info(f'norm pos: {pos_dist}')
            class_loss = bce_loss(pos_pred.squeeze(), one_labels.squeeze())
            metric_ACC.update_acc(pos_pred.squeeze(), one_labels.squeeze())  # zero dist means similar

            if neg_all_merged_vectors is None:
                neg_all_merged_vectors = neg_dist.data.cpu()
            else:
                neg_all_merged_vectors = torch.cat([neg_all_merged_vectors, neg_dist.data.cpu()], dim=0)

            forward_end = time.time()
            if utils.MY_DEC.enabled:
                self.logger.info(f'########### anch-neg forward time: {forward_end - forward_start}')

            metric_ACC.update_acc(neg_pred.squeeze(), zero_labels.squeeze())  # 1 dist means different

            class_loss += bce_loss(neg_pred.squeeze(), zero_labels.squeeze())

            if loss_fn is not None:

                loss = self.trpl_weight * ext_loss + self.bce_weight * class_loss
                train_batchhard_loss += ext_loss.item()

            else:
                loss = self.bce_weight * class_loss

            train_loss += loss.item()
            train_bce_loss += class_loss.item()

            if debug_grad:
                raise Exception('Debug grad not implemented for batchhard')

            loss.backward()  # training with triplet loss

            opt.step()

            if loss_fn is not None:
                t.set_postfix(loss=f'{train_loss / (batch_id) :.4f}',
                              bce_loss=f'{train_bce_loss / batch_id:.4f}',
                              batchhard=f'{train_batchhard_loss / batch_id:.4f}',
                              train_acc=f'{metric_ACC.get_acc():.4f}'
                              )
            else:
                t.set_postfix(loss=f'{train_loss / (batch_id) :.4f}',
                              bce_loss=f'{train_bce_loss / batch_id:.4f}',
                              train_acc=f'{metric_ACC.get_acc():.4f}'
                              )

            t.update()
            end = time.time()
            if utils.MY_DEC.enabled:
                self.logger.info(f'########### one batch time: {end - start}')

        if self.merge_method == 'diff-sim':

            merged_vectors['pos-diff'] = pos_all_merged_vectors[:, :(pos_all_merged_vectors.shape[1] // 2)]
            merged_vectors['pos-sim'] = pos_all_merged_vectors[:, (pos_all_merged_vectors.shape[1] // 2):]

            merged_vectors['neg-diff'] = neg_all_merged_vectors[:, :(neg_all_merged_vectors.shape[1] // 2)]
            merged_vectors['neg-sim'] = neg_all_merged_vectors[:, (neg_all_merged_vectors.shape[1] // 2):]
        else:
            merged_vectors[f'pos-{self.merge_method}'] = pos_all_merged_vectors
            merged_vectors[f'neg-{self.merge_method}'] = neg_all_merged_vectors

        for name, param in merged_vectors.items():
            self.writer.add_histogram(name, param.flatten(), epoch)
            self.writer.flush()

        return t, (train_loss, train_bce_loss, train_batchhard_loss), ([], [])

    def plot_classifier_hist(self, classifier_weights_methods, titles, plot_title, save_path, tb_title, epoch):

        plt.figure(figsize=(15, 15))
        legends = list(map(lambda x: x + ' value distribution', titles))
        colors = []
        lines = []
        if len(classifier_weights_methods) >= 1:
            colors = ['b']
            lines = [Line2D([0], [0], color="b", lw=4)]

        if len(classifier_weights_methods) >= 2:
            colors += ['r']
            lines += [Line2D([0], [0], color="r", lw=4)]

        if len(classifier_weights_methods) >= 3:
            colors += ['g']
            lines += [Line2D([0], [0], color="g", lw=4)]

        if len(classifier_weights_methods) >= 4:
            colors += ['y']
            lines += [Line2D([0], [0], color="y", lw=4)]

        if len(classifier_weights_methods) >= 5:
            colors += ['c']
            lines += [Line2D([0], [0], color="c", lw=4)]

        if len(classifier_weights_methods) >= 6:
            colors += ['m']
            lines += [Line2D([0], [0], color="m", lw=4)]

        if len(classifier_weights_methods) >= 7:
            colors += ['darkblue']
            lines += [Line2D([0], [0], color="darkblue", lw=4)]

        if len(classifier_weights_methods) >= 8:
            colors += ['peru']
            lines += [Line2D([0], [0], color="peru", lw=4)]

        if len(classifier_weights_methods) >= 9:
            raise Exception('too many sub method types for plotting')

        max = 0
        min = 100
        for act, title, color in zip(classifier_weights_methods, titles, colors):
            flatten_act = act.flatten().cpu().numpy()
            if max < flatten_act.max():
                max = flatten_act.max()
            if min > flatten_act.min():
                min = flatten_act.min()
            plt.hist(flatten_act, bins=100, alpha=0.4, color=color)
            self.writer.add_histogram(f'{tb_title}/{title}', flatten_act, epoch, bins=25)

        self.writer.flush()
        plt.axis('on')
        plt.xlim(left=min - 0.1, right=max + 0.1)
        plt.legend(lines, legends)
        plt.title(plot_title)
        plt.savefig(save_path)
        plt.close('all')

    def update_bce_tco(self, epoch):
        if self.bcotco_freq != 0 and epoch % self.bcotco_freq == 0:
            self.bce_weight /= self.bco_base
            self.trpl_weight /= self.tco_base

            weight_sum = self.bce_weight + self.trpl_weight
            self.bce_weight /= weight_sum
            self.trpl_weight /= weight_sum

            self.logger.info(f'epoch: {epoch}, bce weight: {self.bce_weight}, tco weight: {self.trpl_weight}')
            print(f'epoch: {epoch}, bce weight: {self.bce_weight}, tco weight: {self.trpl_weight}')
        return

    def _tb_get_important_hparams(self, args):
        important_hp = {'Dataset': args.dataset_name,
                        'Cls LR': args.lr_new,
                        'ResNet LR': args.lr_resnet,
                        'weight decay': args.weight_decay,
                        'merge method': args.merge_method,
                        'Loss': args.loss,
                        'Batch size': args.batch_size,
                        'Cls extra layers': args.extra_layer}

        if args.loss == 'trpl':
            important_hp['bce coeff'] = args.bcecoefficient
            important_hp['trpl coeff'] = args.trplcoefficient

            if args.bcotco_freq != 0:
                important_hp['bcotco freq'] = args.bcotco_freq
                important_hp['bco base'] = args.bco_base
                important_hp['tco base'] = args.tco_base

        if 'diff-sim' in args.merge_method:
            important_hp['softmax-diffsim'] = args.softmax_diff_sim

        return important_hp

    def save_best_negatives(self, args, net, loader):
        embbeddings, labels, seens, ids = self.get_embeddings(args, net, loader)

        res = utils.evaluation(args, embbeddings, labels, ids, self.writer,
                               loader, Kset=[1, 2, 4, 5, 8, 10, 100, 1000], split='total', path=self.save_path,
                               gpu=args.cuda,
                               path_to_lbl2chain=os.path.join(args.splits_file_path, 'label2chain.csv'))
        self.logger.info(str(res))


class BaslineModel:
    def __init__(self, args, model, logger, loss_fn, model_name, id_str=''):
        self.logger = logger
        self.model = model
        self.loss_fn = loss_fn  # batch hard
        self.bh_k = args.bh_K
        self.bh_p = args.bh_P
        self.model_name = model_name
        self.tensorboard_path = os.path.join(args.local_path, args.tb_path, self.model_name)
        self.writer = SummaryWriter(self.tensorboard_path)

        self.save_path = os.path.join(args.local_path, args.save_path, self.model_name)
        utils.create_save_path(self.save_path, id_str, self.logger)

        self.gen_plot_path = f'{self.save_path}/plots/'

        utils.make_dirs(self.tensorboard_path)

        self.gen_plot_path = f'{self.save_path}/plots/'
        utils.make_dirs(self.gen_plot_path)
        utils.make_dirs(os.path.join(self.gen_plot_path, 'train'))
        utils.make_dirs(os.path.join(self.gen_plot_path, 'val'))

        self.class_diffs = {'train':
                                {'between_class_average': [],
                                 'between_class_min': [],
                                 'between_class_max': [],
                                 'in_class_average': [],
                                 'in_class_min': [],
                                 'in_class_max': []},
                            'val':
                                {'between_class_average': [],
                                 'between_class_min': [],
                                 'between_class_max': [],
                                 'in_class_average': [],
                                 'in_class_min': [],
                                 'in_class_max': []}}
        self.silhouette_scores = {'train': [],
                                  'val': []}

        self.aug_mask = args.aug_mask

    def train_epoch(self, t, args, train_loader, opt, epoch):
        train_loss = 0

        labels = torch.Tensor([[i for _ in range(self.bh_k)] for i in range(self.bh_p)]).flatten()
        if args.cuda:
            labels = Variable(labels.cuda())
        else:
            labels = Variable(labels)

        for batch_id, imgs in enumerate(train_loader, 1):

            imgs = imgs.reshape(-1, imgs.shape[2], imgs.shape[3], imgs.shape[4])
            start = time.time()
            # import pdb
            # pdb.set_trace()
            if args.cuda:
                imgs = Variable(imgs.cuda())
            else:
                imgs = Variable(imgs)

            # if not drew_graph:
            #     self.writer.add_graph(self.model, (imgs.detach()), verbose=True)
            #     self.writer.flush()
            #     drew_graph = True

            self.model.train()
            # device = f'cuda:{net.device_ids[0]}'
            opt.zero_grad()
            forward_start = time.time()
            feats = self.model.forward(imgs)
            forward_end = time.time()

            if utils.MY_DEC.enabled:
                self.logger.info(f'########### baseline forward time: {forward_end - forward_start}')

            # if args.verbose:
            #     self.logger.info(f'norm pos: {pos_dist}')

            loss = self.loss_fn(feats, labels)

            train_loss += loss.item()

            loss.backward()  # training with triplet loss

            opt.step()

            t.set_postfix(loss=f'{train_loss / (batch_id) :.4f}')
            t.update()

            end = time.time()
            if utils.MY_DEC.enabled:
                self.logger.info(f'########### baseline one batch time: {end - start}')

        return t, train_loss

    def train(self, args, train_loader, val_loader):

        epochs = args.epochs

        opt = torch.optim.Adam([{'params': self.model.conv1.parameters()},
                                {'params': self.model.bn1.parameters()},
                                {'params': self.model.relu.parameters()},
                                {'params': self.model.maxpool.parameters()},
                                {'params': self.model.layer1.parameters()},
                                {'params': self.model.layer2.parameters()},
                                {'params': self.model.layer3.parameters()},
                                {'params': self.model.layer4.parameters()},
                                {'params': self.model.avgpool.parameters()},
                                {'params': self.model.fc.parameters(), 'lr': args.lr_new}],
                               lr=args.lr_resnet, weight_decay=args.weight_decay)

        # opt = torch.optim.Adam([{'params': self.model.parameters()}],
        #                        lr=args.lr_resnet)
        # net.ft_net.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        opt.zero_grad()

        for epoch in range(1, epochs + 1):
            with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}') as t:

                all_batches_start = time.time()

                utils.print_gpu_stuff(args.cuda, 'before train epoch')

                t, train_loss = self.train_epoch(t, args, train_loader, opt, epoch)

            if (epoch) % args.test_freq == 0:
                self.logger.info(f'Eval: epoch {epoch}')
                print(f'Eval: epoch {epoch}')
                self.model.eval()

                self.make_emb_db(args, val_loader,
                                 eval_sampled=True,
                                 eval_per_class=True,
                                 mode='val',
                                 epoch=epoch,
                                 k_at_n=True)

            # self.logger.info(
            #     f'Train_Fewshot_Acc: {train_fewshot_acc}, Train_Fewshot_loss: {train_fewshot_loss},\n '
            #     f'Train_Fewshot _Right: {train_fewshot_right}, Train_Fewshot_Error: {train_fewshot_error}')

        self.writer.add_scalar('Train/Loss', train_loss / len(train_loader), epoch)
        # self.writer.add_scalar('Train/BCE_Loss', train_bce_loss / len(train_loader), epoch)
        # self.writer.add_scalar('Train/Fewshot_Loss', train_fewshot_loss / len(train_loader_fewshot), epoch)
        # self.writer.add_scalar('Train/Fewshot_Acc', train_fewshot_acc, epoch)
        self.writer.flush()

    def make_emb_db(self, args, data_loader, eval_sampled, eval_per_class,
                    mode='val', epoch=-1, k_at_n=True):

        self.model.eval()
        # device = f'cuda:{net.device_ids[0]}'

        batch_size = args.db_batch

        steps = int(np.ceil(len(data_loader) / batch_size))

        test_classes = np.zeros(((len(data_loader.dataset))))
        test_seen = np.zeros(((len(data_loader.dataset))))
        test_paths = np.empty(dtype='S20', shape=((len(data_loader.dataset))))
        if args.baseline_model == 'resnet50':
            test_feats = np.zeros((len(data_loader.dataset), 256))
        # elif args.feat_extractor == 'resnet18':
        #     test_feats = np.zeros((len(data_loader.dataset), 512))
        # elif args.feat_extractor == 'vgg16':
        #     test_feats = np.zeros((len(data_loader.dataset), 4096))
        else:
            raise Exception('Not handled baseline mdoel')

        with tqdm(total=len(data_loader), desc=f'Get Embeddings {epoch}') as t:
            for idx, tpl in enumerate(data_loader):

                end = min((idx + 1) * batch_size, len(test_feats))

                if mode != 'train':
                    (img, lbl, seen, path) = tpl
                else:
                    (img, lbl, path) = tpl

                if args.cuda:
                    img = img.cuda()

                img = Variable(img)

                output = self.model.forward(img)
                output = output.data.cpu().numpy()

                test_feats[idx * batch_size:end, :] = output
                test_classes[idx * batch_size:end] = lbl
                test_paths[idx * batch_size:end] = path

                if mode != 'train':
                    test_seen[idx * batch_size:end] = seen.to(int)
                t.update()

        utils.save_h5(f'{args.dataset_name}_{mode}_ids', test_paths, 'S20',
                      os.path.join(self.save_path, f'{args.dataset_name}_{mode}Ids.h5'))
        utils.save_h5(f'{args.dataset_name}_{mode}_classes', test_classes, 'i8',
                      os.path.join(self.save_path, f'{args.dataset_name}_{mode}Classes.h5'))
        utils.save_h5(f'{args.dataset_name}_{mode}_feats', test_feats, 'f',
                      os.path.join(self.save_path, f'{args.dataset_name}_{mode}Feats.h5'))
        if mode != 'train':
            utils.save_h5(f'{args.dataset_name}_{mode}_seen', test_seen, 'i2',
                          os.path.join(self.save_path, f'{args.dataset_name}_{mode}Seen.h5'))

        test_feats = utils.load_h5(f'{args.dataset_name}_{mode}_feats',
                                   os.path.join(self.save_path, f'{args.dataset_name}_{mode}Feats.h5'))
        test_classes = utils.load_h5(f'{args.dataset_name}_{mode}_classes',
                                     os.path.join(self.save_path, f'{args.dataset_name}_{mode}Classes.h5'))
        if mode != 'train':
            test_seen = utils.load_h5(f'{args.dataset_name}_{mode}_seen',
                                      os.path.join(self.save_path, f'{args.dataset_name}_{mode}Seen.h5'))

        if epoch != -1:
            diff_class_path = os.path.join(self.gen_plot_path, f'{args.dataset_name}_{mode}/class_diff_plot.png')
            self.plot_class_diff_plots(test_feats, test_classes,
                                       epoch=epoch,
                                       mode=mode,
                                       path=diff_class_path)

        silhouette_path = ['', '']
        silhouette_path[0] = os.path.join(self.gen_plot_path, f'{args.dataset_name}_{mode}/silhouette_scores_plot.png')
        silhouette_path[1] = os.path.join(self.gen_plot_path,
                                          f'{args.dataset_name}_{mode}/silhouette_scores_dist_plot_{epoch}.png')

        self.plot_silhouette_score(test_feats, test_classes, epoch, mode, silhouette_path)

        # import pdb
        # pdb.set_trace()
        if k_at_n:
            utils.calculate_k_at_n(args, test_feats, test_classes, test_seen, logger=self.logger,
                                   limit=args.limit_samples,
                                   run_number=args.number_of_runs,
                                   save_path=self.save_path,
                                   sampled=eval_sampled,
                                   even_sampled=False,
                                   per_class=eval_per_class,
                                   mode=mode)

            self.logger.info('results at: ' + self.save_path)

    def plot_class_diff_plots(self, img_feats, img_classes, epoch, mode, path):
        dists = cosine_distances(img_feats)
        res = utils.get_distances(dists, img_classes)
        for k, v in self.class_diffs[mode].items():
            v.append(res[k])

        colors = ['r', 'b', 'y', 'g', 'c', 'm']
        epochs = [i for i in range(1, epoch + 1)]
        legends = []
        colors_reordered = []

        plt.figure(figsize=(10, 10))
        for (k, v), c in zip(self.class_diffs[mode].items(), colors):
            if len(v) > 1:
                plt.plot(epochs, v, color=c, linewidth=2, markersize=12)
            else:
                plt.scatter(epochs, v, color=c)
            legends.append(k)
            colors_reordered.append(c)

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Euclidean Distance')
        plt.xlim(left=0, right=epoch + 5)
        plt.legend([Line2D([0], [0], color=colors_reordered[0], lw=4),
                    Line2D([0], [0], color=colors_reordered[1], lw=4),
                    Line2D([0], [0], color=colors_reordered[2], lw=4),
                    Line2D([0], [0], color=colors_reordered[3], lw=4),
                    Line2D([0], [0], color=colors_reordered[4], lw=4),
                    Line2D([0], [0], color=colors_reordered[5], lw=4)], legends)

        plt.title(f'{mode} class diffs')

        plt.savefig(path)
        plt.close('all')

    def plot_silhouette_score(self, X, labels, epoch, mode, path):

        self.silhouette_scores[mode].append(silhouette_score(X, labels, metric='cosine'))
        samples_silhouette = silhouette_samples(X, labels)

        if epoch != -1:
            epochs = [i for i in range(1, epoch + 1)]

            plt.figure(figsize=(10, 10))
            if len(self.silhouette_scores[mode]) > 1:
                plt.plot(epochs, self.silhouette_scores[mode], linewidth=2, markersize=12)
            else:
                plt.scatter(epochs, self.silhouette_scores[mode])

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel(f'Silhouette Score')
            plt.xlim(left=0, right=epoch + 5)

            plt.title(f'Silhouette Scores for {mode} set')

            plt.savefig(path[0])

        plt.close('all')

        plt.figure(figsize=(10, 10))

        plt.hist(samples_silhouette, bins=40)

        plt.grid(True)
        plt.xlabel('Silhouette Score')
        plt.ylabel(f'Freq')
        plt.xlim(left=-1.1, right=1.1)

        plt.title(f'Silhouette Scores Distribution on {mode} set')

        plt.savefig(path[1])
        plt.close('all')
