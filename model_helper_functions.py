import datetime
import os
import pickle
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import metrics
import utils


class ModelMethods:

    def __init__(self, args, logger, model='top'):  # res or top
        id_str = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')
        id_str = '-time_' + id_str.replace('.', '-')

        self.model = model
        self.model_name = self._parse_args(args)

        self.no_negative = args.no_negative
        self.bce_weight = args.bcecoefficient

        self.tensorboard_path = os.path.join(args.tb_path, self.model_name + id_str)
        self.logger = logger
        self.writer = SummaryWriter(self.tensorboard_path)

        if args.pretrained_model_dir == '':
            self.save_path = os.path.join(args.save_path, self.model_name + id_str)
            utils.create_save_path(self.save_path, id_str, self.logger)
        else:
            self.logger.info(f"Using pretrained path... \nargs.pretrained_model_dir: {args.pretrained_model_dir}")
            self.save_path = os.path.join(args.save_path, args.pretrained_model_dir)

        self.logger.info("** Save path: " + self.save_path)
        self.logger.info("** Tensorboard path: " + self.tensorboard_path)

        if args.debug_grad:
            self.draw_grad = True
            self.plt_save_path = f'{self.save_path}/loss_plts/'
            os.mkdir(self.plt_save_path)
        else:
            self.draw_grad = False
            self.plt_save_path = ''

        self.created_image_heatmap_path = False

        self.scatter_plot_path = f'{self.save_path}/scatter_plots/'
        os.mkdir(self.scatter_plot_path)
        os.mkdir(os.path.join(self.scatter_plot_path, 'train'))
        os.mkdir(os.path.join(self.scatter_plot_path, 'val'))

        if args.cam:
            os.mkdir(f'{self.save_path}/heatmap/')
            self.cam_all = 0
            self.cam_neg = np.array([0 for _ in range(9)])
            self.cam_pos = np.array([0 for _ in range(9)])

    def _parse_args(self, args):
        # name = 'model-betteraug-distmlp-' + self.model
        name = 'model-' + self.model

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
                          'margin',
                          'loss',
                          'overfit_num',
                          'bcecoefficient',
                          'debug_grad']

        for arg in vars(args):
            if str(arg) in important_args:
                if str(arg) == 'debug_grad' and not getattr(args, arg):
                    continue
                elif str(arg) == 'overfit_num' and getattr(args, arg) == 0:
                    continue
                name += '-' + str(arg) + '_' + str(getattr(args, arg))

        return name

    def _tb_project_embeddings(self, args, net, loader, k):
        imgs, lbls = loader.dataset.get_k_samples(k)

        lbls = list(map(lambda x: x.argmax(), lbls))

        imgs = torch.stack(imgs)
        # lbls = torch.stack(lbls)

        print('imgs.shape', imgs.shape)
        if args.cuda:
            imgs_c = Variable(imgs.cuda())
        else:
            imgs_c = Variable(imgs)

        features, logits = net.forward(imgs_c, is_feat=True)
        feats = features[-1]

        print('feats.shape', feats.shape)

        self.writer.add_embedding(mat=feats.view(k, -1), metadata=lbls, label_img=imgs)
        self.writer.flush()

    def _tb_draw_histograms(self, args, net, epoch):

        for name, param in net.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(name, param.flatten(), epoch)

        self.writer.flush()

    def train_classify(self, net, loss_fn, args, trainLoader, valLoader):
        net.train()

        opt = torch.optim.Adam(net.parameters(), lr=args.lr_siamese)
        opt.zero_grad()

        train_losses = []
        time_start = time.time()
        queue = deque(maxlen=20)

        # print('steps:', args.max_steps)

        # epochs = int(np.ceil(args.max_steps / len(trainLoader)))
        epochs = 1

        total_batch_id = 0
        metric = metrics.Metric_Accuracy()

        for epoch in range(epochs):

            train_loss = 0
            metric.reset_acc()

            with tqdm(total=len(trainLoader), desc=f'Epoch {epoch + 1}/{epochs}') as t:
                for batch_id, (img, label) in enumerate(trainLoader, 1):

                    # print('input: ', img1.size())

                    if args.cuda:
                        img, label = Variable(img.cuda()), Variable(label.cuda())
                    else:
                        img, label = Variable(img), Variable(label)

                    net.train()
                    opt.zero_grad()

                    output = net.forward(img)
                    metric.update_acc(output, label)
                    loss = loss_fn(output, label)
                    # print('loss: ', loss.item())
                    train_loss += loss.item()
                    loss.backward()

                    opt.step()
                    total_batch_id += 1
                    t.set_postfix(loss=f'{train_loss / batch_id:.4f}', train_acc=f'{metric.get_acc():.4f}')

                    train_losses.append(train_loss)

                    t.update()

        return net

    def draw_heatmaps(self, net, loss_fn, bce_loss, args, cam_loader, transform_for_model=None,
                      transform_for_heatmap=None, epoch=0, count=1):

        net.eval()
        heatmap_path = f'{self.save_path}/heatmap/'
        heatmap_path_perepoch = os.path.join(heatmap_path, f'epoch_{epoch}/')

        os.mkdir(heatmap_path_perepoch)
        self.cam_all += 1
        for id, (anch_path, pos_path, neg_path) in enumerate(cam_loader, 1):

            anch_hm_path = os.path.join(heatmap_path, f'image_anch_{id}/')
            pos_hm_path = os.path.join(heatmap_path, f'image_pos_{id}/')
            neg_hm_path = os.path.join(heatmap_path, f'image_neg_{id}/')

            anchpos_hm_path = os.path.join(heatmap_path, f'image_anch_pos_{id}/')
            anchneg_hm_path = os.path.join(heatmap_path, f'image_anch_neg_{id}/')

            if not self.created_image_heatmap_path:
                os.mkdir(anch_hm_path)
                os.mkdir(pos_hm_path)
                os.mkdir(neg_hm_path)
                os.mkdir(anchpos_hm_path)
                os.mkdir(anchneg_hm_path)


            self.logger.info(f'Anch path: {anch_path}')
            self.logger.info(f'Pos path: {pos_path}')
            self.logger.info(f'Neg path: {neg_path}')

            anch = transform_for_model(Image.open(anch_path))
            pos = transform_for_model(Image.open(pos_path))
            neg = transform_for_model(Image.open(neg_path))

            anch = anch.reshape(shape=(1, anch.shape[0], anch.shape[1], anch.shape[2]))
            pos = pos.reshape(shape=(1, pos.shape[0], pos.shape[1], pos.shape[2]))
            neg = neg.reshape(shape=(1, neg.shape[0], neg.shape[1], neg.shape[2]))

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

            #
            # anch_org = cv2.imread(paths[0][0])
            # pos_org = cv2.imread(paths[1][0])
            # neg_org = cv2.imread(paths[2][0])

            # import pdb
            # pdb.set_trace()

            class_loss = 0
            ext_loss = 0
            pos_pred, pos_dist, anch_feat, pos_feat, acts = net.forward(anch, pos, feats=True, hook=True)

            # print(f'cam pos {id - 1}: ', torch.sigmoid(pos_pred).item())
            self.cam_pos[id - 1] += int(torch.sigmoid(pos_pred).item() < 0.5)

            ks = list(map(lambda x: int(x), args.k_best_maps))

            for k in ks:
                acts_tmp = []
                anch_hm_file_path = os.path.join(anch_hm_path, f'k_{k}_epoch_{epoch}.png')
                pos_hm_file_path = os.path.join(pos_hm_path, f'k_{k}_epoch_{epoch}.png')

                anchpos_anch_hm_file_path = os.path.join(anchpos_hm_path, f'k_{k}_epoch_{epoch}_anch.png')
                anchpos_pos_hm_file_path = os.path.join(anchpos_hm_path, f'k_{k}_epoch_{epoch}_pos.png')
                pos_max_indices = torch.topk(pos_dist, k=k).indices

                acts_tmp.append(acts[0][:, pos_max_indices, :, :].squeeze(dim=0))
                acts_tmp.append(acts[1][:, pos_max_indices, :, :].squeeze(dim=0))

                self.apply_forward_heatmap(acts_tmp,
                                           [('anch', anch_org), ('pos', pos_org)],
                                           id,
                                           heatmap_path_perepoch,
                                           individual_paths=[anch_hm_file_path,
                                                             pos_hm_file_path],
                                           pair_paths=[anchpos_anch_hm_file_path, anchpos_pos_hm_file_path])

            pos_class_loss = bce_loss(pos_pred.squeeze(), zero_labels.squeeze())
            pos_class_loss.backward(retain_graph=True)
            class_loss = pos_class_loss

            self.apply_grad_heatmaps(net.get_activations_gradient(),
                                     net.get_activations().detach(),
                                     {'anch': anch_org,
                                      'pos': pos_org}, 'bce_anch_pos', id, heatmap_path_perepoch)

            neg_pred, neg_dist, _, neg_feat, acts = net.forward(anch, neg, feats=True, hook=True)
            # print(f'cam neg {id - 1}: ', torch.sigmoid(neg_pred).item())
            self.cam_neg[id - 1] += int(torch.sigmoid(neg_pred).item() >= 0.5)

            # print('neg_pred', torch.sigmoid(neg_pred))

            for k in ks:
                acts_tmp = []

                anch_hm_file_path = os.path.join(anch_hm_path, f'k_{k}_epoch_{epoch}.png')
                neg_hm_file_path = os.path.join(neg_hm_path, f'k_{k}_epoch_{epoch}.png')

                anchneg_anch_hm_file_path = os.path.join(anchneg_hm_path, f'k_{k}_epoch_{epoch}_anch.png')
                anchneg_neg_hm_file_path = os.path.join(anchneg_hm_path, f'k_{k}_epoch_{epoch}_neg.png')
                neg_max_indices = torch.topk(neg_dist, k=k).indices

                acts_tmp.append(acts[0][:, neg_max_indices, :, :].squeeze(dim=0))
                acts_tmp.append(acts[1][:, neg_max_indices, :, :].squeeze(dim=0))

                self.apply_forward_heatmap(acts_tmp,
                                           [('anch', anch_org), ('neg', neg_org)],
                                           id,
                                           heatmap_path_perepoch,
                                           individual_paths=[anch_hm_file_path,
                                                             neg_hm_file_path],
                                           pair_paths=[anchneg_anch_hm_file_path, anchneg_neg_hm_file_path])


            neg_class_loss = bce_loss(neg_pred.squeeze(), one_labels.squeeze())
            neg_class_loss.backward(retain_graph=True)
            class_loss += neg_class_loss

            self.apply_grad_heatmaps(net.get_activations_gradient(),
                                     net.get_activations().detach(),
                                     {'anch': anch_org,
                                      'neg': neg_org}, 'bce_anch_neg', id, heatmap_path_perepoch)

            if loss_fn is not None:
                ext_batch_loss, parts = self.get_loss_value(args, loss_fn, pos_dist, neg_dist)
                ext_loss = ext_batch_loss

                ext_loss.backward(retain_graph=True)
                ext_loss /= self.no_negative
                self.apply_grad_heatmaps(net.get_activations_gradient(),
                                         net.get_activations().detach(),
                                         {'anch': anch_org,
                                          'pos': pos_org,
                                          'neg': neg_org}, 'triplet', id, heatmap_path_perepoch)

                class_loss /= (self.no_negative + 1)

                loss = ext_loss + self.bce_weight * class_loss

            else:

                loss = self.bce_weight * class_loss

            loss.backward()
            self.apply_grad_heatmaps(net.get_activations_gradient(),
                                     net.get_activations().detach(),
                                     {'anch': anch_org,
                                      'pos': pos_org,
                                      'neg': neg_org}, 'all', id, heatmap_path_perepoch)

        self.created_image_heatmap_path = True

        self.logger.info(f'CAM: anch-pos acc: {self.cam_pos / self.cam_all}')
        self.logger.info(f'CAM: anch-neg acc: {self.cam_neg / self.cam_all}')

    def to_numpy_axis_order_change(self, t):
        t = t.numpy()
        t = np.moveaxis(t.squeeze(), 0, -1)
        return t

    def apply_grad_heatmaps(self, grads, activations, img_dict, label, id, path):

        pooled_gradients = torch.mean(grads, dim=[0, 2, 3])

        for i in range(len(pooled_gradients)):
            activations[:, i, :, :] *= pooled_gradients[i]

        anch_org = img_dict['anch']
        heatmap = utils.get_heatmap(activations, shape=(anch_org.shape[0], anch_org.shape[1]))

        for l, i in img_dict.items():
            path_ = os.path.join(path, f'cam_{id}_{label}_{l}.png')
            utils.merge_heatmap_img(i, heatmap, path=path_)

    def apply_forward_heatmap(self, acts, img_list, id, path, individual_paths=None, pair_paths=None):

        heatmaps = []
        for idx, (l, i) in enumerate(img_list):
            heatmap = utils.get_heatmap(acts[idx], shape=(i.shape[0], i.shape[1]), label=l)
            heatmaps.append(heatmap)
            # path_ = os.path.join(path, f'cam_{id}_{l}.png')
            # utils.merge_heatmap_img(i, heatmap, path=path_)

            if individual_paths is not None:
                utils.merge_heatmap_img(i, heatmap, path=individual_paths[idx])

        # dist_heatmap = torch.pow((heatmaps[0] - heatmaps[1]), 2)
        # import pdb
        # pdb.set_trace()

        dist_heatmap = utils.get_heatmap(utils.vector_merge_function(acts[0], acts[1]),
                                         shape=(i.shape[0], i.shape[1]),
                                         label='subtractionn')
        # dist_heatmap = utils.vector_merge_function(heatmaps[0], heatmaps[1])
        # dist_heatmap = np.power(heatmaps[0] - heatmaps[1], 2)
        for idx, (l, i) in enumerate(img_list):
            utils.merge_heatmap_img(i, dist_heatmap, path=pair_paths[idx])

    def train_metriclearning(self, net, loss_fn, bce_loss, args, train_loader, val_loaders, val_loaders_fewshot,
                             train_loader_fewshot, cam_args=None, db_loaders=None):
        net.train()
        val_tol = args.early_stopping
        train_db_loader = db_loaders[0]
        val_db_loader = db_loaders[1]

        if net.mask:
            opt = torch.optim.Adam([{'params': net.sm_net.parameters()},
                                    {'params': net.ft_net.parameters(), 'lr': args.lr_resnet},
                                    {'params': net.input_layer.parameters(), 'lr': args.lr_siamese}],
                                   lr=args.lr_siamese)
        else:
            opt = torch.optim.Adam([{'params': net.sm_net.parameters()},
                                    {'params': net.ft_net.parameters(), 'lr': args.lr_resnet}], lr=args.lr_siamese)
        # net.ft_net.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        opt.zero_grad()

        train_losses = []
        time_start = time.time()
        queue = deque(maxlen=20)

        epochs = args.epochs

        metric_ACC = metrics.Metric_Accuracy()

        max_val_acc = 0
        max_val_acc_knwn = 0
        max_val_acc_unknwn = 0
        val_acc = 0
        val_rgt = 0
        val_err = 0
        best_model = ''

        drew_graph = False

        val_counter = 0

        for epoch in range(epochs):

            train_loss = 0
            train_bce_loss = 0
            train_triplet_loss = 0
            pos_parts = []
            neg_parts = []

            metric_ACC.reset_acc()

            with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{args.epochs}') as t:
                if self.draw_grad:
                    grad_save_path = os.path.join(self.plt_save_path, f'grads/epoch_{epoch}/')
                    # print(grad_save_path)
                    os.makedirs(grad_save_path)

                for batch_id, (anch, pos, neg) in enumerate(train_loader, 1):

                    # print('input: ', img1.size())

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
                        self.writer.add_graph(net, (anch, pos), verbose=True)
                        self.writer.flush()
                        drew_graph = True

                    net.train()
                    opt.zero_grad()

                    pos_pred, pos_dist, anch_feat, pos_feat = net.forward(anch, pos, feats=True)
                    if args.verbose:
                        print(f'norm pos: {pos_dist}')
                    class_loss = bce_loss(pos_pred.squeeze(), zero_labels.squeeze())
                    metric_ACC.update_acc(pos_pred.squeeze(), zero_labels.squeeze())  # zero dist means similar

                    for neg_iter in range(self.no_negative):
                        neg_pred, neg_dist, _, neg_feat = net.forward(anch, neg[:, neg_iter, :, :, :].squeeze(dim=1),
                                                                      feats=True)
                        # neg_dist.register_hook(lambda x: print(f'neg_dist grad:{x}'))
                        # neg_pred.register_hook(lambda x: print(f'neg_pred grad:{x}'))

                        if args.verbose:
                            print(f'norm neg {neg_iter}: {neg_dist}')

                        metric_ACC.update_acc(neg_pred.squeeze(), one_labels.squeeze())  # 1 dist means different

                        class_loss += bce_loss(neg_pred.squeeze(), one_labels.squeeze())
                        if loss_fn is not None:
                            ext_batch_loss, parts = self.get_loss_value(args, loss_fn, pos_dist, neg_dist)

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
                        loss = ext_loss + self.bce_weight * class_loss
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

                            print('got triplet loss grads')

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

                        print('got bce grads')

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

                    train_losses.append(train_loss)

                    t.update()

                #
                # svm = SVC()
                # knn = KNeighborsClassifier(n_neighbors=1)
                #
                # metric_SVC = self.linear_classifier(train_embeddings, svm, metric_SVC)
                # metric_KNN = self.linear_classifier(train_embeddings, knn, metric_KNN)

                if args.loss == 'maxmargin':
                    plt.hist([np.array(pos_parts).flatten(), np.array(neg_parts).flatten()], bins=30, alpha=0.3,
                             label=['pos', 'neg'])
                    plt.title(f'Losses Epoch {epoch}')
                    plt.legend(loc='upper right')
                    plt.savefig(f'{self.plt_save_path}/pos_part_{epoch}.png')

                if bce_loss is None:
                    bce_loss = loss_fn

                train_fewshot_acc, train_fewshot_loss, train_fewshot_right, train_fewshot_error = self.apply_fewshot_eval(
                    args, net, train_loader_fewshot, bce_loss)

                self.logger.info(f'Train_Fewshot_Acc: {train_fewshot_acc}, Train_Fewshot_loss: {train_fewshot_loss},\n '
                                 f'Train_Fewshot_Right: {train_fewshot_right}, Train_Fewshot_Error: {train_fewshot_error}')

                self.writer.add_scalar('Train/Loss', train_loss / len(train_loader), epoch)
                if loss_fn is not None:
                    self.writer.add_scalar('Train/Triplet_Loss', train_triplet_loss / len(train_loader), epoch)
                self.writer.add_scalar('Train/BCE_Loss', train_bce_loss / len(train_loader), epoch)
                self.writer.add_scalar('Train/Fewshot_Loss', train_fewshot_loss / len(train_loader_fewshot), epoch)

                self.writer.add_scalar('Train/Acc', metric_ACC.get_acc(), epoch)
                self.writer.add_scalar('Train/Fewshot_Acc', train_fewshot_acc, epoch)
                self.writer.flush()

                if val_loaders is not None and (epoch + 1) % args.test_freq == 0:
                    net.eval()

                    val_acc_unknwn, val_acc_knwn = -1, -1

                    if args.eval_mode == 'fewshot':

                        val_rgt_knwn, val_err_knwn, val_acc_knwn = self.test_fewshot(args, net,
                                                                                     val_loaders_fewshot[0],
                                                                                     bce_loss, val=True,
                                                                                     epoch=epoch, comment='known')
                        self.test_metric(args, net, val_loaders[0],
                                         loss_fn, bce_loss, val=True,
                                         epoch=epoch, comment='known')

                        val_rgt_unknwn, val_err_unknwn, val_acc_unknwn = self.test_fewshot(args, net,
                                                                                           val_loaders_fewshot[1],
                                                                                           bce_loss,
                                                                                           val=True,
                                                                                           epoch=epoch,
                                                                                           comment='unknown')
                        self.test_metric(args, net, val_loaders[1],
                                         loss_fn, bce_loss, val=True,
                                         epoch=epoch, comment='unknown')

                    elif args.eval_mode == 'simple':  # todo not compatible with new data-splits
                        val_rgt, val_err, val_acc = self.test_simple(args, net, val_loaders, loss_fn, val=True,
                                                                     epoch=epoch)
                    else:
                        raise Exception('Unsupporeted eval mode')

                    self.logger.info('known val acc: [%f], unknown val acc [%f]' % (val_acc_knwn, val_acc_unknwn))
                    self.logger.info('*' * 30)
                    if val_acc_knwn > max_val_acc_knwn:
                        self.logger.info(
                            'known val acc: [%f], beats previous max [%f]' % (val_acc_knwn, max_val_acc_knwn))
                        self.logger.info('known rights: [%d], known errs [%d]' % (val_rgt_knwn, val_err_knwn))
                        max_val_acc_knwn = val_acc_knwn

                    if val_acc_unknwn > max_val_acc_unknwn:
                        self.logger.info(
                            'unknown val acc: [%f], beats previous max [%f]' % (val_acc_unknwn, max_val_acc_unknwn))
                        self.logger.info(
                            'unknown rights: [%d], unknown errs [%d]' % (val_rgt_unknwn, val_err_unknwn))
                        max_val_acc_unknwn = val_acc_unknwn

                    val_acc = ((val_rgt_knwn + val_rgt_unknwn) * 1.0) / (
                            val_rgt_knwn + val_rgt_unknwn + val_err_knwn + val_err_unknwn)

                    self.writer.add_scalar('Total_Val/Acc', val_acc, epoch)
                    self.writer.flush()

                    val_rgt = (val_rgt_knwn + val_rgt_unknwn)
                    val_err = (val_err_knwn + val_err_unknwn)

                if val_acc > max_val_acc:
                    val_counter = 0
                    self.logger.info(
                        'saving model... current val acc: [%f], previous val acc [%f]' % (val_acc, max_val_acc))
                    best_model = self.save_model(args, net, epoch, val_acc)
                    max_val_acc = val_acc

                    queue.append(val_rgt * 1.0 / (val_rgt + val_err))

            if epoch % 5 == 0:
                self.make_emb_db(args, net, train_db_loader,
                                 eval_sampled=args.sampled_results,
                                 eval_per_class=args.per_class_results,
                                 newly_trained=True,
                                 batch_size=args.db_batch,
                                 mode='train',
                                 epoch=epoch,
                                 k_at_n=False)

                self.make_emb_db(args, net, val_db_loader,
                                 eval_sampled=args.sampled_results,
                                 eval_per_class=args.per_class_results,
                                 newly_trained=True,
                                 batch_size=args.db_batch,
                                 mode='val',
                                 epoch=epoch,
                                 k_at_n=False)

            if args.cam:
                self.logger.info(f'Drawing heatmaps on epoch {epoch}...')
                self.draw_heatmaps(net=net,
                                   loss_fn=loss_fn,
                                   bce_loss=bce_loss,
                                   args=args,
                                   cam_loader=cam_args[0],
                                   transform_for_model=cam_args[1],
                                   transform_for_heatmap=cam_args[2],
                                   epoch=epoch,
                                   count=1)

                self.logger.info(f'DONE drawing heatmaps on epoch {epoch}!!!')

            self._tb_draw_histograms(args, net, epoch)

        with open('train_losses', 'wb') as f:
            pickle.dump(train_losses, f)

        acc = 0.0
        for d in queue:
            acc += d
        print("#" * 70)
        print('queue len: ', len(queue))

        if args.project_tb:
            print("Start projecting")
            # self._tb_project_embeddings(args, net.ft_net, train_loader, 1000)
            print("Projecting done")

        return net, best_model

    def train_fewshot(self, net, loss_fn, args, train_loader, val_loaders):
        net.train()
        val_tol = args.early_stopping
        opt = torch.optim.Adam([{'params': net.sm_net.parameters()},
                                {'params': net.ft_net.parameters(), 'lr': args.lr_resnet}], lr=args.lr_siamese)

        opt.zero_grad()

        train_losses = []
        time_start = time.time()
        queue = deque(maxlen=20)

        # print('steps:', args.max_steps)

        # epochs = int(np.ceil(args.max_steps / len(trainLoader)))
        epochs = args.epochs

        metric = metrics.Metric_Accuracy()

        max_val_acc = 0
        max_val_acc_knwn = 0
        max_val_acc_unknwn = 0
        best_model = ''

        drew_graph = False

        val_counter = 0

        for epoch in range(epochs):

            train_loss = 0
            metric.reset_acc()

            with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{args.epochs}') as t:
                for batch_id, (img1, img2, label) in enumerate(train_loader, 1):

                    # print('input: ', img1.size())

                    if args.cuda:
                        img1, img2, label = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda())
                    else:
                        img1, img2, label = Variable(img1), Variable(img2), Variable(label)

                    if not drew_graph:
                        self.writer.add_graph(net, (img1, img2), verbose=True)
                        self.writer.flush()
                        drew_graph = True

                    net.train()
                    opt.zero_grad()

                    output = net.forward(img1, img2)
                    metric.update_acc(output.squeeze(), label.squeeze())
                    loss = loss_fn(output, label)
                    # print('loss: ', loss.item())
                    train_loss += loss.item()
                    loss.backward()
                    # plt = self.plot_grad_flow(net.named_parameters())
                    # import pdb
                    # pdb.set_trace()
                    opt.step()
                    t.set_postfix(loss=f'{train_loss / batch_id:.4f}', train_acc=f'{metric.get_acc():.4f}')

                    # if total_batch_id % args.log_freq == 0:
                    #     logger.info('epoch: %d, batch: [%d]\tacc:\t%.5f\tloss:\t%.5f\ttime lapsed:\t%.2f s' % (
                    #         epoch, batch_id, metric.get_acc(), train_loss / args.log_freq, time.time() - time_start))
                    #     train_loss = 0
                    #     metric.reset_acc()
                    #     time_start = time.time()

                    train_losses.append(train_loss)

                    t.update()

                self.writer.add_scalar('Train/Loss', train_loss / len(train_loader), epoch)
                self.writer.add_scalar('Train/Acc', metric.get_acc(), epoch)
                self.writer.flush()

                if val_loaders is not None and (epoch + 1) % args.test_freq == 0:
                    net.eval()

                    val_acc_unknwn, val_acc_knwn = -1, -1

                    if args.eval_mode == 'fewshot':

                        val_rgt_knwn, val_err_knwn, val_acc_knwn = self.test_fewshot(args, net, val_loaders[0],
                                                                                     loss_fn, val=True,
                                                                                     epoch=epoch, comment='known')
                        val_rgt_unknwn, val_err_unknwn, val_acc_unknwn = self.test_fewshot(args, net,
                                                                                           val_loaders[1], loss_fn,
                                                                                           val=True,
                                                                                           epoch=epoch,
                                                                                           comment='unknown')

                    elif args.eval_mode == 'simple':  # todo not compatible with new data-splits
                        val_rgt, val_err, val_acc = self.test_simple(args, net, val_loaders, loss_fn, val=True,
                                                                     epoch=epoch)
                    else:
                        raise Exception('Unsupporeted eval mode')

                    self.logger.info('known val acc: [%f], unknown val acc [%f]' % (val_acc_knwn, val_acc_unknwn))
                    self.logger.info('*' * 30)
                    if val_acc_knwn > max_val_acc_knwn:
                        self.logger.info(
                            'known val acc: [%f], beats previous max [%f]' % (val_acc_knwn, max_val_acc_knwn))
                        self.logger.info('known rights: [%d], known errs [%d]' % (val_rgt_knwn, val_err_knwn))
                        max_val_acc_knwn = val_acc_knwn

                    if val_acc_unknwn > max_val_acc_unknwn:
                        self.logger.info(
                            'unknown val acc: [%f], beats previous max [%f]' % (val_acc_unknwn, max_val_acc_unknwn))
                        self.logger.info(
                            'unknown rights: [%d], unknown errs [%d]' % (val_rgt_unknwn, val_err_unknwn))
                        max_val_acc_unknwn = val_acc_unknwn

                    val_acc = ((val_rgt_knwn + val_rgt_unknwn) * 1.0) / (
                            val_rgt_knwn + val_rgt_unknwn + val_err_knwn + val_err_unknwn)

                    self.writer.add_scalar('Total_Val/Acc', val_acc, epoch)
                    self.writer.flush()

                    val_rgt = (val_rgt_knwn + val_rgt_unknwn)
                    val_err = (val_err_knwn + val_err_unknwn)

                if val_acc > max_val_acc:
                    val_counter = 0
                    self.logger.info(
                        'saving model... current val acc: [%f], previous val acc [%f]' % (val_acc, max_val_acc))
                    best_model = self.save_model(args, net, epoch, val_acc)
                    max_val_acc = val_acc

                    queue.append(val_rgt * 1.0 / (val_rgt + val_err))

            self._tb_draw_histograms(args, net, epoch)

        with open('train_losses', 'wb') as f:
            pickle.dump(train_losses, f)

        acc = 0.0
        for d in queue:
            acc += d
        print("#" * 70)
        print('queue len: ', len(queue))

        if args.project_tb:
            print("Start projecting")
            # self._tb_project_embeddings(args, net.ft_net, train_loader, 1000)
            print("Projecting done")

        return net, best_model

    def test_simple(self, args, net, data_loader, loss_fn, val=False, epoch=0):
        net.eval()

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

        if val:
            prompt_text = comment + f' VAL METRIC LEARNING epoch {epoch}:\tcorrect:\t%d\terror:\t%d\tval_acc:%f\tval_loss:%f\t'
            prompt_text_tb = comment + '_Val'
        else:
            prompt_text = comment + ' TEST METRIC LEARNING:\tcorrect:\t%d\terror:\t%d\ttest_acc:%f\ttest_loss:%f\t'
            prompt_text_tb = comment + '_Test'

        tests_right, tests_error = 0, 0

        test_loss = 0
        test_bce_loss = 0
        test_triplet_loss = 0
        loss = 0

        for _, (anch, pos, neg) in enumerate(data_loader, 1):

            one_labels = torch.tensor([1 for _ in range(anch.shape[0])], dtype=float)
            zero_labels = torch.tensor([0 for _ in range(anch.shape[0])], dtype=float)

            if args.cuda:
                anch, pos, neg, one_labels, zero_labels = anch.cuda(), pos.cuda(), neg.cuda(), one_labels.cuda(), zero_labels.cuda()
            anch, pos, neg, one_labels, zero_labels = Variable(anch), Variable(pos), Variable(neg), Variable(
                one_labels), Variable(zero_labels)

            ###
            pos_pred, pos_dist, anch_feat, pos_feat = net.forward(anch, pos, feats=True)
            class_loss = bce_loss(pos_pred.squeeze(), zero_labels.squeeze())

            for neg_iter in range(self.no_negative):
                # print(anch.shape)
                # print(neg[:, neg_iter, :, :, :].squeeze(dim=1).shape)
                neg_pred, neg_dist, _, neg_feat = net.forward(anch, neg[:, neg_iter, :, :, :].squeeze(dim=1),
                                                              feats=True)

                class_loss += bce_loss(neg_pred.squeeze(), one_labels.squeeze())

                if loss_fn is not None:
                    ext_batch_loss, parts = self.get_loss_value(args, loss_fn, pos_dist, neg_dist)

                    if neg_iter == 0:
                        ext_loss = ext_batch_loss
                    else:
                        ext_loss += ext_batch_loss

            class_loss /= (self.no_negative + 1)
            if loss_fn is not None:
                ext_loss /= self.no_negative
                test_triplet_loss += ext_loss.item()
                loss = ext_loss + self.bce_weight * class_loss
            else:
                loss = self.bce_weight * class_loss

            test_loss += loss.item()

            test_bce_loss += class_loss.item()

        self.logger.info('$' * 70)

        # self.writer.add_scalar(f'{prompt_text_tb}/Triplet_Loss', test_loss / len(data_loader), epoch)
        self.logger.error(f'{prompt_text_tb}/Loss:  {test_loss / len(data_loader)}, epoch: {epoch}')
        self.writer.add_scalar(f'{prompt_text_tb}/Loss', test_loss / len(data_loader), epoch)
        if loss_fn is not None:
            self.logger.error(f'{prompt_text_tb}/Triplet_Loss: {test_triplet_loss / len(data_loader)}, epoch: {epoch}')
            self.writer.add_scalar(f'{prompt_text_tb}/Triplet_Loss', test_triplet_loss / len(data_loader), epoch)
        self.logger.error(f'{prompt_text_tb}/BCE_Loss: {test_bce_loss / len(data_loader)}, epoch: {epoch}')
        self.writer.add_scalar(f'{prompt_text_tb}/BCE_Loss', test_bce_loss / len(data_loader), epoch)

        # self.writer.add_scalar(f'{prompt_text_tb}/Acc', test_acc, epoch)
        self.writer.flush()

        return

    def test_fewshot(self, args, net, data_loader, loss_fn, val=False, epoch=0, comment=''):
        net.eval()

        if val:
            prompt_text = comment + f' VAL FEW SHOT epoch {epoch}:\tcorrect:\t%d\terror:\t%d\tval_acc:%f\tval_loss:%f\t'
            prompt_text_tb = comment + '_Val'
        else:
            prompt_text = comment + ' TEST FEW SHOT:\tcorrect:\t%d\terror:\t%d\ttest_acc:%f\ttest_loss:%f\t'
            prompt_text_tb = comment + '_Test'

        test_acc, test_loss, tests_right, tests_error = self.apply_fewshot_eval(args, net, data_loader, loss_fn)

        self.logger.info('$' * 70)
        self.logger.info(prompt_text % (tests_right, tests_error, test_acc, test_loss))
        self.logger.info('$' * 70)

        self.writer.add_scalar(f'{prompt_text_tb}/Fewshot_Loss', test_loss, epoch)
        self.writer.add_scalar(f'{prompt_text_tb}/Fewshot_Acc', test_acc, epoch)
        self.writer.flush()

        return tests_right, tests_error, test_acc

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

        if newly_trained:
            net.eval()
            if batch_size is None:
                batch_size = args.batch_size

            steps = int(np.ceil(len(data_loader) / batch_size))

            test_classes = np.zeros(((len(data_loader.dataset))))
            test_seen = np.zeros(((len(data_loader.dataset))))
            test_paths = np.empty(dtype='S20', shape=((len(data_loader.dataset))))
            if args.feat_extractor == 'resnet50':
                test_feats = np.zeros((len(data_loader.dataset), 2048))
            elif args.feat_extractor == 'resnet18':
                test_feats = np.zeros((len(data_loader.dataset), 512))
            else:
                raise Exception('Not handled feature extractor')

            for idx, tpl in enumerate(data_loader):

                end = min((idx + 1) * batch_size, len(test_feats))

                if mode != 'train':
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

                if mode != 'train':
                    test_seen[idx * batch_size:end] = seen.to(int)

            utils.save_h5(f'{mode}_ids', test_paths, 'S20', os.path.join(self.save_path, f'{mode}Ids.h5'))
            utils.save_h5(f'{mode}_classes', test_classes, 'i8', os.path.join(self.save_path, f'{mode}Classes.h5'))
            utils.save_h5(f'{mode}_feats', test_feats, 'f', os.path.join(self.save_path, f'{mode}Feats.h5'))
            if mode != 'train':
                utils.save_h5(f'{mode}_seen', test_seen, 'i2', os.path.join(self.save_path, f'{mode}Seen.h5'))

        test_feats = utils.load_h5(f'{mode}_feats', os.path.join(self.save_path, f'{mode}Feats.h5'))
        test_classes = utils.load_h5(f'{mode}_classes', os.path.join(self.save_path, f'{mode}Classes.h5'))
        if mode != 'train':
            test_seen = utils.load_h5(f'{mode}_seen', os.path.join(self.save_path, f'{mode}Seen.h5'))

        # pca_path = os.path.join(self.scatter_plot_path, f'pca_{epoch}.png')
        tsne_path = os.path.join(self.scatter_plot_path, f'{mode}/tsne_{epoch}.png')

        # self.draw_dim_reduced(test_feats, test_classes, method='pca', title="on epoch " + str(epoch), path=pca_path)
        self.draw_dim_reduced(test_feats, test_classes, method='tsne', title=f"{mode}, epoch: " + str(epoch),
                              path=tsne_path)

        # import pdb
        # pdb.set_trace()
        if k_at_n:
            utils.calculate_k_at_n(args, test_feats, test_classes, test_seen, logger=self.logger,
                                   limit=args.limit_samples,
                                   run_number=args.number_of_runs,
                                   save_path=self.save_path,
                                   sampled=eval_sampled,
                                   per_class=eval_per_class,
                                   mode=mode)

            self.logger.info('results at: ' + self.save_path)

    def load_model(self, args, net, best_model):
        checkpoint = torch.load(os.path.join(self.save_path, best_model))
        self.logger.info('Loading model %s from epoch [%d]' % (best_model, checkpoint['epoch']))
        net.load_state_dict(checkpoint['model_state_dict'])
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

    def save_model(self, args, net, epoch, val_acc):
        best_model = 'model-epoch-' + str(epoch + 1) + '-val-acc-' + str(val_acc) + '.pt'
        torch.save({'epoch': epoch, 'model_state_dict': net.state_dict()},
                   self.save_path + '/' + best_model)
        return best_model

    def getBack(self, var_grad_fn):
        print(var_grad_fn)
        for n in var_grad_fn.next_functions:
            if n[0]:
                try:
                    tensor = getattr(n[0], 'variable')
                    print(n[0])
                    print('Tensor with grad found:', tensor)
                    print(' - gradient:', tensor.grad)
                    print()
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
                    print(n, p)
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
        if batch_size is None:
            batch_size = args.batch_size

        if args.feat_extractor == 'resnet50':
            embs = np.zeros((len(data_loader.dataset), 2048))
        elif args.feat_extractor == 'resnet18':
            embs = np.zeros((len(data_loader.dataset), 512))

        labels = np.zeros((len(data_loader.dataset)))
        seen = np.zeros((len(data_loader.dataset)))

        for idx, (img, lbl, seen, _) in enumerate(data_loader):

            if args.cuda:
                img = img.cuda()
            img = Variable(img)

            output = net.forward(img, None, single=True)
            output = output.data.cpu().numpy()

            end = min((idx + 1) * batch_size, len(embs))

            embs[idx * batch_size:end, :] = output
            labels[idx * batch_size:end] = lbl
            seen[idx * batch_size:end] = seen.to(int)

        return embs, labels, seen

    def apply_fewshot_eval(self, args, net, data_loader, loss_fn):

        right, error = 0, 0

        label = np.ones(shape=args.way, dtype=np.float32)
        label[0] = 0
        label = torch.from_numpy(label)
        loss = 0
        if args.cuda:
            label = Variable(label.cuda())
        else:
            label = Variable(label)

        for _, (img1, img2) in enumerate(data_loader, 1):
            if args.cuda:
                img1, img2 = img1.cuda(), img2.cuda()
            img1, img2 = Variable(img1), Variable(img2)
            pred_vector, dist = net.forward(img1, img2)
            loss += loss_fn(pred_vector.reshape((-1,)), label.reshape((-1,))).item()
            pred_vector = pred_vector.reshape((-1,)).data.cpu().numpy()
            pred = np.argmin(pred_vector)
            if pred == 0:
                right += 1
            else:
                error += 1

        acc = right * 1.0 / (right + error)

        return acc, loss, right, error

    def get_loss_value(self, args, loss_fn, pos_dist, neg_dist):
        if args.loss == 'trpl':
            loss = loss_fn(pos_dist, neg_dist)
            parts = []
        elif args.loss == 'maxmargin':
            loss, parts = loss_fn(pos_dist, neg_dist)
        else:
            loss = None
            parts = None

        return loss, parts

    # todo make customized dataloader for cam
    # todo easy cases?
