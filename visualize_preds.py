import os

import utils
import numpy as np


model_paths = {
    # '1e-5_1e-3': './normalizeOrNotnormalize_preds/model-bs20-1gpus-sNOTnormalized-dsn_hotels-nor_200-fe_resnet50-pool_spoc-el_0-nn_1-bs_20-lrs_0.001-lrr_1e-05-m_1.0-loss_trpl-mm_diff-sim-bco_1.0-igsz_224-time_2021-01-24_14-26-14-305022/',
    # '1e-5_3e-4': './normalizeOrNotnormalize_preds/model-bs20-1gpus-sNOTnormalized-dsn_hotels-nor_200-fe_resnet50-pool_spoc-el_0-nn_1-bs_20-lrs_0.0003-lrr_1e-05-m_1.0-loss_trpl-mm_diff-sim-bco_1.0-igsz_224-time_2021-01-25_02-23-58-365886'
    # 'NOTnormalized': './normalizeOrNotnormalize_preds/model-bs20-1gpus-concat-NOTnormalize-dsn_hotels-nor_200-fe_resnet50-pool_spoc-el_2-nn_1-bs_20-lrs_0.3-lrr_3e-06-m_1.0-loss_trpl-mm_concat-bco_1.0-igsz_300-time_2021-01-31_22-21-52-281218_44017506',
    # 'el4': '/Users/aarash/Files/courses/mcgill_courses/mila/research/projects/ht-image/el_alot/el4/',
    # 'el5': '/Users/aarash/Files/courses/mcgill_courses/mila/research/projects/ht-image/el_alot/el5/'
    # 'el3_1': '/Users/aarash/files/courses/mcgill_courses/mila/research/projects/ht-image/30epochs_extra_layers_npzs/3el1',
    # 'el3_2': '/Users/aarash/files/courses/mcgill_courses/mila/research/projects/ht-image/30epochs_extra_layers_npzs/3el2',
    # 'el4_1': '/Users/aarash/files/courses/mcgill_courses/mila/research/projects/ht-image/30epochs_extra_layers_npzs/4el1',
    # 'el4_2': '/Users/aarash/files/courses/mcgill_courses/mila/research/projects/ht-image/30epochs_extra_layers_npzs/4el2',
    # 'el2_1': '/Users/aarash/files/courses/mcgill_courses/mila/research/projects/ht-image/30epochs_extra_layers_npzs/2el1',
    # 'el2_2': '/Users/aarash/files/courses/mcgill_courses/mila/research/projects/ht-image/30epochs_extra_layers_npzs/2el2',
    # 'bn_el2': '/Users/aarash/files/courses/mcgill_courses/mila/research/projects/ht-image/30epochs_extra_layers_npzs/bn_2el',
    # 'bn_el3': '/Users/aarash/files/courses/mcgill_courses/mila/research/projects/ht-image/30epochs_extra_layers_npzs/bn_3el',
    # 'bn_el4': '/Users/aarash/files/courses/mcgill_courses/mila/research/projects/ht-image/30epochs_extra_layers_npzs/bn_4el',
    # 'gem_el2': '/Users/aarash/files/courses/mcgill_courses/mila/research/projects/ht-image/gem_el/el2',
    # 'gem_el3': '/Users/aarash/files/courses/mcgill_courses/mila/research/projects/ht-image/gem_el/el3',
    # 'gem_el4': '/Users/aarash/files/courses/mcgill_courses/mila/research/projects/ht-image/gem_el/el4',
    # '16472008': '/Users/aarash/files/courses/mcgill_courses/mila/research/projects/ht-image/16472008'
    # 'best?diff-sim': '/Users/aarash/files/courses/mcgill_courses/mila/research/projects/ht-image/new_eval_npz/model-bs20-1gpus-10epochs_smallerhp_onlybce-dsn_hotels-nor_200-fe_resnet50-pool_spoc-el_0-nn_1-bs_20-lrs_0.01-lrr_3e-06-loss_bce-mm_diff-sim-decay_0.0-igsz_224-time_2021-03-03_20-52-10-867248_16699114',
    'best_diff-sim': '/Users/aarash/files/courses/mcgill_courses/mila/research/projects/ht-image/new_eval_npz/model-bs20-1gpus-40epochs_bestbces-dsn_hotels-nor_200-fe_resnet50-pool_spoc-el_0-nn_1-bs_20-lrs_0.03-lrr_3e-06-loss_bce-mm_diff-sim-decay_0.0-igsz_224-time_2021-03-07_02-14-22-060388_16807500',
    'secondbest_diff-sim': '/Users/aarash/files/courses/mcgill_courses/mila/research/projects/ht-image/new_eval_npz/model-bs20-1gpus-40epochs_bestbces-dsn_hotels-nor_200-fe_resnet50-pool_spoc-el_0-nn_1-bs_20-lrs_0.01-lrr_3e-06-loss_bce-mm_diff-sim-decay_0.0-igsz_224-time_2021-03-07_02-12-50-126889_16807499',
    'thirdbest_diff-sim': '/Users/aarash/files/courses/mcgill_courses/mila/research/projects/ht-image/new_eval_npz/model-bs20-1gpus-40epochs_bestbces-dsn_hotels-nor_200-fe_resnet50-pool_spoc-el_0-nn_1-bs_20-lrs_0.1-lrr_3e-06-loss_bce-mm_diff-sim-decay_0.0-igsz_224-time_2021-03-07_02-14-22-060221_16807501',



}

for k, v in model_paths.items():
    for ep in range(31, 0, -1):
        train_path = os.path.join(v, f'train_preds_epoch{ep}.npz')
        val_known_path = os.path.join(v, f'val_preds_knwn_epoch{ep}.npz')
        val_unknown_path = os.path.join(v, f'val_preds_unknwn_epoch{ep}.npz')
        if os.path.exists(val_known_path):

            print('FUCK', val_known_path)

            vk_pos_preds, vk_neg_preds = utils.get_pos_neg_preds(val_known_path)
            vu_pos_preds, vu_neg_preds = utils.get_pos_neg_preds(val_unknown_path)



            if os.path.exists(train_path):
                t_pos_preds, t_neg_preds = utils.get_pos_neg_preds(train_path)
                utils.plot_pred_hist(t_pos_preds, t_neg_preds, title=f'Train Ep {ep} model {k}',
                                     savepath=os.path.join(v, f't_ep{ep}_{k}'))

            utils.plot_pred_hist(vk_pos_preds, vk_neg_preds, title=f'Val seen Ep {ep} model {k}',
                                 savepath=os.path.join(v, f'vk_ep{ep}_{k}'))
            utils.plot_pred_hist(vu_pos_preds, vu_neg_preds, title=f'Val unseen Ep {ep} model {k}',
                                 savepath=os.path.join(v, f'vu_ep{ep}_{k}'))

            # break
    for ep in range(31, 0, -1):
        train_path_neg = os.path.join(v, f'train_preds_neg_epoch{ep}.npz')
        val_known_path_neg = os.path.join(v, f'val_preds_knwn_neg_epoch{ep}.npz')
        val_unknown_path_neg = os.path.join(v, f'val_preds_unknwn_neg_epoch{ep}.npz')

        train_path_pos = os.path.join(v, f'train_preds_pos_epoch{ep}.npz')
        val_known_path_pos = os.path.join(v, f'val_preds_knwn_pos_epoch{ep}.npz')
        val_unknown_path_pos = os.path.join(v, f'val_preds_unknwn_pos_epoch{ep}.npz')

        if os.path.exists(val_known_path_neg):
            print(val_known_path_neg)

            vk_pos_preds = np.load(val_known_path_pos, allow_pickle=True)['arr_0']
            vk_neg_preds = np.load(val_known_path_neg, allow_pickle=True)['arr_0']

            vu_pos_preds = np.load(val_unknown_path_pos, allow_pickle=True)['arr_0']
            vu_neg_preds = np.load(val_unknown_path_neg, allow_pickle=True)['arr_0']



            # if os.path.exists(train_path):
            #     t_pos_preds, t_neg_preds = utils.get_pos_neg_preds(train_path)
            #     utils.plot_pred_hist(t_pos_preds, t_neg_preds, title=f'Train Ep {ep} model {k}',
            #                          savepath=os.path.join(v, f't_ep{ep}_{k}'))

            utils.plot_pred_hist(vk_pos_preds, vk_neg_preds, title=f'Val seen Ep {ep} model {k}',
                                 savepath=os.path.join(v, f'vk_ep{ep}_{k}'), normalizefactor=1)
            utils.plot_pred_hist(vu_pos_preds, vu_neg_preds, title=f'Val unseen Ep {ep} model {k}',
                                 savepath=os.path.join(v, f'vu_ep{ep}_{k}'), normalizefactor=1)

            # break

# class BaslineModel:
#     def __init__(self, args, model, logger, loss_fn):
#         self.logger = logger
#         self.model = model
#         self.loss_fn = loss_fn # batch hard
#         self.bh_k = args.bh_K
#         self.bh_p = args.bh_P
#
#         self.save_path = '' #todo
#         self.gen_plot_path = '' #todo
#
#         self.class_diffs = {'train':
#                                 {'between_class_average': [],
#                                  'between_class_min': [],
#                                  'between_class_max': [],
#                                  'in_class_average': [],
#                                  'in_class_min': [],
#                                  'in_class_max': []},
#                             'val':
#                                 {'between_class_average': [],
#                                  'between_class_min': [],
#                                  'between_class_max': [],
#                                  'in_class_average': [],
#                                  'in_class_min': [],
#                                  'in_class_max': []}}
#         self.silhouette_scores = {'train': [],
#                                   'val': []}
#
#         self.aug_mask = args.aug_mask
#
#     def train_epoch(self, t, args, train_loader, opt, epoch):
#         train_loss = 0
#
#
#         labels = torch.Tensor([[i for _ in range(self.bh_k)] for i in range(self.bh_p)]).flatten()
#         if args.cuda:
#             labels = Variable(labels.cuda())
#         else:
#             labels = Variable(labels)
#
#         for batch_id, imgs in enumerate(train_loader, 1):
#             start = time.time()
#
#             if args.cuda:
#                 imgs = Variable(imgs.cuda())
#             else:
#                 imgs = Variable(imgs)
#
#             # if not drew_graph:
#             #     self.writer.add_graph(self.model, (imgs.detach()), verbose=True)
#             #     self.writer.flush()
#             #     drew_graph = True
#
#             self.model.train()
#             # device = f'cuda:{net.device_ids[0]}'
#             opt.zero_grad()
#             forward_start = time.time()
#             feats = self.model.forward(imgs)
#             forward_end = time.time()
#
#
#             if utils.MY_DEC.enabled:
#                 self.logger.info(f'########### forward time: {forward_end - forward_start}')
#
#             # if args.verbose:
#             #     self.logger.info(f'norm pos: {pos_dist}')
#
#             loss = self.loss_fn(feats, labels)
#
#             train_loss += loss.item()
#
#             loss.backward()  # training with triplet loss
#
#             opt.step()
#
#             t.set_postfix(loss=f'{train_loss / (batch_id) :.4f}')
#             t.update()
#
#             end = time.time()
#             if utils.MY_DEC.enabled:
#                 self.logger.info(f'########### one batch time: {end - start}')
#
#         return t, train_loss
#
#     def train(self, args, epochs, train_loader):
#
#
#         if args.aug_mask:
#             opt = torch.optim.Adam([{'params': self.model.parameters()}],
#                                    lr=args.lr_resnet)
#         else:
#             opt = torch.optim.Adam([{'params': self.model.sm_net.parameters()}],
#                                    lr=args.lr_resnet)
#         # net.ft_net.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         opt.zero_grad()
#
#         for epoch in range(1, epochs + 1):
#             with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}') as t:
#
#                 all_batches_start = time.time()
#
#                 utils.print_gpu_stuff(args.cuda, 'before train epoch')
#
#                 t, train_loss = self.train_epoch(args, t, train_loader, opt, epoch)
#
#             if (epoch) % args.test_freq == 0:
#                 self.model.eval()
#
#                 self.make_emb_db(args, val_loader, )
#
#     def make_emb_db(self, args, data_loader, eval_sampled, eval_per_class, batch_size=None,
#                     mode='val', epoch=-1, k_at_n=True):
#
#         self.model.eval()
#         # device = f'cuda:{net.device_ids[0]}'
#         if batch_size is None:
#             batch_size = args.batch_size
#
#         steps = int(np.ceil(len(data_loader) / batch_size))
#
#         test_classes = np.zeros(((len(data_loader.dataset))))
#         test_seen = np.zeros(((len(data_loader.dataset))))
#         test_paths = np.empty(dtype='S20', shape=((len(data_loader.dataset))))
#         if args.feat_extractor == 'resnet50':
#             test_feats = np.zeros((len(data_loader.dataset), 2048))
#         elif args.feat_extractor == 'resnet18':
#             test_feats = np.zeros((len(data_loader.dataset), 512))
#         elif args.feat_extractor == 'vgg16':
#             test_feats = np.zeros((len(data_loader.dataset), 4096))
#         else:
#             raise Exception('Not handled feature extractor')
#
#         for idx, tpl in enumerate(data_loader):
#
#             end = min((idx + 1) * batch_size, len(test_feats))
#
#             if mode != 'train':
#                 (img, lbl, seen, path) = tpl
#             else:
#                 (img, lbl, path) = tpl
#
#             if args.cuda:
#                 img = img.cuda()
#
#             img = Variable(img)
#
#             output = self.model.forward(img)
#             output = output.data.cpu().numpy()
#
#             test_feats[idx * batch_size:end, :] = output
#             test_classes[idx * batch_size:end] = lbl
#             test_paths[idx * batch_size:end] = path
#
#             if mode != 'train':
#                 test_seen[idx * batch_size:end] = seen.to(int)
#
#         utils.save_h5(f'{mode}_ids', test_paths, 'S20', os.path.join(self.save_path, f'{mode}Ids.h5'))
#         utils.save_h5(f'{mode}_classes', test_classes, 'i8', os.path.join(self.save_path, f'{mode}Classes.h5'))
#         utils.save_h5(f'{mode}_feats', test_feats, 'f', os.path.join(self.save_path, f'{mode}Feats.h5'))
#         if mode != 'train':
#             utils.save_h5(f'{mode}_seen', test_seen, 'i2', os.path.join(self.save_path, f'{mode}Seen.h5'))
#
#         test_feats = utils.load_h5(f'{mode}_feats', os.path.join(self.save_path, f'{mode}Feats.h5'))
#         test_classes = utils.load_h5(f'{mode}_classes', os.path.join(self.save_path, f'{mode}Classes.h5'))
#         if mode != 'train':
#             test_seen = utils.load_h5(f'{mode}_seen', os.path.join(self.save_path, f'{mode}Seen.h5'))
#
#         if epoch != -1:
#             diff_class_path = os.path.join(self.gen_plot_path, f'{mode}/class_diff_plot.png')
#             self.plot_class_diff_plots(test_feats, test_classes,
#                                        epoch=epoch,
#                                        mode=mode,
#                                        path=diff_class_path)
#
#         silhouette_path = ['', '']
#         silhouette_path[0] = os.path.join(self.gen_plot_path, f'{mode}/silhouette_scores_plot.png')
#         silhouette_path[1] = os.path.join(self.gen_plot_path, f'{mode}/silhouette_scores_dist_plot_{epoch}.png')
#
#         self.plot_silhouette_score(test_feats, test_classes, epoch, mode, silhouette_path)
#
#         # import pdb
#         # pdb.set_trace()
#         if k_at_n:
#             utils.calculate_k_at_n(args, test_feats, test_classes, test_seen, logger=self.logger,
#                                    limit=args.limit_samples,
#                                    run_number=args.number_of_runs,
#                                    save_path=self.save_path,
#                                    sampled=eval_sampled,
#                                    per_class=eval_per_class,
#                                    mode=mode)
#
#             self.logger.info('results at: ' + self.save_path)
#
#     def plot_class_diff_plots(self, img_feats, img_classes, epoch, mode, path):
#         res = utils.get_euc_distances(img_feats, img_classes)
#         for k, v in self.class_diffs[mode].items():
#             v.append(res[k])
#
#         colors = ['r', 'b', 'y', 'g', 'c', 'm']
#         epochs = [i for i in range(1, epoch + 1)]
#         legends = []
#         colors_reordered = []
#
#         plt.figure(figsize=(10, 10))
#         for (k, v), c in zip(self.class_diffs[mode].items(), colors):
#             if len(v) > 1:
#                 plt.plot(epochs, v, color=c, linewidth=2, markersize=12)
#             else:
#                 plt.scatter(epochs, v, color=c)
#             legends.append(k)
#             colors_reordered.append(c)
#
#         plt.grid(True)
#         plt.xlabel('Epoch')
#         plt.ylabel('Euclidean Distance')
#         plt.xlim(left=0, right=epoch + 5)
#         plt.legend([Line2D([0], [0], color=colors_reordered[0], lw=4),
#                     Line2D([0], [0], color=colors_reordered[1], lw=4),
#                     Line2D([0], [0], color=colors_reordered[2], lw=4),
#                     Line2D([0], [0], color=colors_reordered[3], lw=4),
#                     Line2D([0], [0], color=colors_reordered[4], lw=4),
#                     Line2D([0], [0], color=colors_reordered[5], lw=4)], legends)
#
#         plt.title(f'{mode} class diffs')
#
#         plt.savefig(path)
#         plt.close('all')
#
#     def plot_silhouette_score(self, X, labels, epoch, mode, path):
#
#         self.silhouette_scores[mode].append(silhouette_score(X, labels, metric='euclidean'))
#         samples_silhouette = silhouette_samples(X, labels)
#
#         if epoch != -1:
#             epochs = [i for i in range(1, epoch + 1)]
#
#             plt.figure(figsize=(10, 10))
#             if len(self.silhouette_scores[mode]) > 1:
#                 plt.plot(epochs, self.silhouette_scores[mode], linewidth=2, markersize=12)
#             else:
#                 plt.scatter(epochs, self.silhouette_scores[mode])
#
#             plt.grid(True)
#             plt.xlabel('Epoch')
#             plt.ylabel(f'Silhouette Score')
#             plt.xlim(left=0, right=epoch + 5)
#
#             plt.title(f'Silhouette Scores for {mode} set')
#
#             plt.savefig(path[0])
#
#         plt.close('all')
#
#         plt.figure(figsize=(10, 10))
#
#         plt.hist(samples_silhouette, bins=40)
#
#         plt.grid(True)
#         plt.xlabel('Silhouette Score')
#         plt.ylabel(f'Freq')
#         plt.xlim(left=-1.1, right=1.1)
#
#         plt.title(f'Silhouette Scores Distribution on {mode} set')
#
#         plt.savefig(path[1])
#         plt.close('all')
