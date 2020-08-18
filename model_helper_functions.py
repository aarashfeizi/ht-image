import datetime
import os
import pickle
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
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

    def _parse_args(self, args):
        name = 'model-' + self.model

        important_args = ['batch_size',
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
                          'overfit_num']

        for arg in vars(args):
            if str(arg) in important_args:
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

    def train_metriclearning(self, net, loss_fn, bce_loss, args, train_loader, val_loaders, val_loaders_fewshot):
        net.train()
        val_tol = args.early_stopping

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

        # print('steps:', args.max_steps)

        # epochs = int(np.ceil(args.max_steps / len(trainLoader)))
        epochs = args.epochs

        metric_ACC = metrics.Metric_Accuracy()

        max_val_acc = 0
        max_val_acc_knwn = 0
        max_val_acc_unknwn = 0
        best_model = ''

        drew_graph = False

        val_counter = 0

        for epoch in range(epochs):

            train_loss = 0
            train_bce_loss = 0
            train_triplet_loss = 0

            metric_ACC.reset_acc()

            with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{args.epochs}') as t:
                for batch_id, (anch, pos, neg) in enumerate(train_loader, 1):

                    # print('input: ', img1.size())

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


                    norm_pos_dist, anch_feat, pos_feat = net.forward(anch, pos, feats=True)
                    print(f'norm pos: {norm_pos_dist}')
                    class_loss = bce_loss(norm_pos_dist.squeeze(), zero_labels.squeeze())
                    metric_ACC.update_acc(norm_pos_dist.squeeze(), zero_labels.squeeze())  # zero dist means similar
                    print('pos loss', class_loss)

                    # bce_loss_value_pos = bce_loss(output_pos.squeeze(), one_labels.squeeze())
                    # train_loss_bces += (bce_loss_value_pos.item())
                    # neg_bce_losses = 0
                    for iter in range(self.no_negative):
                        norm_neg_dist, _, neg_feat = net.forward(anch, neg[:, iter, :, :, :].squeeze(dim=1), feats=True)
                        #
                        # self.logger.info(f'pos_dist = {(norm_pos_dist ** 2).sum(dim=1)}')
                        # self.logger.info(f'neg_dist = {(norm_neg_dist ** 2).sum(dim=1)}')
                        # self.logger.info(f'pos - neg = {(norm_pos_dist ** 2).sum(dim=1) - (norm_neg_dist ** 2).sum(dim=1)}')
                        # self.logger.info(f'pos_dist_total = {sum((norm_pos_dist ** 2).sum(dim=1))}')
                        # self.logger.info(f'neg_dist_total = {sum((norm_neg_dist ** 2).sum(dim=1))}')
                        print(f'norm neg {iter}: {norm_neg_dist.reshape((1, -1))}')


                        class_loss += bce_loss(norm_neg_dist.squeeze(), one_labels.squeeze())

                        metric_ACC.update_acc(norm_neg_dist.squeeze(), one_labels.squeeze())  # 1 dist means different

                        print('neg loss', class_loss)

                        if iter == 0:
                            ext_loss = loss_fn(anch_feat, pos_feat, neg_feat)
                        else:
                            ext_loss += loss_fn(anch_feat, pos_feat, neg_feat)

                        # bce_loss_value_neg = bce_loss(output_neg.squeeze(), zero_labels.squeeze())

                        # neg_bce_losses += (bce_loss_value_neg.item())
                    # print('loss: ', loss.item())

                    # train_loss_bces += neg_bce_losses / self.no_negative

                    ext_loss /= self.no_negative
                    class_loss /= (self.no_negative + 1)

                    loss = ext_loss + class_loss

                    train_loss += loss.item()
                    train_triplet_loss += ext_loss.item()
                    train_bce_loss += class_loss.item()
                    loss.backward()  # training with triplet loss
                    opt.step()
                    # plt = self.plot_grad_flow(net.named_parameters())
                    # pdb.set_trace()
                    # self.getBack(loss.grad_fn)

                    t.set_postfix(loss=f'{train_loss / (batch_id) :.4f}',
                                  bce_loss=f'{train_bce_loss / batch_id:.4f}',
                                  triplet_loss=f'{train_triplet_loss / batch_id:.4f}',
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

                self.writer.add_scalar('Train/Loss', train_loss / len(train_loader), epoch)
                self.writer.add_scalar('Train/Triplet_Loss', train_triplet_loss / len(train_loader), epoch)
                self.writer.add_scalar('Train/BCE_Loss', train_bce_loss / len(train_loader), epoch)

                self.writer.add_scalar('Train/Acc', metric_ACC.get_acc(), epoch)
                self.writer.flush()

                if val_loaders is not None and epoch % args.test_freq == 0:
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

                if val_loaders is not None and epoch % args.test_freq == 0:
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
            norm_pos_dist, anch_feat, pos_feat = net.forward(anch, pos, feats=True)
            class_loss = bce_loss(norm_pos_dist.squeeze(), zero_labels.squeeze())

            for iter in range(self.no_negative):
                print(anch.shape)
                print(neg[:, iter, :, :, :].squeeze(dim=1).shape)
                norm_neg_dist, _, neg_feat = net.forward(anch, neg[:, iter, :, :, :].squeeze(dim=1),
                                                    feats=True)

                class_loss += bce_loss(norm_neg_dist.squeeze(), one_labels.squeeze())

                if iter == 0:
                    ext_loss = loss_fn(anch_feat, pos_feat, neg_feat)
                else:
                    ext_loss += loss_fn(anch_feat, pos_feat, neg_feat)

            ext_loss /= self.no_negative
            class_loss /= (self.no_negative + 1)

            loss = ext_loss + class_loss
            test_loss += loss.item()
            test_triplet_loss += ext_loss.item()
            test_bce_loss += class_loss.item()

        self.logger.info('$' * 70)

        # self.writer.add_scalar(f'{prompt_text_tb}/Triplet_Loss', test_loss / len(data_loader), epoch)
        self.writer.add_scalar(f'{prompt_text_tb}/Loss', test_loss / len(data_loader), epoch)
        self.writer.add_scalar(f'{prompt_text_tb}/Triplet_Loss', test_triplet_loss / len(data_loader), epoch)
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

        tests_right, tests_error = 0, 0

        test_label = np.ones(shape=args.way, dtype=np.float32)
        test_label[0] = 0
        test_label = torch.from_numpy(test_label).reshape((-1, 1))
        test_loss = 0
        if args.cuda:
            test_label = Variable(test_label.cuda())
        else:
            test_label = Variable(test_label)

        for _, (test1, test2) in enumerate(data_loader, 1):
            if args.cuda:
                test1, test2 = test1.cuda(), test2.cuda()
            test1, test2 = Variable(test1), Variable(test2)
            output = net.forward(test1, test2)
            test_loss += loss_fn(output, test_label).item()
            output = output.data.cpu().numpy()
            pred = np.argmin(output)
            if pred == 0:
                tests_right += 1
            else:
                tests_error += 1

        test_acc = tests_right * 1.0 / (tests_right + tests_error)
        self.logger.info('$' * 70)
        self.logger.info(prompt_text % (tests_right, tests_error, test_acc, test_loss))
        self.logger.info('$' * 70)

        self.writer.add_scalar(f'{prompt_text_tb}/Loss', test_loss, epoch)
        self.writer.add_scalar(f'{prompt_text_tb}/Acc', test_acc, epoch)
        self.writer.flush()

        return tests_right, tests_error, test_acc

    def make_emb_db(self, args, net, data_loader, eval_sampled, eval_per_class, newly_trained=True, batch_size=None,
                    mode='val'):
        """

        :param batch_size:
        :param eval_sampled:
        :param eval_per_class:
        :param newly_trained:
        :param mode:
        :param args: utils args
        :param net: trained top_model network
        :param data_loader: DataLoader object
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

            for idx, (img, lbl, seen, path) in enumerate(data_loader):

                if args.cuda:
                    img = img.cuda()
                img = Variable(img)

                output = net.forward(img, None, single=True)
                output = output.data.cpu().numpy()

                end = min((idx + 1) * batch_size, len(test_feats))

                test_feats[idx * batch_size:end, :] = output
                test_classes[idx * batch_size:end] = lbl
                test_paths[idx * batch_size:end] = path
                test_seen[idx * batch_size:end] = seen.to(int)

            utils.save_h5(f'{mode}_ids', test_paths, 'S20', os.path.join(self.save_path, f'{mode}Ids.h5'))
            utils.save_h5(f'{mode}_classes', test_classes, 'i8', os.path.join(self.save_path, f'{mode}Classes.h5'))
            utils.save_h5(f'{mode}_feats', test_feats, 'f', os.path.join(self.save_path, f'{mode}Feats.h5'))
            utils.save_h5(f'{mode}_seen', test_seen, 'i2', os.path.join(self.save_path, f'{mode}Seen.h5'))

        test_feats = utils.load_h5(f'{mode}_feats', os.path.join(self.save_path, f'{mode}Feats.h5'))
        test_classes = utils.load_h5(f'{mode}_classes', os.path.join(self.save_path, f'{mode}Classes.h5'))
        test_seen = utils.load_h5(f'{mode}_seen', os.path.join(self.save_path, f'{mode}Seen.h5'))

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
