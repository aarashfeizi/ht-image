import json
import logging
import os
import sys
from argparse import Namespace

from torch.utils.data import DataLoader

import model_helper_functions
import utils
from cub_dataloader import *
from hotel_dataloader import *
from losses import TripletLoss, MaxMarginLoss
from models.top_model import *


###
# todo for next week

# Average per class for metrics (k@n) ???


def _logger():
    logging.basicConfig(format='%(asctime)s - %(message)s', stream=sys.stdout, level=logging.INFO)
    return logging.getLogger()


def main():
    logger = _logger()

    args = utils.get_args()

    with open(f'./dataset_info_{args.env}.json', 'r') as d:
        dataset_info = json.load(d)

    args_dict = vars(args)
    args_dict.update(dataset_info[args.dataset_name])
    args = Namespace(**args_dict)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # args.dataset_folder = dataset_info[args.dataset_name][]

    basic_aug = (args.overfit_num == 0)

    data_transforms_train, transform_list_train = utils.TransformLoader(args.image_size,
                                                                        rotate=args.rotate).get_composed_transform(
        aug=args.aug, random_crop=basic_aug, basic_aug=basic_aug)

    logger.info(f'train transforms: {transform_list_train}')

    data_transforms_val, transform_list_val = utils.TransformLoader(args.image_size,
                                                                    rotate=args.rotate).get_composed_transform(
        aug=args.aug, random_crop=False, basic_aug=basic_aug)
    logger.info(f'val transforms: {transform_list_val}')

    if args.gpu_ids != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print("use gpu:", args.gpu_ids, "to train.")

    train_set = None
    test_set = None
    val_set = None
    val_set_known_fewshot = None
    val_set_unknown_fewshot = None

    train_metric_dataset = None
    train_few_shot_dataset = None
    test_few_shot_dataset = None
    db_dataset = None

    if args.dataset_name == 'hotels':
        train_metric_dataset = HotelTrain_Metric
        train_few_shot_dataset = HotelTrain_FewShot
        test_few_shot_dataset = HotelTest_FewShot
        db_dataset = Hotel_DB
    elif args.dataset_name == 'cub':
        train_metric_dataset = CUBTrain_Metric
        train_few_shot_dataset = CUBTrain_FewShot
        test_few_shot_dataset = CUBTest_FewShot
        db_dataset = CUB_DB
    else:
        logger.error(f'Dataset not suppored:  {args.dataset_name}')

    # train_classification_dataset = CUBClassification(args, transform=data_transforms, mode='train')

    print('*' * 10)
    if args.metric_learning:
        train_set = train_metric_dataset(args, transform=data_transforms_train, mode='train', save_pictures=False,
                                         overfit=True)
        print('*' * 10)
        val_set_known_metric = train_metric_dataset(args, transform=data_transforms_val, mode='val_seen',
                                                    save_pictures=False, overfit=False)
        print('*' * 10)
        val_set_unknown_metric = train_metric_dataset(args, transform=data_transforms_val, mode='val_unseen',
                                                      save_pictures=False, overfit=False)


    else:
        train_set = train_few_shot_dataset(args, transform=data_transforms_train, mode='train', save_pictures=False)
        print('*' * 10)

    train_set_fewshot = test_few_shot_dataset(args, transform=data_transforms_train, mode='train', save_pictures=False)

    val_set_known_fewshot = test_few_shot_dataset(args, transform=data_transforms_val, mode='val_seen',
                                                  save_pictures=False)
    print('*' * 10)
    val_set_unknown_fewshot = test_few_shot_dataset(args, transform=data_transforms_val, mode='val_unseen',
                                                    save_pictures=False)

    if args.test:
        test_set_known = test_few_shot_dataset(args, transform=data_transforms_val, mode='test_seen')
        print('*' * 10)
        test_set_unknown = test_few_shot_dataset(args, transform=data_transforms_val, mode='test_unseen')
        print('*' * 10)

        # todo test not supported for metric learning

    if args.cbir:
        db_set = db_dataset(args, transform=data_transforms_val, mode='val')
        # db_set_train = db_dataset(args, transform=data_transforms_val, mode='train_seen')  # 4 images per class

    logger.info(f'few shot evaluation way: {args.way}')

    # train_classify_loader = DataLoader(train_classification_dataset, batch_size=args.batch_size, shuffle=False,
    #                                    num_workers=args.workers)
    test_loaders = []

    if args.test:
        test_loaders.append(
            DataLoader(test_set_known, batch_size=args.way, shuffle=False, num_workers=args.workers))
        test_loaders.append(
            DataLoader(test_set_unknown, batch_size=args.way, shuffle=False, num_workers=args.workers))

    # workers = 4
    # pin_memory = False
    if args.find_best_workers:
        workers, pin_memory = utils.get_best_workers_pinmemory(args, train_set,
                                                               pin_memories=[True],
                                                               starting_from=0,
                                                               logger=logger)
    else:
        workers = args.workers
        pin_memory = args.pin_memory

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=workers,
                              pin_memory=pin_memory)

    train_loader_fewshot = DataLoader(train_set_fewshot, batch_size=args.way, shuffle=False, num_workers=workers,
                                      pin_memory=pin_memory)

    val_loaders_fewshot = utils.get_val_loaders(args, val_set, val_set_known_fewshot, val_set_unknown_fewshot, workers,
                                                pin_memory)

    if args.metric_learning:
        val_loaders_metric = utils.get_val_loaders(args, val_set, val_set_known_metric, val_set_unknown_metric, workers,
                                                   pin_memory, batch_size=args.batch_size)

        # train_loader_classify = DataLoader(train_classify, batch_size=args.batch_size, shuffle=False,
        #                                    num_workers=workers,
        #                                    pin_memory=pin_memory)
        # val_loader_classify = DataLoader(val_classify, batch_size=args.batch_size, shuffle=False, num_workers=workers,
        #                                  pin_memory=pin_memory)

    if args.cbir:
        db_loader = DataLoader(db_set, batch_size=args.db_batch, shuffle=False, num_workers=workers,
                               pin_memory=pin_memory)

        # db_loader_train = DataLoader(db_set_train, batch_size=args.db_batch, shuffle=False, num_workers=workers,
        #                              pin_memory=pin_memory)

    if args.loss == 'bce':
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    elif args.loss == 'trpl':
        loss_fn_bce = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss_fn = TripletLoss(margin=args.margin, args=args)
    elif args.loss == 'maxmargin':
        loss_fn_bce = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss_fn = MaxMarginLoss(margin=args.margin, args=args)
        # loss_fn = torch.nn.TripletMarginLoss(margin=args.margin, p=2)
    else:
        raise Exception('Loss function not supported: ' + args.loss)

    num_classes = train_set.num_classes
    logger.info(f'Num classes in train: {num_classes}')

    model_methods_top = model_helper_functions.ModelMethods(args, logger, 'top')
    tm_net = top_module(args=args, num_classes=num_classes, mask=args.mask)

    print(model_methods_top.save_path)

    # multi gpu
    if len(args.gpu_ids.split(",")) > 1:
        tm_net = torch.nn.DataParallel(tm_net)

    if args.cuda:
        tm_net.cuda()

    logger.info('Training Top')
    if args.pretrained_model_name == '':
        logger.info('Training')
        if args.metric_learning:
            tm_net, best_model_top = model_methods_top.train_metriclearning(net=tm_net, loss_fn=loss_fn,
                                                                            bce_loss=loss_fn_bce, args=args,
                                                                            train_loader=train_loader,
                                                                            val_loaders=val_loaders_metric,
                                                                            val_loaders_fewshot=val_loaders_fewshot,
                                                                            train_loader_fewshot=train_loader_fewshot)
        else:
            tm_net, best_model_top = model_methods_top.train_fewshot(net=tm_net, loss_fn=loss_fn, args=args,
                                                                     train_loader=train_loader,
                                                                     val_loaders=val_loaders_fewshot)
        logger.info('Calculating K@Ns for Validation')

        # model_methods_top.make_emb_db(args, tm_net, db_loader_train,
        #                               eval_sampled=False,
        #                               eval_per_class=True, newly_trained=True,
        #                               batch_size=args.db_batch,
        #                               mode='train_sampled')

        model_methods_top.make_emb_db(args, tm_net, db_loader,
                                      eval_sampled=args.sampled_results,
                                      eval_per_class=args.per_class_results, newly_trained=True,
                                      batch_size=args.db_batch,
                                      mode='val')
    else:  # test
        logger.info('Testing')
        best_model_top = args.pretrained_model_name

    if args.katn and args.pretrained_model_name != '':
        logger.info(f"Not training, loading {best_model_top} model...")
        tm_net = model_methods_top.load_model(args, tm_net, best_model_top)
        logger.info('Calculating K@Ns for Validation')
        # model_methods_top.make_emb_db(args, tm_net, db_loader_train,
        #                               eval_sampled=False,
        #                               eval_per_class=True, newly_trained=True,
        #                               batch_size=args.db_batch,
        #                               mode='train_sampled')
        model_methods_top.make_emb_db(args, tm_net, db_loader,
                                      eval_sampled=args.sampled_results,
                                      eval_per_class=args.per_class_results, newly_trained=False,
                                      batch_size=args.db_batch,
                                      mode='val')

    # testing
    if args.test:
        logger.info(f"Loading {best_model_top} model...")
        tm_net = model_methods_top.load_model(args, tm_net, best_model_top)

        model_methods_top.test_fewshot(args, tm_net, test_loaders[0], loss_fn, comment='known')
        model_methods_top.test_fewshot(args, tm_net, test_loaders[1], loss_fn, comment='unknown')
    else:
        logger.info("NO TESTING DONE.")
    #  learning_rate = learning_rate * 0.95


if __name__ == '__main__':
    main()
