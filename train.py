import json
from argparse import Namespace

from torch.utils.data import DataLoader

import model_helper_functions
from losses import TripletLoss, MaxMarginLoss, BatchHard
from models.top_model import *
from my_datasets import *

###
# todo for next week

# Average per class for metrics (k@n) ???

EVAL_SET_NAMES = {1: ['total'],
                  2: ['seen', 'unseen']}



def main():
    args = utils.get_args()
    utils.MY_DEC.enabled = args.verbose

    with open(os.path.join(args.project_path, f'dataset_info_{args.env}.json'), 'r') as d:
        dataset_info = json.load(d)

    args_dict = vars(args)

    if args_dict['loss'] == 'batchhard':
        args_dict['batch_size'] = args_dict['bh_P'] * args_dict['bh_K']

    max_bt = np.maximum(args_dict['bcecoefficient'], args_dict['trplcoefficient'])
    bco = args_dict['bcecoefficient'] / max_bt
    tco = args_dict['trplcoefficient'] / max_bt

    args_dict['bcecoefficient'] = bco
    args_dict['trplcoefficient'] = tco

    args_dict.update(dataset_info[args.dataset_name])
    args = Namespace(**args_dict)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # args.dataset_folder = dataset_info[args.dataset_name][]
    model_name, id_str = utils.get_logname(args)

    logger = utils.get_logger(model_name, args.env)

    logger.info(f'Verbose: {args.verbose}')
    logger.info(f'Batch size is {args_dict["batch_size"]}')
    basic_aug = (args.overfit_num == 0)

    data_transforms_train, transform_list_train = utils.TransformLoader(args.image_size,
                                                                        rotate=args.rotate).get_composed_transform(
        aug=args.aug, random_crop=basic_aug)

    logger.info(f'train transforms: {transform_list_train}')

    data_transforms_val, transform_list_val = utils.TransformLoader(args.image_size,
                                                                    rotate=args.rotate).get_composed_transform(
        aug=args.aug, random_crop=False)
    logger.info(f'val transforms: {transform_list_val}')

    cam_data_transforms, cam_transform_list = utils.TransformLoader(args.image_size,
                                                                    rotate=args.rotate).get_composed_transform(
        aug=args.aug, random_crop=False, for_network=False)

    logger.info(f'cam transforms: {transform_list_val}')

    # import pdb
    # pdb.set_trace()

    if args.gpu_ids != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        logger.info(f"use gpu: {args.gpu_ids} to train.")

    train_set = None
    test_set = None
    val_set = None
    val_set_known_fewshot = None
    val_set_unknown_fewshot = None

    train_metric_dataset = None
    train_few_shot_dataset = None
    test_few_shot_dataset = None
    db_dataset = None


    logger.info('*' * 10)
    cam_train_set = cam_val_set_known_metric = cam_val_set_unknown_metric = None
    if args.cam:
        cam_img_paths = utils.read_img_paths(os.path.join(args.project_path, args.cam_path), local_path=args.local_path)
    else:
        cam_img_paths = None

    logger.info(str(cam_img_paths))

    # cam_train_set = train_metric_dataset(args, transform=data_transforms_val, mode='train', save_pictures=False,
    #                                      overfit=True, return_paths=True)
    # logger.info('*' * 10)
    # cam_val_set_known_metric = train_metric_dataset(args, transform=cam_data_transforms, mode='val_seen',
    #                                                 save_pictures=False, overfit=False, return_paths=True)
    # logger.info('*' * 10)
    # cam_val_set_unknown_metric = train_metric_dataset(args, transform=data_transforms_val, mode='val_unseen',
    #                                                   save_pictures=False, overfit=False, return_paths=True)
    is_batchhard = (args.loss == 'batchhard')
    logger.info('*' * 10)

    train_set = Metric_Dataset_Train(args, transform=data_transforms_train, mode=args.train_folder_name,
                                     save_pictures=False, overfit=True,
                                     batchhard=[is_batchhard, args.bh_P, args.bh_K])

    logger.info('*' * 10)
    val_set_known_metric = None
    if args.vs_folder_name != 'none':
        val_set_known_metric = Metric_Dataset_Train(args, transform=data_transforms_val, mode=args.vs_folder_name,
                                                    save_pictures=False, overfit=False,
                                                    batchhard=[False, args.bh_P, args.bh_K])
    else:
        val_set_known_metric = Metric_Dataset_Train(args, transform=data_transforms_val, mode=args.ts_folder_name,
                                                    save_pictures=False, overfit=False,
                                                    batchhard=[False, args.bh_P, args.bh_K])
    val_set_unknown_metric = None
    if args.vu_folder_name != 'none':
        logger.info('*' * 10)
        val_set_unknown_metric = Metric_Dataset_Train(args, transform=data_transforms_val, mode=args.vu_folder_name,
                                                      save_pictures=False, overfit=False,
                                                      batchhard=[False, args.bh_P, args.bh_K])
    test_set_known_metric = None
    test_set_unknown_metric = None
    if args.test:
        logger.info('*' * 10)
        test_set_known_metric = Metric_Dataset_Train(args, transform=data_transforms_val, mode=args.ts_folder_name,
                                                     save_pictures=False, overfit=False,
                                                     batchhard=[False, args.bh_P, args.bh_K])

        if args.tu_folder_name != 'none':
            logger.info('*' * 10)
            test_set_unknown_metric = Metric_Dataset_Train(args, transform=data_transforms_val,
                                                           mode=args.tu_folder_name,
                                                           save_pictures=False, overfit=False,
                                                           batchhard=[False, args.bh_P, args.bh_K])

    train_set_fewshot = FewShot_Dataset_Test(args, transform=data_transforms_train, mode=args.train_folder_name,
                                             save_pictures=False)

    if args.vs_folder_name != 'none':
        val_set_known_fewshot = FewShot_Dataset_Test(args, transform=data_transforms_val, mode=args.vs_folder_name,
                                                     save_pictures=False)
    else:
        val_set_known_fewshot = FewShot_Dataset_Test(args, transform=data_transforms_val, mode=args.ts_folder_name,
                                                     save_pictures=False)
    if args.vu_folder_name != 'none':
        logger.info('*' * 10)
        val_set_unknown_fewshot = FewShot_Dataset_Test(args, transform=data_transforms_val, mode=args.vu_folder_name,
                                                       save_pictures=False)

    # val_set_known_edgepred = test_edgepred_dataset(args, transform=data_transforms_val, mode='val_seen',
    #                                               save_pictures=False)
    # logger.info('*' * 10)
    # val_set_unknown_edgepred = test_edgepred_dataset(args, transform=data_transforms_val, mode='val_unseen',
    #                                                 save_pictures=False)

    if args.test:
        test_set_known = FewShot_Dataset_Test(args, transform=data_transforms_val, mode=args.ts_folder_name)
        logger.info('*' * 10)
        if args.tu_folder_name != 'none':
            test_set_unknown = FewShot_Dataset_Test(args, transform=data_transforms_val, mode=args.tu_folder_name)
            logger.info('*' * 10)

        # todo test not supported for metric learning

    val_db_set = None

    train_db_set = DB_Dataset(args, transform=data_transforms_val, mode=args.train_folder_name)

    if args.vs_folder_name != 'none':
        val_db_set = DB_Dataset(args, transform=data_transforms_val, mode=args.vs_folder_name)
    else:
        val_db_set = DB_Dataset(args, transform=data_transforms_val, mode=args.ts_folder_name)

    if args.test:
        test_db_set = DB_Dataset(args, transform=data_transforms_val, mode=args.ts_folder_name)
    # db_set_train = db_dataset(args, transform=data_transforms_val, mode='train_seen')  # 4 images per class

    logger.info(f'few shot evaluation way: {args.way}')

    # train_classify_loader = DataLoader(train_classification_dataset, batch_size=args.batch_size, shuffle=False,
    #                                    num_workers=args.workers)
    test_loaders = []

    if args.test:
        test_loaders.append(
            DataLoader(test_set_known, batch_size=args.way, shuffle=False, num_workers=args.workers,
                       drop_last=args.drop_last))
        if args.tu_folder_name != 'none':
            test_loaders.append(
                DataLoader(test_set_unknown, batch_size=args.way, shuffle=False, num_workers=args.workers,
                           drop_last=args.drop_last))

    # workers = 4
    # pin_memory = False
    if args.find_best_workers:
        workers, pin_memory = utils.get_best_workers_pinmemory(args, train_set,
                                                               pin_memories=[True],
                                                               starting_from=4,
                                                               logger=logger)
    else:
        workers = args.workers
        pin_memory = args.pin_memory

    if args.loss == 'batchhard':
        bs = args.bh_P
    else:
        bs = args.batch_size

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=False, num_workers=workers,
                              pin_memory=pin_memory, drop_last=args.drop_last)

    train_loader_fewshot = DataLoader(train_set_fewshot, batch_size=args.way, shuffle=False, num_workers=workers,
                                      pin_memory=pin_memory, drop_last=args.drop_last)

    val_loaders_fewshot = utils.get_val_loaders(args, val_set_known_fewshot, val_set_unknown_fewshot, workers,
                                                pin_memory)

    val_loaders_edgepred = None
    # val_loaders_edgepred = utils.get_val_loaders(args, val_set, val_set_known_edgepred, val_set_unknown_edgepred, workers,
    #                                             pin_memory, batch_size=args.way * args.test_k)

    dl_cam_train = dl_cam_val_known = dl_cam_val_unknown = None
    # if args.cam:
    # dl_cam_train = DataLoader(cam_train_set, batch_size=1, shuffle=False, num_workers=workers,
    #                           pin_memory=pin_memory)
    # dl_cam_val_known = DataLoader(cam_val_set_known_metric, batch_size=1, shuffle=False, num_workers=workers,
    #                               pin_memory=pin_memory)
    # dl_cam_val_unknown = DataLoader(cam_val_set_unknown_metric, batch_size=1, shuffle=False, num_workers=workers,
    #                                 pin_memory=pin_memory)

    val_loaders_metric = utils.get_val_loaders(args, val_set_known_metric, val_set_unknown_metric, workers,
                                               pin_memory, batch_size=args.batch_size)
    if args.test:
        test_loaders_metric = utils.get_val_loaders(args, test_set_known_metric, test_set_unknown_metric, workers,
                                                    pin_memory, batch_size=args.batch_size)

        # train_loader_classify = DataLoader(train_classify, batch_size=args.batch_size, shuffle=False,
        #                                    num_workers=workers,
        #                                    pin_memory=pin_memory)
        # val_loader_classify = DataLoader(val_classify, batch_size=args.batch_size, shuffle=False, num_workers=workers,
        #                                  pin_memory=pin_memory)
    val_db_loader = None

    train_db_loader = DataLoader(train_db_set, batch_size=args.db_batch, shuffle=False, num_workers=workers,
                                 pin_memory=pin_memory, drop_last=args.drop_last)

    val_db_loader = DataLoader(val_db_set, batch_size=args.db_batch, shuffle=False, num_workers=workers,
                               pin_memory=pin_memory, drop_last=args.drop_last)

    if args.test:
        test_db_loader = DataLoader(test_db_set, batch_size=args.db_batch, shuffle=False, num_workers=workers,
                                    pin_memory=pin_memory, drop_last=args.drop_last)

    if args.loss == 'bce':
        loss_fn_bce = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss_fn = None
    elif args.loss == 'trpl':
        loss_fn_bce = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss_fn = TripletLoss(margin=args.margin, args=args, soft=args.softmargin)
    elif args.loss == 'maxmargin':
        loss_fn_bce = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss_fn = MaxMarginLoss(margin=args.margin, args=args)
        # loss_fn = torch.nn.TripletMarginLoss(margin=args.margin, p=2)
    elif args.loss == 'batchhard':
        loss_fn_bce = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss_fn = BatchHard(margin=args.margin, args=args, soft=args.softmargin)
    else:
        raise Exception('Loss function not supported: ' + args.loss)

    num_classes = train_set.num_classes
    logger.info(f'Num classes in train: {num_classes}')

    cam_images_len = len(cam_img_paths) if cam_img_paths is not None else 0

    if args.baseline_model != '':
        net = utils.get_resnet(args, args.baseline_model)
        model_methods_top = model_helper_functions.BaslineModel(args=args, logger=logger, model=net, loss_fn=loss_fn,
                                                                model_name=model_name, id_str=id_str)
    else:
        model_methods_top = model_helper_functions.ModelMethods(args, logger, 'top', cam_images_len=cam_images_len,
                                                                model_name=model_name, id_str=id_str)
        net = top_module(args=args, num_classes=num_classes, mask=args.aug_mask, fourth_dim=args.fourth_dim)

    logger.info(model_methods_top.save_path)

    # multi gpu
    # if len(args.gpu_ids.split(",")) > 1:
    #     tm_net = torch.nn.DataParallel(tm_net)
    #
    # device = f'cuda:0'

    # device = f'cuda:{tm_net.device_ids[0]}'

    if args.cuda:
        if torch.cuda.device_count() > 1:
            logger.info(f'torch.cuda.device_count() = {torch.cuda.device_count()}')
            net = nn.DataParallel(net)
        logger.info(f'Let\'s use {torch.cuda.device_count()} GPUs!')
        utils.print_gpu_stuff(args.cuda, 'before model to gpu')
        net = net.cuda()
        utils.print_gpu_stuff(args.cuda, 'after model to gpu')

    logger.info('Training Top')
    if args.baseline_model != '':
        logger.info('Training')
        model_methods_top.train(args, train_loader, val_db_loader)

    if args.pretrained_model_name == '':
        logger.info('Training')
        print('Total parameters:', utils.get_number_of_parameters(net))
        logger.info(f'Total parameters: {utils.get_number_of_parameters(net)}')
        net, best_model_top = model_methods_top.train_metriclearning(net=net, loss_fn=loss_fn,
                                                                     bce_loss=loss_fn_bce, args=args,
                                                                     train_loader=train_loader,
                                                                     val_loaders=val_loaders_metric,
                                                                     val_loaders_fewshot=val_loaders_fewshot,
                                                                     train_loader_fewshot=train_loader_fewshot,
                                                                     cam_args=[cam_img_paths,
                                                                               data_transforms_val,
                                                                               cam_data_transforms],
                                                                     db_loaders=[train_db_loader, val_db_loader],
                                                                     val_loaders_edgepred=val_loaders_edgepred)
        logger.info('Calculating K@Ns for Validation')

        # model_methods_top.make_emb_db(args, tm_net, db_loader_train,
        #                               eval_sampled=False,
        #                               eval_per_class=True, newly_trained=True,
        #                               batch_size=args.db_batch,
        #                               mode='train_sampled')
        if val_db_loader:
            model_methods_top.make_emb_db(args, net, val_db_loader,
                                          eval_sampled=args.sampled_results,
                                          eval_per_class=args.per_class_results, newly_trained=False,
                                          batch_size=args.db_batch,
                                          mode='val')
    else:  # test
        logger.info('Testing without training')
        best_model_top = args.pretrained_model_name
        logger.info(f"Not training, loading {best_model_top} model...")
        net = model_methods_top.load_model(args, net, best_model_top)

    if args.cam and args.pretrained_model_name != '':
        logger.info(f'Drawing heatmaps on epoch {-1}...')
        model_methods_top.draw_heatmaps(net=net,
                                        loss_fn=loss_fn,
                                        bce_loss=loss_fn_bce,
                                        args=args,
                                        cam_loader=cam_img_paths,
                                        transform_for_model=data_transforms_val,
                                        transform_for_heatmap=cam_data_transforms,
                                        epoch=-1,
                                        count=1,
                                        draw_all_thresh=args.draw_all_thresh)
        logger.info(f'DONE drawing heatmaps on epoch {-1}!!!')

    if args.katn and args.pretrained_model_name != '':
        logger.info('Calculating K@Ns for Validation')
        # model_methods_top.make_emb_db(args, tm_net, db_loader_train,
        #                               eval_sampled=False,
        #                               eval_per_class=True, newly_trained=True,
        #                               batch_size=args.db_batch,
        #                               mode='train_sampled')
        if val_db_loader:
            model_methods_top.make_emb_db(args, net, val_db_loader,
                                          eval_sampled=args.sampled_results,
                                          eval_per_class=args.per_class_results, newly_trained=True,
                                          batch_size=args.db_batch,
                                          mode='val')

    # testing
    if args.test:

        logger.info(f"Loading {best_model_top} model...")
        net = model_methods_top.load_model(args, net, best_model_top)
        with torch.no_grad():
            for tlm, comm in zip(test_loaders_metric, EVAL_SET_NAMES[len(test_loaders_metric)]):
                model_methods_top.test_metric(args, net, tlm, loss_fn, loss_fn_bce, val=False, epoch=-1, comment=comm)

            if args.katn:
                logger.info('Calculating K@Ns for Test')
                # model_methods_top.make_emb_db(args, tm_net, db_loader_train,
                #                               eval_sampled=False,
                #                               eval_per_class=True, newly_trained=True,
                #                               batch_size=args.db_batch,
                #                               mode='train_sampled')
                model_methods_top.make_emb_db(args, net, test_db_loader,
                                              eval_sampled=args.sampled_results,
                                              eval_per_class=args.per_class_results, newly_trained=True,
                                              batch_size=args.db_batch,
                                              mode='test')

    else:
        logger.info("NO TESTING DONE.")
    #  learning_rate = learning_rate * 0.95


if __name__ == '__main__':
    main()

# python3 /home/aarash/projects/def-rrabba/aarash/ht-image-twoloss/ht-image/train.py -cuda     -env beluga     -dsp ~/scratch/aarash/     -dsn hotels     -fe resnet50     -tbp tensorboard_hlr/         -sp savedmodels         -gpu 0         -wr 10     -pim     -w 10         -bs 10         -tf 1         -sf 1         -ep 5         -lrs 0.1     -lrr 0.03     -por 5000     -es 20     -cbir     -dbb 60     -el 0     -nor 200     -ls 4     -mtlr     -lss batchhard     -mg 1              -bco 1   -ppth ./     -lpth ./     -jid $SLURM_JOB_ID -bm resnet50 -k 4 -p 18
