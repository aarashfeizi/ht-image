import argparse
import json
import random

import torch
from matplotlib.lines import Line2D
from sklearn.metrics.pairwise import cosine_distances
from torch.autograd import Variable

import model_helper_functions
import utils
from models import resnet
from my_datasets import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def main():
    args = utils.get_args()

    with open(os.path.join(args.project_path, f'dataset_info_{args.env}.json'), 'r') as d:
        dataset_info = json.load(d)

    args_dict = vars(args)

    args_dict.update(dataset_info[args.dataset_name])
    args = argparse.Namespace(**args_dict)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # args.dataset_folder = dataset_info[args.dataset_name][]
    model_name, id_str = utils.get_logname(args)

    logger = utils.get_logger(model_name, args.env)

    data_transforms_val, transform_list_val = utils.TransformLoader(args.image_size,
                                                                    rotate=args.rotate).get_composed_transform(
        aug=args.aug, random_crop=False)
    logger.info(f'val transforms: {transform_list_val}')

    net = resnet.simple_resnet50(args, pretrained=True)

    if args.cuda:
        net.cuda()

    workers = args.workers
    pin_memory = args.pin_memory


    if args.vs_folder_name != 'none':
        val_db_set = DB_Dataset(args, transform=data_transforms_val, mode=args.vs_folder_name)
    else:
        val_db_set = DB_Dataset(args, transform=data_transforms_val, mode=args.ts_folder_name)

    val_db_loader = DataLoader(val_db_set, batch_size=args.db_batch, shuffle=False, num_workers=workers,
                               pin_memory=pin_memory, drop_last=args.drop_last)

    if args.test:
        test_db_set = DB_Dataset(args, transform=data_transforms_val, mode=args.ts_folder_name)

        test_db_loader = DataLoader(test_db_set, batch_size=args.db_batch, shuffle=False, num_workers=workers,
                                    pin_memory=pin_memory, drop_last=args.drop_last)

    model_methods = model_helper_functions.ModelMethods(args, logger, 'top', cam_images_len=0,
                                                            model_name=model_name, id_str=id_str)

    embbeddings, labels, seens = model_methods.get_embeddings(args, net, val_db_loader)

    embedding_name = f'{args.dataset_name}_{args.pretrained_model}_{args.extra_name}'

    utils.save_h5(f'{embedding_name}_classes', labels, 'i8',
                  os.path.join(model_methods.save_path, f'{embedding_name}_Classes.h5'))
    utils.save_h5(f'{embedding_name}_feats', embbeddings, 'f',
                  os.path.join(model_methods.save_path, f'{embedding_name}_Feats.h5'))
    utils.save_h5(f'{embedding_name}_seen', seens, 'i2',
                  os.path.join(model_methods.save_path, f'{embedding_name}_Seen.h5'))



if __name__ == '__main__':
    main()
