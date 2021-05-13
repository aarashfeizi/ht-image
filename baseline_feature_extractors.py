import argparse
import json

from torch.utils.data import DataLoader

import model_helper_functions
from models.resnet import simple_resnet50, simple_resnet18
from my_datasets import *

MODEL_BACKBONE = {'resnet50': simple_resnet50,
                  'resnet18': simple_resnet18}


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

    if args.gpu_ids != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        logger.info(f"use gpu: {args.gpu_ids} to train.")

    data_transforms_val, transform_list_val = utils.TransformLoader(args.image_size,
                                                                    rotate=args.rotate).get_composed_transform(
        aug=args.aug, random_crop=False)
    logger.info(f'val transforms: {transform_list_val}')

    net = MODEL_BACKBONE[args.feat_extractor](args, pretrained=True)

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

    if args.store_features_knn:
        train_db_set = DB_Dataset(args, transform=data_transforms_val, mode=args.train_folder_name)

        train_db_loader = DataLoader(train_db_set, batch_size=args.db_batch, shuffle=False, num_workers=workers,
                                     pin_memory=pin_memory, drop_last=args.drop_last)



        embeddings, labels, seens, ids = model_methods.get_embeddings(args, net, train_db_loader)
        print(embeddings.shape)
        input()
        knn_path = os.path.join(model_methods.save_path, 'knn')
        utils.save_knn(embeddings, knn_path, gpu=args.cuda)

        with open(os.path.join(knn_path, 'labels.pkl'), 'wb') as f:
            import pickle
            pickle.dump(labels, f)

        loader = train_db_loader

    elif not args.test:
        embeddings, labels, seens, ids = model_methods.get_embeddings(args, net, val_db_loader)
        loader = val_db_loader

        embedding_name = f'{args.dataset_name}_val_{args.pretrained_model}_{args.extra_name}'

        utils.save_h5(f'{embedding_name}_classes', labels, 'i8',
                      os.path.join(model_methods.save_path, f'{embedding_name}_Classes.h5'))
        utils.save_h5(f'{embedding_name}_feats', embeddings, 'f',
                      os.path.join(model_methods.save_path, f'{embedding_name}_Feats.h5'))
        utils.save_h5(f'{embedding_name}_seen', seens, 'i2',
                      os.path.join(model_methods.save_path, f'{embedding_name}_Seen.h5'))

    else:
        embeddings, labels, seens, ids = model_methods.get_embeddings(args, net, test_db_loader)
        loader = test_db_loader

        embedding_name = f'{args.dataset_name}_test_{args.pretrained_model}_{args.extra_name}'

        utils.save_h5(f'{embedding_name}_classes', labels, 'i8',
                      os.path.join(model_methods.save_path, f'{embedding_name}_Classes.h5'))
        utils.save_h5(f'{embedding_name}_feats', embeddings, 'f',
                      os.path.join(model_methods.save_path, f'{embedding_name}_Feats.h5'))
        utils.save_h5(f'{embedding_name}_seen', seens, 'i2',
                      os.path.join(model_methods.save_path, f'{embedding_name}_Seen.h5'))

    unique_seens = np.unique(seens)
    prmpt = 'Test results' if args.test else 'Val results'
    print(prmpt)
    if len(unique_seens) == 2:
        seen_res = utils.evaluation(args, embeddings[seens == 1], labels[seens == 1],
                                    ids[seens == 1], model_methods.writer, loader,
                                    Kset=[1, 2, 4, 5, 8, 10, 100, 1000], split='seen', path=model_methods.save_path,
                                    gpu=args.cuda, tb_draw=True)
        print(f'Seen length: {len(labels[seens == 1])}')
        print(f'K@1, K@2, K@4, K@5, K@8, K@10, K@100, K@1000')
        print(seen_res)
        unseen_res = utils.evaluation(args, embeddings[seens == 0], labels[seens == 0],
                                      ids[seens == 0], model_methods.writer, loader,
                                      Kset=[1, 2, 4, 5, 8, 10, 100, 1000], split='unseen', path=model_methods.save_path,
                                      gpu=args.cuda, tb_draw=True)
        print(f'Unseen length: {len(labels[seens == 0])}')
        print(f'K@1, K@2, K@4, K@5, K@8, K@10, K@100, K@1000')
        print(unseen_res)

        res = utils.evaluation(args, embeddings, labels, ids, model_methods.writer,
                               loader, Kset=[1, 2, 4, 5, 8, 10, 100, 1000], split='total', path=model_methods.save_path,
                               gpu=args.cuda, tb_draw=True)
        print(f'Total length: {len(labels)}')
        print(f'K@1, K@2, K@4, K@5, K@8, K@10, K@100, K@1000')
        print(res)

    elif len(unique_seens) == 1:
        res = utils.evaluation(args, embeddings, labels, ids, model_methods.writer,
                               loader, Kset=[1, 2, 4, 5, 8, 10, 100, 1000], split='total', path=model_methods.save_path,
                               gpu=args.cuda, path_to_lbl2chain=os.path.join(args.splits_file_path, 'label2chain.csv'),
                               tb_draw=True)
        print(f'Total length: {len(labels)}')
        print(f'K@1, K@2, K@4, K@5, K@8, K@10, K@100, K@1000')
        print(res)
    else:
        raise Exception(f"More than 2 values in 'seens'. len(unique_seens) = {len(unique_seens)}")


if __name__ == '__main__':
    main()
