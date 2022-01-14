import argparse
import os
import pickle
import time

import faiss
import h5py
import numpy as np
import timm
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import dataset_loaders
import proxy_anchor_models as pa
import softtriple_models as st
import sup_contrastive_models as sc
import pnpp_models as pnpp
import htv2_models as htv2

# on hlr:
# python evaluation.py -chk ../SupContrast/save/SupCon/hotels_models/SupCon_hotels_resnet50_lr_0.01_decay_0.0001_bsz_32_temp_0.1_trial_0_cosine/last.pth -name SupCon_hotels_resnet50_lr_0.01_decay_0.0001_bsz_32_temp_0.1_trial_0_cosine/ --kset 1 2 4 8 10 100 1000 --model_type resnet50 -d hotels -dr ../../datasets/ --baseline supcontrastive --gpu_ids 6

dataset_choices = ['cars', 'cub', 'hotels']
BASELINE_MODELS = ['ours', 'softtriple', 'proxy-anchor', 'supcontrastive', 'proxyncapp', 'htv2', 'resnet50']

DATASET_SIZES = {'cars': {'test': 8131},
                 'cub': {'test': 5924},
                 'hotels': {'val1_small': 3060,
                            'val2_small': 2397,
                            'val3_small': 2207,
                            'val4_small': 2348}}

# softtriplet loss code
def evaluate_recall_at_k(X, Y, Kset, gpu=False, k=5, metric='cosine', dist_matrix=None, ):

    if X.dtype != np.float32:
        print(f'X type as not np.float32! was {X.dtype}')
        X = X.astype(np.float32)

    num = X.shape[0]
    classN = np.max(Y) + 1
    kmax = min(np.max(Kset), num)
    print(f'kmax = {kmax}')
    recallK = np.zeros(len(Kset))

    start = time.time()
    print(f'**** Evaluation, calculating rank dist: {metric}')

    if dist_matrix is not None:
        num = Y.shape[0]
        minval = np.min(dist_matrix) - 1.
        dist_matrix -= np.diag(np.diag(dist_matrix))
        dist_matrix += np.diag(np.ones(num) * minval)
        indices = (-dist_matrix).argsort()[:, :-1]

    else:
        if not X.flags['C_CONTIGUOUS']:
            X = np.ascontiguousarray(X)
        distances, indices, self_distance = get_faiss_knn(X, k=int(kmax), gpu=gpu, metric=metric)

    print(f'**** Evaluation, calculating dist rank DONE. Took {time.time() - start}s')

    YNN = Y[indices]

    for i in range(0, len(Kset)):
        pos = 0.
        for j in range(0, num):
            if Y[j] in YNN[j, :Kset[i]]:
                pos += 1.

        recallK[i] = pos / num
    return recallK

def evaluate_roc(X, Y, n=0):

    if n == 0:
        n = X.shape[0]

    pred_matrix = (cosine_similarity(X) + 1) / 2



    assert np.max(pred_matrix) <= (1 + 1e-5)
    assert np.min(pred_matrix) >= (0 - 1e-5)

    pred_matrix = np.clip(pred_matrix, a_min=0, a_max=1)

    labels, counts = np.unique(Y, return_counts=True)
    labels = labels[counts > 1]

    idxs = np.array([i for i in range(len(Y))], dtype=int)

    true_labels = []
    pred_values = []

    with tqdm(total=n, desc='Calc ROC_AUC') as t:
        for i in range(n):

            anch_lbl = np.random.choice(labels, size=1)

            anch_idx = np.random.choice(idxs[Y == anch_lbl], size=1)
            pos_idx = np.random.choice(idxs[Y == anch_lbl], size=1)
            while pos_idx == anch_idx:
                pos_idx = np.random.choice(idxs[Y == anch_lbl], size=1)

            neg_idx = np.random.choice(idxs[Y != anch_lbl], size=1)

            true_labels.append(1)
            pred_values.append(pred_matrix[anch_idx, pos_idx])

            true_labels.append(0)
            pred_values.append(pred_matrix[anch_idx, neg_idx])

            t.update()

    roc_auc = roc_auc_score(true_labels, pred_values)

    return roc_auc

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

def get_features_and_labels(args, model, loader):
    features = []
    labels = []

    with tqdm(total=len(loader), desc='Getting features...') as t:
        for idx, batch in enumerate(loader):
            img, lbl = batch
            if args.cuda:
                f = model(img.cuda())
            else:
                f = model(img)

            if args.baseline == 'softtriple':
                f = F.normalize(f, p=2, dim=1)

            features.append(f.cpu().detach().numpy())
            labels.append(lbl)

            t.update()

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

def proxyanchor_load_model_resnet50(save_path, args):
    if args.cuda:
        checkpoint = torch.load(save_path, map_location=torch.device(0))
    else:
        checkpoint = torch.load(save_path, map_location=torch.device('cpu'))

    net = pa.Resnet50(embedding_size=args.sz_embedding,
                                      pretrained=True,
                                      is_norm=1,
                                      bn_freeze=1)


    net.load_state_dict(checkpoint['model_state_dict'])

    if args.cuda:
        net = net.cuda()

    return net

def supcontrastive_load_model_resnet50(save_path, args):
    if args.cuda:
        checkpoint = torch.load(save_path, map_location=torch.device(0))
    else:
        checkpoint = torch.load(save_path, map_location=torch.device('cpu'))

    net = sc.resnet50()

    new_checkpoint = {}
    for key, value in checkpoint['model'].items():
        if key.startswith('encoder'):
            new_checkpoint[key[8:]] = value
        elif key.startswith('head'):
            pass

    net.load_state_dict(new_checkpoint)

    if args.cuda:
        net = net.cuda()

    return net

def softtriple_load_model_resnet50(save_path, args):
    if args.cuda:
        checkpoint = torch.load(save_path, map_location=torch.device(0))
    else:
        checkpoint = torch.load(save_path, map_location=torch.device('cpu'))

    net = timm.create_model('resnet50', num_classes=args.sz_embedding)

    net.load_state_dict(checkpoint)

    if args.cuda:
        net = net.cuda()

    return net

def proxyncapp_load_model_resnet50(save_path, args):
    if args.cuda:
        checkpoint = torch.load(save_path, map_location=torch.device(0))
    else:
        checkpoint = torch.load(save_path, map_location=torch.device('cpu'))


    net = pnpp.get_model(args.sz_embedding)

    if args.trained_with_mltp_gpu:
        net = torch.nn.DataParallel(net)

    net.load_state_dict(checkpoint)

    if args.cuda:
        net = net.cuda()

    return net

def htv2_load_model_resnet50(save_path, args):
    if args.cuda:
        checkpoint = torch.load(save_path, map_location=torch.device(0))
    else:
        checkpoint = torch.load(save_path, map_location=torch.device('cpu'))

    net = htv2.get_top_module(args)

    if args.trained_with_mltp_gpu:
        net = torch.nn.DataParallel(net)

    net.load_state_dict(checkpoint['model_state_dict'])

    if args.cuda:
        net = net.cuda()

    return net

def softtriple_load_model_inception(save_path, args):
    if args.cuda:
        checkpoint = torch.load(save_path, map_location=torch.device(0))
    else:
        checkpoint = torch.load(save_path, map_location=torch.device('cpu'))

    net = st.bninception(args.sz_embedding)

    net.load_state_dict(checkpoint)

    if args.cuda:
        net = net.cuda()

    return net


def resnet_load_model(save_path, args):
    # if args.cuda:
    #     checkpoint = torch.load(save_path, map_location=torch.device(0))
    # else:
    #     checkpoint = torch.load(save_path, map_location=torch.device('cpu'))

    net = timm.create_model('resnet50', pretrained=True, num_classes=0)

    if args.cuda:
        net = net.cuda()

    return net


def pca(features, sz_embedding):
    pca_model = PCA(n_components=sz_embedding)

    print(f'Performing PCA to reduce dim from {features.shape[1]} to {sz_embedding}')
    new_features = pca_model.fit_transform(features)

    return new_features


def check(args, all_data):
    val_keys = DATASET_SIZES[args.dataset].keys()
    for provided_data, val_type in zip(all_data, val_keys):
        if provided_data[0].shape[0] != DATASET_SIZES[args.dataset][val_type]:
            print(f'Val type {val_type} should be {DATASET_SIZES[args.dataset][val_type]} images, but is {provided_data[0].shape[0]}')
            return False
    print(f'All sizes for {val_keys} were checked and are correct')
    return True



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cuda', '--cuda', default=False, action='store_true')
    parser.add_argument('-trained_with_mltp_gpu', '--trained_with_mltp_gpu', default=False, action='store_true')

    parser.add_argument('-gpu', '--gpu_ids', default='', help="gpu ids used to train")  # before: default="0,1,2,3"


    parser.add_argument('-X', '--X', nargs='+', default=[],
                        help="Different features for datasets (order important)")
    # parser.add_argument('-X_desc', '--X_desc', nargs='+', default=[],
    #                     help="Different features desc for datasets (order important)") # for h5 or npz files

    parser.add_argument('-Y', '--Y', nargs='+', default=[],
                        help="Different labels for datasets (order important)")
    # parser.add_argument('-Y_desc', '--Y_desc', nargs='+', default=[],
    #                     help="Different labels desc for datasets (order important)")  # for h5 or npz files



    parser.add_argument('-emb', '--sz_embedding', default=512, type=int)
    parser.add_argument('-b', '--sz_batch', default=32, type=int)
    parser.add_argument('-w', '--nb_workers', default=10, type=int)

    parser.add_argument('-d', '--dataset', default=None, choices=dataset_choices)
    parser.add_argument('-dr', '--data_root', default='../hotels')
    parser.add_argument('--baseline', default='proxy-anchor', choices=BASELINE_MODELS)
    parser.add_argument('--model_type', default='resnet50', choices=['bninception', 'resnet50'])

    parser.add_argument('--pca_to_dim', action='store_true')


    parser.add_argument('-chk', '--checkpoint', default=None, help='Path to checkpoint')
    parser.add_argument('--kset', nargs='+', default=[1, 2, 4, 8, 16, 32, 100])
    parser.add_argument('--roc_n', default=0, type=int)

    parser.add_argument('-elp', '--eval_log_path', default='./eval_logs')
    parser.add_argument('-name', '--name', default=None, type=str)

    parser.add_argument('--metric', default='cosine', choices=['cosine', 'euclidean'])

    args = parser.parse_args()

    all_data = []

    kset = [int(k) for k in args.kset]

    if args.name is None:
        raise Exception('Provide --name')

    # provide model and extract embeddings here
    if len(args.X) == 0:

        if args.gpu_ids != '':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

        eval_datasets = []
        if args.dataset == 'hotels':
            for i in range(1, 5):
                eval_datasets.append(dataset_loaders.load(
                name=args.dataset,
                root=args.data_root,
                transform=dataset_loaders.utils.make_transform(
                    is_train=False,
                    ),
                valset=i))
        else:
            eval_datasets = [dataset_loaders.load(
                name=args.dataset,
                root=args.data_root,
                transform=dataset_loaders.utils.make_transform(
                    is_train=False,
                    ))]

        if args.baseline == 'proxy-anchor':
            net = proxyanchor_load_model_resnet50(args.checkpoint, args)
        elif args.baseline == 'supcontrastive':
            net = supcontrastive_load_model_resnet50(args.checkpoint, args)
        elif args.baseline == 'softtriple':
            if args.model_type == 'resnet50':
                net = softtriple_load_model_resnet50(args.checkpoint, args)
            elif args.model_type == 'bninception':
                net = softtriple_load_model_inception(args.checkpoint, args)
        elif args.baseline == 'resnet50':
            net = resnet_load_model(args.checkpoint, args)
        elif args.baseline == 'proxyncapp':
            net = proxyncapp_load_model_resnet50(args.checkpoint, args)
        elif args.baseline == 'htv2':
            net = htv2_load_model_resnet50(args.checkpoint, args)
        eval_ldrs = []
        for dtset in eval_datasets:
            eval_ldrs.append(torch.utils.data.DataLoader(
                dtset,
                batch_size=args.sz_batch,
                shuffle=False,
                num_workers=args.nb_workers,
                drop_last=False,
                pin_memory=True
            ))


        for ldr in eval_ldrs:
            features, labels = get_features_and_labels(args, net, ldr)
            all_data.append((features, labels))
    else: # X and Y should be provided
        for idx, (x, y) in enumerate(zip(args.X, args.Y)):
            if x.endswith('.pkl'):
                with open(x, 'rb') as f:
                    features = pickle.load(f)
            elif x.endswith('.npz'):  # tood
                features = np.load(x)
            elif x.endswith('.h5'):
                with h5py.File(x, 'r') as hf:
                    features = hf['data'][:]
            else:
                raise Exception(f'{x} data format not supported')

            if y.endswith('.pkl'):
                with open(y, 'rb') as f:
                    labels = pickle.load(f)
            elif y.endswith('.npz'): # tood
                labels = np.load(y)
            elif y.endswith('.h5'):
                with h5py.File(y, 'r') as hf:
                    labels = hf['data'][:]
            else:
                raise Exception(f'{y} data format not supported')

            if torch.is_tensor(features):
                features = features.cpu().numpy()

            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()

            all_data.append((features, labels))

    if check(args, all_data):
        print('Assertion completed!')

    results = f'{args.dataset}\n'
    for idx, (features, labels) in enumerate(all_data, 1):

        if features.shape[1] != args.sz_embedding:
            if args.pca_to_dim:
                features = pca(features, args.sz_embedding)
            else:
                raise Exception(
                    f'--pca_to_dim is set to False and feature dim {features.shape[1]} not equal to expected dim {args.sz_embedding}')

        print(f'{idx}: Calc Recall at {kset}')
        rec = evaluate_recall_at_k(features, labels, Kset=kset, metric=args.metric)
        print(kset)
        print(rec)
        results += f'{idx}: Calc Recall at {kset}' + '\n' + str(kset) + '\n' + str(rec) + '\n'

        print('*' * 10)
        print(f'{idx}: Calc AUC_ROC')
        auc = evaluate_roc(features, labels, n=args.roc_n)
        print(f'{idx}: AUC_ROC:', auc)
        results += f'\n\n{idx}: AUC_ROC: {auc}\n\n'
        results += '*' * 20

    with open(os.path.join(args.eval_log_path, args.name + ".txt"), 'w') as f:
        f.write(results)

if __name__ == '__main__':
    main()
