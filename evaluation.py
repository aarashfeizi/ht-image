import os

import numpy as np
import time
import faiss
import torchvision.models
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import argparse
import pickle
import torch

import dataset_loaders

dataset_choices = ['cars', 'cub', 'hotels']

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

    sim_matrix = cosine_similarity(X)
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
            pred_values.append(sim_matrix[anch_idx, pos_idx])

            true_labels.append(0)
            pred_values.append(sim_matrix[anch_idx, neg_idx])

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

    for idx, batch in enumerate(loader):
        img, lbl = batch
        if args.cuda:
            f = model(img.cuda())
        else:
            f = model(img)

        features.append(f.cpu().detach().numpy())
        labels.append(lbl)

    return features, labels

def load_model_resnet50(save_path, args):
    if args.cuda:
        checkpoint = torch.load(save_path)
    else:
        checkpoint = torch.load(save_path, map_location=torch.device('cpu'))

    net = torchvision.models.resnet50(False)

    net.load_state_dict(checkpoint)

    if args.cuda:
        net = net.cuda()

    return net

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cuda', '--cuda', default=False, action='store_true')
    parser.add_argument('-gpu', '--gpu_ids', default='', help="gpu ids used to train")  # before: default="0,1,2,3"
    parser.add_argument('-X', '--X', default=None, type=str)  # 'features.pkl'
    parser.add_argument('-Y', '--Y', default=None, type=str)  # 'labels.pkl'
    parser.add_argument('-d', '--dataset', default='hotels', choices=dataset_choices)
    parser.add_argument('-dr', '--data_root', default='../hotels')
    parser.add_argument('-chk', '--checkpoint', default=None, help='Path to checkpoint')
    parser.add_argument('--kset', nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--roc_n', default=0, type=int)
    parser.add_argument('--metric', default='cosine', choices=['cosine', 'euclidean'])

    args = parser.parse_args()

    all_data = []
    if args.dataset is not None:

        if args.gpu_ids != '':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

        eval_ldrs = []
        if args.dataset == 'hotels':
            for i in range(1, 5):
                eval_ldrs.append(dataset_loaders.load(
                name=args.dataset,
                root=args.data_root,
                transform=dataset_loaders.utils.make_transform(
                    is_train=False,
                    ),
                valset=i))
        else:
            eval_ldrs = [dataset_loaders.load(
                name=args.dataset,
                root=args.data_root,
                transform=dataset_loaders.utils.make_transform(
                    is_train=False,
                    ))]

        net = load_model_resnet50(args.checkpoint, args)
        for ldr in eval_ldrs:
            features, labels = get_features_and_labels(args, net, ldr)
            all_data.append((features, labels))

    else: # X and Y should be provided
        with open(args.X, 'rb') as f:
            features = pickle.load(f)
            if torch.is_tensor(features):
                features = features.cpu().numpy()

        with open(args.Y, 'rb') as f:
            labels = pickle.load(f)
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
        all_data.append((features, labels))

    for idx, (features, labels) in enumerate(all_data, 1):
        print(f'{idx}: Calc Recall at {args.kset}')
        rec = evaluate_recall_at_k(features, labels, Kset=args.kset, metric=args.metric)
        print(args.kset)
        print(rec)

        print('*' * 10)
        print(f'{idx}: Calc AUC_ROC')
        auc = evaluate_roc(features, labels, n=args.roc_n)
        print(f'{idx}: AUC_ROC:', auc)

if __name__ == '__main__':
    main()
