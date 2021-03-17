
import shutil
import os
import argparse
import pandas as pd
from tqdm import tqdm

SPLITS = ['hotels_train',
          'hotels_val_seen',
          'hotels_val_unseen',
          'hotels_test_seen',
          'hotels_test_unseen']

trainPrefix = 'train/'
test_seen_Prefix = 'test_seen/'
test_umseen_Prefix = 'test_unseen/'
val_seen_Prefix = 'val_seen/'
val_unseen_Prefix = 'val_unseen/'

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--split_path', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--target_path', type=str)

    return parser.parse_args()

def restructure(dataset_path, target_path, path_csv, prefix):
    images = list(path_csv.image)
    labels = list(path_csv.label)
    with tqdm(total=len(images), desc=f'{prefix}') as t:
        for im, classInd in zip(images, labels):
            fname = im.split('/')[-1]
            ddr = os.path.join(target_path, prefix, str(classInd))
            if not os.path.exists(ddr):
                os.makedirs(ddr)
            shutil.move(os.path.join(dataset_path, im), os.path.join(ddr, fname))
            t.update()


def main():
    args = get_args()

    for prefix in SPLITS:
        print(f'Doing {prefix}')
        p = os.path.join(args.split_path, prefix)
        path_csv = pd.read_csv(p + '.csv')
        restructure(args.dataset_path, args.target_path, path_csv, prefix)

if __name__ == "__main__":
    main()