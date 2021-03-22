import argparse
import os
import shutil

from tqdm import tqdm

train_prefix = 'train'
val_prefix = 'val'
trainval_prefix = 'trainval'
test_prefix = 'test'


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--imagestxt_path', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--target_path', type=str)

    return parser.parse_args()


def restructure(dataset_path, target_path, imagestxt_path):

    with open(imagestxt_path, 'r') as f:
        length = len(f.readlines())

    with tqdm(total = length,  desc='Copying') as t:
        with open(imagestxt_path, 'r') as f:
            # e.g. "1 001.Black_footed_Albatross/Black_Footed_Albatross_0014_89.jpg"
            for line in f.readlines():
                path = line.strip().split(' ')[1]
                fname = path.strip().split('/')[-1]
                classInd = path.strip().split('.')[0]
                ddr_test = None
                ddr_train = None
                ddr_trainval = None
                ddr_val = None

                if int(classInd) > 100:
                    ddr_test = os.path.join(target_path, test_prefix, str(classInd))

                elif int(classInd) <= 100 and int(classInd) > 80:
                    ddr_val = os.path.join(target_path, val_prefix, str(classInd))
                    ddr_trainval = os.path.join(target_path, trainval_prefix, str(classInd))

                else:  # classInd < 80
                    ddr_train = os.path.join(target_path, train_prefix, str(classInd))
                    ddr_trainval = os.path.join(target_path, trainval_prefix, str(classInd))

                for ddr in [ddr_train, ddr_val, ddr_trainval, ddr_test]:
                    if ddr:
                        if not os.path.exists(ddr):
                            os.makedirs(ddr)
                        shutil.copy(os.path.join(dataset_path, path), os.path.join(ddr, fname))
                t.update()


def main():
    args = get_args()
    restructure(args.dataset_path, args.target_path, args.imagestxt_path)


if __name__ == "__main__":
    main()
