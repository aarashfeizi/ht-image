import argparse
import os

import pandas as pd
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--path', default='')
parser.add_argument('-sp', '--splits_path', default='')
parser.add_argument('-rf', '--resized_folder', default='')

args = parser.parse_args()

path = args.path
splits_path = args.splits_path
resized_folder = args.resized_folder

all = []

all.extend(list(pd.read_csv(os.path.join(splits_path, 'hotels_test_seen.csv')).image))
all.extend(list(pd.read_csv(os.path.join(splits_path, 'hotels_test_unseen.csv')).image))
all.extend(list(pd.read_csv(os.path.join(splits_path, 'hotels_val_seen.csv')).image))
all.extend(list(pd.read_csv(os.path.join(splits_path, 'hotels_val_unseen.csv')).image))
all.extend(list(pd.read_csv(os.path.join(splits_path, 'hotels_train.csv')).image))
errors = ''

for i, p in enumerate(all):
    if i % 10 == 0:
        print((i + 1) / len(all))
    try:
        img = Image.open(os.path.join(path, p))
    except:
        errors += f'ERROR IN READING {p}, index: {i}\n'
        continue
    shape = img.size
    p_dirs = p[:p.rfind('/')]
    if not os.path.exists(os.path.join(resized_folder, p_dirs)):
        os.makedirs(os.path.join(resized_folder, p_dirs))
    ratio = 224 / shape[0]
    img.resize((224, int(ratio * shape[1])), Image.ANTIALIAS).save(os.path.join(resized_folder, p))

if errors == '':
    print('No errors!')
else:
    print(errors)
