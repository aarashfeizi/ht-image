#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --job-name=hc-hotels_trainval-difficulteval-lrs3e-3-lrr1e-6-10way-centercrop300-resnet50-k@nsall-limit4-nor200-afterbug-neg5
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=46G
#SBATCH --time=2-0
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aarash.feizi@mail.mcgill.ca

source /home/aarash/projects/def-rrabba/aarash/venv-general/bin/activate

echo "running"

python3 train.py -cuda \
	-dsp /home/aarash/scratch/aarash/ \
	-dsn hotels \
	-fe resnet50 \
        -sp savedmodels \
        -gpu 0 \
        -wr 18 \
	-pim \
	-w 10 \
        -bs 16 \
        -tf 1 \
        -sf 1 \
        -ep 1500 \
        -lrs 3e-3 \
	-lrr 1e-6 \
	-dst new \
	-por 5000 \
	-es 20 \
	-cbir \
	-dbb 64 \
	-el 0 \
	-nor 200 \
	-ls 4 \
	-fbw \
	-nn 5 \

	#-katn \
	#-mn model-epoch-69-val-acc-0.645.pt \
	#-pmd model-top-dataset_name_hotels-dataset_split_type_new-aug_False-rotate_0.0-feat_extractor_resnet50-freeze_ext_False-extra_layer_0-batch_size_16-lr_siamese_0.001-lr_resnet_1e-06-normalize_False-time_2020-07-04_10-05-29-353358 \

	#-n \
	#-fbw \
	#-el \
	#-tst \
	#-el \
	#-fr \
	#-el \
