#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --job-name=hc-hotels_trainval-lrsLRS-lrrLRR-10way-randomresizedcrop228-resnet50-limit4-nor200-tripletloss-marginMAR-el0-nnNN
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
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
        -bs 4 \
        -tf 1 \
        -sf 1 \
        -ep 1500 \
	-is 228 \
        -lrs LRS \
	-lrr LRR \
	-por 5000 \
	-es 20 \
	-cbir \
	-dbb 64 \
	-el 0 \
	-nor 200 \
	-ls 4 \
	-fbw \
	-mtlr \
	-lss trpl \
	-mg MAR \
       	-nn NN \

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
