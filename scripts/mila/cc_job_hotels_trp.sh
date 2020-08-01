#!/bin/bash
#SBATCH --job-name=hc-hotels_trainval-difficulteval-lrs3e-3-lrr1e-6-10way-randomresizedcrop300-resnet50-k@nsall-limit4-nor200-tripletloss-margin10-el0
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=46G
#SBATCH --time=2-0
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aarash.feizi@mail.mcgill.ca

conda activate conda-general

echo "running"


python3 train.py -cuda \
	-dsp /network/datasets/hotels50K.var/ \
	-spp ~/projects/ \
	-df hotels50K_extract \
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
	-mtlr \
	-lss trpl \
	-mg 10 \

