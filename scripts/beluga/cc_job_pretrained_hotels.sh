#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --job-name=hc-hotels_trainval-difficulteval-lrs3e-3-lrr1e-6-10way-randomresizedcrop300-resnet50-k@nsall-limit4-nor200-afterbug-neg1-notrain-ontrainset5
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=46G
#SBATCH --time=1-0
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
        -wr 10 \
	-pim \
	-w 10 \
        -bs 16 \
        -tf 1 \
        -sf 1 \
        -ep 1 \
        -lrs 3e-3 \
	-lrr 1e-6 \
	-dst new \
	-por 5000\
	-es 20 \
	-cbir \
	-dbb 64 \
	-el 0 \
	-katn \
	-ls 4 \
	-nor 200 \
	-mn model-epoch-95-val-acc-0.66625.pt \
	-pmd model-top-dataset_name_hotels-limit_samples_4-number_of_runs_200-feat_extractor_resnet50-extra_layer_0-batch_size_16-lr_siamese_0.003-lr_resnet_1e-06-early_stopping_20-normalize_False-time_2020-07-16_13-47-03-061899 \
	#--no-sampled_results \
	#--no-per_class_results \

	#-mn model-epoch-92-val-acc-0.6325.pt \
	#-pmd model-top-dataset_name_hotels-dataset_split_type_new-aug_False-rotate_0.0-feat_extractor_resnet18-freeze_ext_False-extra_layer_0-batch_size_8-lr_siamese_0.003-lr_resnet_1e-06-normalize_False-time_2020-06-30_04-23-34-955829 \
	#-ls 4 \
	#-nor 500 \
	#-n \
	#-fbw \
	#-el \
	#-tst \
