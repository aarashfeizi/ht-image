#!/bin/bash

read -a negs -p "Negtive per positive: "
read -a mars -p "Margins: "
read -a lrr_list -p "ResNet lr:"
read -a lrs_list -p "Siamese lr:"


for k in "${negs[@]}"; do
	for w in "${mars[@]}"; do
		for lrr in "${lrr_list[@]}"; do
			for lrs in "${lrs_list[@]}"; do
				cp temp_cc_job_hotels_trp.sh cc_hotels.${k}_${w}_${lrr}_${lrs}.sh

				sed -i "s/NN/${k}/" cc_hotels.${k}_${w}_${lrr}_${lrs}.sh

				sed -i "s/MAR/${w}/" cc_hotels.${k}_${w}_${lrr}_${lrs}.sh

				sed -i "s/LRR/${lrr}/" cc_hotels.${k}_${w}_${lrr}_${lrs}.sh

				sed -i "s/LRS/${lrs}/" cc_hotels.${k}_${w}_${lrr}_${lrs}.sh
				#sed -i "s/MD/${4}/" cc_hotels.${k}_${w}.sh
				#if [ ${4} == 'A' ]
				#then
				#	sed -i 's/ADV/-adv/' cc_job.${k}_${w}.sh
				#else
				#	sed -i 's/ ADV//' cc_job.${k}_${w}.sh
			#fi
				sbatch cc_hotels.${k}_${w}_${lrr}_${lrs}.sh
				rm cc_hotels.${k}_${w}_${lrr}_${lrs}.sh
			done
		done
	done
done
