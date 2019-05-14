#!/bin/bash

echo $(( 50 * ( 32 + 1 ) ))

for STAGE in 1 2 3; do
	PREV_STAGE=$(( ${STAGE} - 1 ))
	python trainer.py \
		--train artefacts/knn-feats-pca_50-knn_32-train.npy \
		--test artefacts/knn-feats-pca_50-knn_32-test.npy \
		--log_path training/stage-${STAGE}.log \
		--load_path training/stage-${PREV_STAGE}.pth \
		--save_path training/state-${STAGE}.pth \
		--arch 850 250 50 \
		-n ${STAGE}
done
