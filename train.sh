#!/bin/bash

echo $(( 50 * ( 32 + 1 ) ))

LR=(NONE 1e-2 5e-3 2.5e-3)

for STAGE in 1 2 3; do
	PREV_STAGE=$(( ${STAGE} - 1 ))
	python trainer.py \
		--train artefacts/knn-feats-pca_50-knn_16-train.npy \
		--test artefacts/knn-feats-pca_50-knn_16-test.npy \
		--log_path  nn-log/stage-${STAGE}.log \
		--load_path models/stage-${PREV_STAGE}.pth \
		--save_path models/stage-${STAGE}.pth \
		--print_every 0 \
		--test_every 0 \
		--arch '850 250 50' \
		--lr ${LR[ $STAGE ]} \
		--ep 10000 \
		-n ${STAGE}
done
