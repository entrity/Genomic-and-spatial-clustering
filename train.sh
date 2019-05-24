#!/bin/bash

. shared.sh

echo $(( 50 * ( 32 + 1 ) ))

LR=(NONE 2e-2 2e-2 2e-3)
EP=(NONE 2500 2500 5000)

function run ()
{
	STAGE="$1"
	OTHER_ARGS="${@:2}"
	PREV_STAGE=$(( ${STAGE} - 1 ))
	python trainer.py \
		--train artefacts/knn-feats-pca_50-knn_32-train.npy \
		--test artefacts/knn-feats-pca_50-knn_32-test.npy \
		--log_path  nn-log/stage-${STAGE}.log \
		--load_path models/stage-${PREV_STAGE}.pth \
		--save_path models/stage-${STAGE}.pth \
		--print_every 0 \
		--test_every 0 \
		--arch "$ARCH" \
		--lr ${LR[ $STAGE ]} \
		--ep ${EP[ $STAGE ]} \
		-n ${STAGE} \
		${OTHER_ARGS[@]}
}

run 1
run 2
run 3
