#!/bin/bash

. shared.sh

echo $(( 50 * ( 32 + 1 ) ))

LR=(NONE 2e-2 2e-2 2e-3)
EP=(NONE 50 150 500)

# These files should be something like artefacts/knn-feats-pca_50-knn_32-train.npy
TRAINSET_FILE="${1}"
TESTSET_FILE="${2}"

function run ()
{
	STAGE="$1"
	OTHER_ARGS="${@:2}"
	PREV_STAGE=$(( ${STAGE} - 1 ))
	python trainer.py \
		--train "$TRAINSET_FILE" \
		--test "$TESTSET_FILE" \
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

