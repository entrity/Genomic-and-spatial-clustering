#!/bin/bash

. shared.sh

python inference.py \
	--data artefacts/knn-feats-pca_50-knn_32.npy \
	--load_path models/stage-3.pth \
	--save_path artefacts/nn-embeddings.npy \
	--arch "$ARCH"
