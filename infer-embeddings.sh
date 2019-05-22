#!/bin/bash

python inference.py \
	--data artefacts/knn-feats-pca_50-knn_16.npy \
	--load_path models/stage-2.pth \
	--save_path artefacts/nn-embeddings.npy \
	--arch '850 250 50'
