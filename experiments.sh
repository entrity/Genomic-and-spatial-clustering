#!/bin/bash

function run () {
	echo -e "...\tPCA\tKM\tK\tDIM"
	echo -e "RUN\t$PCA\t$KM\t$K\t$DIM\t$OUT"
	mkdir -p "$OUT"
	./cluster.py --knn $K --km $KM --dim $DIM --scatterplot --savedir "$OUT" --graph "$OUT/sparse-graph.npy" --gen-npy "$GEN_NPY"
	./evaluation.py --graph "$OUT/sparse-graph.npy" --clusters "$OUT/kmeans-dim-$DIM-k-$KM.pkl" --gen-npy "$GEN_NPY"
}

rm log.txt

# PCA vs no PCA
for PCA in 0 50; do
	GEN_NPY="data/transcriptomes-$PCA.npy"
	./preprocess.py --pca $PCA --gen-npy "$GEN_NPY"
	# Number of clusters
	for KM in 12 24; do
		# Order of k-nearest-neighbours graph
		for K in 4 8 16 32; do
			# Number of eigenvectors to use
			for DIM in 512 4 256 8 0 128 64 32; do
				OUT=intermediates/no-spatial-knn-$K-pca-$PCA
				run --no-spatial | tee -a "log.txt"
				OUT=intermediates/knn-$K-pca-$PCA
				run | tee -a "log.txt"
			done
		done
	done
done
