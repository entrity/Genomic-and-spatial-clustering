#!/bin/bash

function run () {
	echo -e "RUN 1\t$K\t$OUT\t$KM"
	mkdir -p "$OUT"
	./graphs.py --knn $K --savedir "$OUT" "${@}"
	./cluster.py --knn $K --km $KM --scatterplot --savedir "$OUT" --graph "$OUT/sparse-graph.npy"
	./evaluation.py --graph "$OUT/sparse-graph.npy" --clusters "$OUT/kmeans-dim-None-k-12.pkl"
	for DIM in 512 4 256 8 128 64 32; do
		echo -e "RUN 2\t$DIM\t$K\t$OUT\t$KM"
		./cluster.py --knn $K --km $KM --dim $DIM --scatterplot --savedir "$OUT" --graph "$OUT/sparse-graph.npy"
		./evaluation.py --graph "$OUT/sparse-graph.npy" --clusters "$OUT/kmeans-dim-$DIM-k-$KM.pkl"
	done
}

rm log.txt

for KM in 12 24; do
for K in 4 8 16 32; do
	OUT=intermediates/no-spatial-knn-$K
	run --no-spatial | tee -a "log.txt"
	OUT=intermediates/knn-$K
	run | tee -a "log.txt"
done
done
