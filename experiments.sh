#!/bin/bash

function run () {
	mkdir -p "$OUT"
	./graphs.py --knn $K --savedir "$OUT" "${@}"
	./cluster.py --knn $K --km $KM --scatterplot --savedir "$OUT" --graph "$OUT/sparse-graph.npy"
}

KM=12
for K in 4 8 16 32; do
	OUT=intermediates/no-spatial-knn-$K
	run --no-spatial
	OUT=intermediates/knn-$K
	run
	break
done
