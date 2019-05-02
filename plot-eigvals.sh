#!/bin/bash

for K in 4 8 16 32; do
	OUT=intermediates/knn-$K
	mkdir -p "$OUT"
	echo "** invoking graphs.py"
	./graphs.py --knn $K --savedir "$OUT"
	echo "** invoking cluster.py"
	./cluster.py --knn $K --savedir "$OUT" --eigplot --graph "$OUT/sparse-graph.npy"
done
