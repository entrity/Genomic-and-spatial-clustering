#!/bin/bash

for K in 4 8 16 32; do
	OUT=intermediates/knn-$K
	mkdir -p "$OUT"
	./graphs.py -k $K -s "$OUT"

done
