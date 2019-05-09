#!/bin/bash

function fn () {
	grep -v -- --1 | while read f; do
		wc -l $f | cut -d' ' -f1
	done # | awk '{ tt += $1; ct += 1 } END { printf "(end) %f \n", tt/ct }'
}

A=debug/a.txt
B=debug/b.txt

ls debug/*km_12*/cluster-*-idxs.txt | fn > $A
ls debug/*km_24*/cluster-*-idxs.txt | fn > $B

python <<HEREDOC
import numpy as np
import matplotlib.pyplot as plt

def run(fname, k):
	print(fname)
	data = np.loadtxt(fname)
	print('med', np.median(data))
	print('mea', data.mean())
	print('min', data.min())
	print('max', data.max())
	# print(np.unique(data, return_counts=True))
	print()
	plt.hist(data, bins=len(np.unique(data)))
	plt.title("class membership for k = %d" % k)
	plt.show()

run('$A', 12)
run('$B', 24)
HEREDOC
