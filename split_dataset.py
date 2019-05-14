import numpy as np
import sys, os, glob

GLOB = sys.argv[1]
knn_feats_files = glob.glob(GLOB)

for fin in knn_feats_files:
	print(fin)
	d = np.load(fin)
	N = len(d)
	partition = round(N * 0.8)
	idxs = np.arange(N)
	np.random.shuffle(idxs)
	train_idxs = idxs[:partition]
	test_idxs  = idxs[partition:]
	train      = d[train_idxs]
	test       = d[test_idxs]
	print('train', train.shape)
	print('test', test.shape)
	print()
	name, ext  = os.path.splitext(fin)
	np.save(name+'-train'+ext, train)
	np.save(name+'-test'+ext,  test)
