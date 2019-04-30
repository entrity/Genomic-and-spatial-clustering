#!/usr/bin/env python3

from scipy.cluster.vq import whiten
from sklearn.decomposition import PCA

import argparse
import sklearn
import numpy as np
import util

# Load x,y-coordinates from a CSV
def load_xy_csv(fpath):
	return np.loadtxt(fpath)
# Load cell ids from a text file
def load_cell_ids_csv(fpath):
	return np.loadtxt(fpath)
# Load class labels from a CSV file
def load_class_labels_csv(fpath):
	return np.loadtxt(fpath)
# Load class labels from a CSV file
def load_transcriptome_csv(fpath):
	return np.loadtxt(fpath)
# Whiten, sort according to cellID, pickle
def preprocess_transcriptome(ndarray, savepath):
	ndarray = whiten(ndarray)
	pca     = PCA(50) # Determined by experiment. Look at eigenvalues.png
	ndarray = pca.fit_transform(ndarray)
	np.save(savepath, ndarray)
	print('Saved to %s' % savepath)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	util.add_default_args(parser)
	args = parser.parse_args()
	xy   = load_xy_csv(args.xy_csv)
	np.save(args.xy_npy, xy)
	gens = load_transcriptome_csv(args.gen_csv)
	preprocess_transcriptome(gens, args.gen_npy)
