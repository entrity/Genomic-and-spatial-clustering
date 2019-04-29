#!/usr/bin/env python3

from scipy.cluster.vq import whiten

import argparse
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
def preprocess(ndarray, savepath, do_whiten):
	if do_whiten:
		ndarray = whiten(ndarray)
		print('Whitening for %s...' % savepath)
	np.save(savepath, ndarray)
	print('Saved to %s' % savepath)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	util.add_default_args(parser)
	args = parser.parse_args()
	xy   = load_xy_csv(args.xy_csv)
	preprocess(xy, args.xy_npy, False)
	gens = load_transcriptome_csv(args.gen_csv)
	preprocess(gens, args.gen_npy, True)
