#!/usr/bin/env python

# import pytorch
import argparse
import numpy as np
import os, pickle
import util, preprocess

DEFAULT_GT_GRAPH_PATH = 'data/gt-graph.npy'

# Return ground-truth graph with A[i,j] in {0,1} for whether i and j are in same cluster
def build_gt_graph(n, cids, lbls):
	A = np.zeros((n,n), np.uint8)
	raise NotImplementedError

def acc(graph):
	# Construct mask for pairs which should have edges
	pos_mask = np.zeros_like(graph, np.uint8)
	# Construct mask for pairs which should not have edges
	neg_mask = np.zeros_like(graph, np.uint8)
	raise NotImplementedError

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	util.add_default_args(parser)
	parser.add_argument('-c', '--clusters', help='Pickle file of kmeans object', default='intermediates/no-spatial-knn-4/kmeans-dim-None-k-12')
	args = parser.parse_args()
	graph = np.load(args.graph)
	if os.path.exists(DEFAULT_GT_GRAPH_PATH):
		gt = np.load(DEFAULT_GT_GRAPH_PATH)
	else:
		assert graph.shape[0] = graph.shape[1], graph.shape
		cids = preprocess.load_cell_ids_csv(args.id_csv).astype(np.int)
		lbls = preprocess.load_class_labels_csv(args.lbl_csv)
		n    = graph.shape[0]
		gt   = build_gt_graph(n, cids, lbls)
		util.debug('Saving %s...' % DEFAULT_GT_GRAPH_PATH)
		np.save(DEFAULT_GT_GRAPH_PATH, gt)
	with open(args.clusters, 'rb') as fin:
		kmeans = pickle.load(fin)
