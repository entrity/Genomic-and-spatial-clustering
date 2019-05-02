#!/usr/bin/env python

# E.g. ./evaluation.py --graph intermediates/no-spatial-knn-4/sparse-graph.npy

import argparse
import numpy as np
import os, pickle
import util, preprocess

DEFAULT_POS_MASK_PATH = 'data/gt-pos.npy'
DEFAULT_NEG_MASK_PATH = 'data/gt-neg.npy'

# Return ground-truth graph with A[i,j] in {0,1} for whether i and j are in same cluster
def build_gt_graphs(n, cids, lbls):
	# Remap arbitrary ids in lbls (using cids) to indices 0 through n
	lbl2idxs = {} # Cluster id to list of cell indices
	cid2idx  = {} # Cell id to cell index
	cid2lbl  = {} # Cell id to cluster id
	for i, row in enumerate(lbls):
		cid, lbl, text_lbl = row
		assert cid not in cid2lbl
		cid2lbl[cid] = lbl
		if lbl not in lbl2idxs:
			lbl2idxs[lbl] = []
	for idx, cid in enumerate(cids):
		assert cid not in cid2idx
		cid2idx[cid] = idx
		if cid not in cid2lbl: continue # Not all of the cells have labels
		lbl = cid2lbl[cid]
		lbl2idxs[lbl].append(idx)
	# Build adjacency matrix (positive connections)
	pos_mask = np.zeros((n,n), np.uint8)
	for lbl in lbl2idxs:
		idxs = lbl2idxs[lbl]
		np.savetxt('debug/cluster-%d-idxs.txt' % lbl, idxs, fmt='%d')
		np.savetxt('debug/cluster-%d-cids.txt' % lbl, cids[idxs], fmt='%d')
		if lbl == -1: continue # -1 is labelled 'NA'
		for idx in idxs:
			pos_mask[idx, idxs] = 1
	# util.debug('Saving %s...' % DEFAULT_POS_MASK_PATH)
	# np.save(DEFAULT_POS_MASK_PATH, pos_mask)
	# Build negative mask (1's for pairs which belong to different clusters)
	neg_mask = np.zeros((n,n), np.uint8)
	for lbl_i in lbl2idxs:
		for lbl_j in lbl2idxs:
			if lbl_i == -1: continue
			if lbl_j == -1: continue
			if lbl_i == lbl_j: continue
			idxs_i = lbl2idxs[lbl_i]
			idxs_j = lbl2idxs[lbl_j]
			for idx_i in idxs_i:
				neg_mask[idx_i, idxs_j] = 1
	# util.debug('Saving %s...' % DEFAULT_NEG_MASK_PATH)
	# np.save(DEFAULT_NEG_MASK_PATH, neg_mask)
	# Return
	return pos_mask, neg_mask

def comb(pos_mask, neg_mask, r):
	poss = pos_mask[r].nonzero()[0]
	negs = neg_mask[r].nonzero()[0]
	toge = [x for x in poss] + [x for x in negs]
	return sorted(toge)


def acc(graph, gt_graph):
	# Construct mask for pairs which should have edges
	pos_mask = gt_graph
	# Construct mask for pairs which should not have edges
	neg_mask = np.zeros_like(graph, np.uint8)
	raise NotImplementedError

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	util.add_default_args(parser)
	parser.add_argument('-c', '--clusters', help='Pickle file of kmeans object', default='intermediates/no-spatial-knn-4/kmeans-dim-None-k-12')
	args = parser.parse_args()
	graph = np.load(args.graph)
	if os.path.exists(DEFAULT_POS_MASK_PATH) and os.path.exists(DEFAULT_NEG_MASK_PATH):
		pos_mask = np.load(DEFAULT_GT_GRAPH_PATH)
		neg_mask = np.load(DEFAULT_NEG_MASK_PATH)
	else:
		assert graph.shape[0] == graph.shape[1], graph.shape
		cids = preprocess.load_cell_ids_csv(args.id_csv).astype(np.int)
		np.savetxt('debug/decimal_keptCellIDs.txt', cids, fmt='%d')
		lbls = preprocess.load_class_labels_csv(args.lbl_csv)
		n    = graph.shape[0]
		pos_mask, neg_mask = build_gt_graphs(n, cids, lbls)
	import IPython; IPython.embed(); # to do: delete
	with open(args.clusters, 'rb') as fin:
		kmeans = pickle.load(fin)
