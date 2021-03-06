#!/usr/bin/env python

# E.g. ./evaluation.py --graph intermediates/no-spatial-knn-4/sparse-graph.npy

import argparse
import numpy as np
import os, pickle
import util, preprocess

DEFAULT_POS_MASK_PATH = 'data/gt-pos.npy'
DEFAULT_NEG_MASK_PATH = 'data/gt-neg.npy'

# Return ground-truth graph with A[i,j] in {0,1} for whether i and j are in same cluster
def build_gt_graphs(n, cids, lbls, savedir=None):
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
		if lbl == -1: continue # -1 is labelled 'NA'
		for idx in idxs:
			pos_mask[idx, idxs] = 1
	pos_mask = pos_mask - np.diag(np.diag(pos_mask))
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
	neg_mask = neg_mask - np.diag(np.diag(neg_mask))
	# util.debug('Saving %s...' % DEFAULT_NEG_MASK_PATH)
	# np.save(DEFAULT_NEG_MASK_PATH, neg_mask)
	# Return
	return pos_mask, neg_mask

# Debugging function used to count nonzero entries in ground-truth graphs
def comb(pos_mask, neg_mask, r):
	poss = pos_mask[r].nonzero()[0]
	negs = neg_mask[r].nonzero()[0]
	toge = [x for x in poss] + [x for x in negs]
	return sorted(toge)

# Let `lbls` be `kmeans.labels_` from a kmeans object
# Return a binary adjacency matrix which indicates whether given nodes are assigned to the same cluster
def graph_from_clusters(lbls):
	lbl2idxs = {}
	for i, lbl in enumerate(lbls):
		if lbl not in lbl2idxs: lbl2idxs[lbl] = []
		lbl2idxs[lbl].append(i)
	n = len(lbls)
	graph = np.zeros((n,n), np.uint8)
	for lbl in lbl2idxs:
		if lbl == -1: continue
		idxs = lbl2idxs[lbl]
		for idx in idxs:
			graph[idx, idxs] = 1
	graph = graph - np.diag(np.diag(graph))
	return graph

# Let all three graphs be binary
def acc(bin_graph, pos_mask, neg_mask):
	positives = pos_mask * bin_graph
	pos_ct = np.count_nonzero(positives)
	negatives = neg_mask * bin_graph
	neg_ct = np.count_nonzero(negatives)
	return pos_ct, neg_ct, pos_ct - neg_ct

def get_ground_truth(graph, id_csv, lbl_csv, savedir=None):
	if os.path.exists(DEFAULT_POS_MASK_PATH) and os.path.exists(DEFAULT_NEG_MASK_PATH):
		pos_mask = np.load(DEFAULT_GT_GRAPH_PATH)
		neg_mask = np.load(DEFAULT_NEG_MASK_PATH)
	else:
		assert graph.shape[0] == graph.shape[1], graph.shape
		cids = preprocess.load_cell_ids_csv(id_csv).astype(np.int)
		np.savetxt('debug/decimal_keptCellIDs.txt', cids, fmt='%d')
		lbls = preprocess.load_class_labels_csv(lbl_csv)
		n    = graph.shape[0]
		pos_mask, neg_mask = build_gt_graphs(n, cids, lbls, savedir)
	return pos_mask, neg_mask

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	util.add_default_args(parser)
	parser.add_argument('-c', '--clusters', help='Pickle file of kmeans object', default='intermediates/no-spatial-knn-4/kmeans-dim-None-k-12')
	args = parser.parse_args()
	graph = np.load(args.sparse)
	pos_mask, neg_mask = get_ground_truth(graph, args.id_csv, args.lbl_csv)
	with open(args.clusters, 'rb') as fin:
		kmeans = pickle.load(fin)
	bin_graph = graph_from_clusters(kmeans.labels_)
	pos_ct, neg_ct, score = acc(bin_graph, pos_mask, neg_mask)
	kl_div_err = my_loss.compute_kl_div_loss().item()
	print('pos\tneg\tscore\t(SCORING)\n%d\t%d\t%f\t%d\t%d\t%d' % (pos_ct, neg_ct, score, np.count_nonzero(bin_graph), np.count_nonzero(pos_mask), np.count_nonzero(neg_mask)))
	#import pdb; pdb.set_trace()
