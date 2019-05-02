#!/usr/bin/env python3

import os
import argparse
import numpy as np
import scipy
import util
from util import tictoc

class BaseGraph(object):
	# @arg A is an adjacency matrix or a path to it
	def __init__(self, A=None, savedir=None):
		self.A = load(A)
		self.savedir = savedir
	# Construct fully-connected adjacency matrix
	def fc_graph(self, xy, transcriptome, savepath=None):
		self.xy = xy # spatial features
		self.transcriptome = transcriptome # genomic features
		assert self.xy.shape[0] == self.transcriptome.shape[0], (self.xy.shape[0], self.transcriptome.shape[0])
		tic = tictoc('Building fully-connected graph...')
		n = self.xy.shape[0]
		distance = np.zeros((n,n)) # opposite of adjacency matrix
		# Add xy distance to top-right triangle
		util.debug('setting spatial distance...')
		for i, xy in enumerate(self.xy):
			if i % 200 == 0: util.debug(i)
			for j in range(i+1, n):
				distance[i,j] += self._xy_dist(self.xy[i,...], self.xy[j,...])
		# Add feature distance to top-right triangle
		util.debug('setting transcriptome distance...')
		for i, gen in enumerate(self.transcriptome):
			if i % 200 == 0: util.debug(i)
			for j in range(i+1, n):
				distance[i,j] += self._transcriptome_dist(self.transcriptome[i,...], self.transcriptome[j,...])
		# Copy to lower-left triangle
		distance += distance.T
		# Create adjacency matrix. (Because I will use k-nearest-neighbours, ranking is all that matters, not relative distance between all points.)
		self.A = distance * -1
		# Remove self-loops (diagonal)
		self.A -= np.diag(np.diag(self.A))
		# Debug
		tictoc('Built fully-connected graph.', tic)
		# Save
		self._save(savepath, 'fc-graph.npy', self.A)
		# Return
		return self.A
	def _save(self, savepath, default_savename, object):
		if savepath is None and self.savedir is None:
			return
		if savepath is None and self.savedir is not None:
			savepath = os.path.join(self.savedir, default_savename)
		util.debug('Saving %s...' % savepath)
		np.save(savepath, object)
	def _sparsified(self, k):
		util.debug('Sparsifying with k-n-n = %d...' % k)
		I = self.A.argsort(axis=1)
		return I[:,-k:]
	def sparsify(self, k):
		nnzs = len(np.nonzero(self.A)[0])
		tic = util.tictoc('Sparsifying matrix A (%d nonzeros)...' % nnzs)
		idxs = self._sparsified(k)
		mask = np.zeros_like(self.A)
		for r, row in enumerate(idxs):
			mask[r, row] = 1
		self.A *= mask
		nnzs = len(np.nonzero(self.A)[0])
		util.tictoc('Sparsified matrix A (%d nonzeros).' % nnzs, tic)
	def _xy_dist(self, a, b):
		assert np.ndim(a) == 1
		assert np.ndim(b) == 1
		assert len(a) == 2
		assert len(b) == 2
		return euclidean_distance(a, b)
	def _transcriptome_dist(self, a, b):
		assert np.ndim(a) == 1
		assert np.ndim(b) == 1
		assert len(a) == 50, len(a) # PCA
		assert len(b) == 50, len(b) # PCA
		return euclidean_distance(a, b)


def euclidean_distance(a, b):
	return np.linalg.norm(a - b)
def cosine_distance(a, b):
	return scipy.spatial.distance.cosine(a, b)

def load(obj):
	if obj is None or isinstance(obj, np.ndarray):
		return obj
	else:
		return np.load(obj)

def run(GraphClass, xy_npy, gen_npy, k_nn, savedir):
	xy  = np.load(xy_npy)
	gen = np.load(gen_npy)
	npy = os.path.join(savedir, 'fc-graph.npy')
	if not os.path.exists(npy): npy = None
	g = GraphClass(npy, savedir)
	if not isinstance(g.A, np.ndarray):
		g.fc_graph(xy, gen)
	g.sparsify(k_nn)
	return xy, gen, g

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	util.add_default_args(parser)
	parser.add_argument('-k', '--k', type=int, help='Number of nearest neighbours to preserve in sparse graph')
	args = parser.parse_args()
	xy, gen, g = run(BaseGraph, args.xy_npy, args.gen_npy, args.k, args.savedir)
