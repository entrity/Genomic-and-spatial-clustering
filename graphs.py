#!/usr/bin/env python3

import os
import argparse
import numpy as np
import scipy
import util
from util import tictoc

class BaseGraph(object):
	# @arg A is an adjacency matrix or a path to it
	def __init__(self, A=None):
		self.A = load(A)
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
		if savepath is not None:
			np.save(savepath, self.A)
		# Return
		return self.A
	def sparsified(self, k):
		I = self.A.argsort(axis=1)
		return I[:,-k:]
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

def run(GraphClass):
	parser = argparse.ArgumentParser()
	util.add_default_args(parser)
	args = parser.parse_args()
	xy  = np.load(args.xy_npy)
	gen = np.load(args.gen_npy)
	fc_graph_path = 'data/base-graph-fc.npy'
	k_nn = 5
	g   = GraphClass()
	if os.path.exists(fc_graph_path):
		g.A = load(fc_graph_path)
	else:
		g.fc_graph(xy, gen, fc_graph_path)
	I = g.sparsified(k_nn)
	return xy, gen, g, I

if __name__ == '__main__':
	xy, gen, g, I = run(BaseGraph)

