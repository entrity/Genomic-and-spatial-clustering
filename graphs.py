#!/usr/bin/env python3

import argparse
import numpy as np
import util
from util import tictoc

class Graph(object):
	def __init__(self, xy, transcriptome):
		self.xy = xy # spatial features
		self.transcriptome = transcriptome # genomic features
		self._fc_graph()
	# Construct fully-connected adjacency matrix
	def _fc_graph(self):
		assert self.xy.shape[0] == self.transcriptome.shape[0], (self.xy.shape[0], self.transcriptome.shape[0])
		tic = tictoc('Building fully-connected graph...')
		n = self.xy.shape[0]
		distance = np.zeros((n,n)) # opposite of adjacency matrix
		# Add xy distance to top-right triangle
		for i, xy in enumerate(self.xy):
			for j in range(i+1, n):
				distance[i,j] += self._xy_dist(self.xy[i,...], self.xy[j,...])
		# Add feature distance to top-right triangle
		for i, gen in enumerate(self.transcriptome):
			for j in range(i+1, n):
				distance[i,j] += self._transcriptome_dist(self.transcriptome[i,...], self.transcriptome[j,...])
		# Copy to lower-left triangle
		distance += distance.T
		# Create adjacency matrix. (Because I will use k-nearest-neighbours, ranking is all that matters, not relative distance between all points.)
		A = distance * -1
		# Remove self-loops (diagonal)
		A -= np.diag(np.diag(A))
		# Debug
		tictoc('Built fully-connected graph.', tic)
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

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	util.add_default_args(parser)
	args = parser.parse_args()
	xy  = np.load(args.xy_npy)
	gen = np.load(args.gen_npy)
	g   = Graph(xy, gen)
