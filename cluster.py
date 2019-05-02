#!/usr/bin/env python3

# Given a sparse graph:
# Build Laplacian
# Compute eigenvectors, eigenvalues of Laplacian

import os, argparse
import numpy as np
import scipy.linalg
import util
from util import tictoc

MAX_EIGVAL = 300

class Clustering(object):
	def __init__(self, graph, savedir=None):
		if isinstance(graph, str):
			graph = np.load(graph)
		self.savedir = savedir
		self.graph   = graph # Adjacency matrix
		self.Lu      = None  # Unnormalized Laplacian
		self.Ln      = None  # Normalized Laplacian
		self.Lrw     = None  # Random-walk Laplacian
	def unnormalized_laplacian(self):
		# (unnormalized Laplacian) = (degree matrix of graph) - (adjacency matrix of graph)
		if self.Lu is None:
			tic = util.tictoc('Making Unnormalized Laplacian...')
			d = np.sum(self.graph, axis=0)
			D = np.diag(d)
			self.Lu = D - self.graph
			tictoc('Made Unnormalized Laplacian.', tic)
		return self.Lu
	def normalized_laplacian(self):
		# $L_N = D^{-1/2} L_U D^{-1/2}$ Normalized Laplacian
		if self.Ln is None:
			tic = util.tictoc('Making Normalized Laplacian...')
			d = np.sum(self.graph, axis=0)
			normD = np.diag(d**(-1./2.))
			self.Ln = np.matmul(np.matmul(normD, self.unnormalized_laplacian()), normD)
			tictoc('Made Normalized Laplacian.', tic)
		return self.Ln
	def random_walk_laplacian(self):
		# $L_{RW} = D^{-1} L_U$ Random walk Laplacian
		if self.Lrw is None:
			tic = util.tictoc('Making Random-walk Laplacian...')
			d = np.sum(self.graph, axis=0)
			rwD = np.diag(d**(-1))
			self.Lrw = np.matmul(rwD, self.unnormalized_laplacian())
			tictoc('Made Random-walk Laplacian.', tic)
		return self.Lrw
	# Return embedding based on Normalized Laplacian
	def embedding_n(self):
		self._embedding( self.normalized_laplacian(), 'normalized' )
	# Return embedding based on Random-walk Laplacian
	def embedding_rw(self):
		self._embedding( self.random_walk_laplacian(), 'random-walk' )
	# def _embedding(self, laplacian, name):
	# 	if self.savedir is not None:


# Return eigendecomposition: vals, vecs
def eigen(laplacian):
	return scipy.linalg.eigh(laplacian)
	# return scipy.linalg.eigh(laplacian, eigvals=(0, MAX_EIGVAL))

# Make image of eigenvalues
def plot_eigvals(graph, k, title, savedir):
	import matplotlib.pyplot as plt
	def _plot_eigvals(laplacian, title):
		vals, vecs = eigen(laplacian)
		fig = plt.figure()
		# import IPython; IPython.embed(); # to do: delete
		fig.suptitle(title)
		x = np.arange(0, len(vals))
		plt.plot(x, vals)
		fig.savefig(os.path.join(savedir, title.replace(' ','-')))
	clustering = Clustering(graph, k)
	lN  = clustering.normalized_laplacian()
	_plot_eigvals(lN, title('Normalized'))
	lRw = clustering.random_walk_laplacian()
	_plot_eigvals(lN, title('Random-walk'))

def run(graph, k):
	clustering = Clustering(graph)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	util.add_default_args(parser)
	parser.add_argument('--knn', type=int, help='Number of nearest neighbours to preserve in sparse graph')
	parser.add_argument('--km', type=int, default=12, help='Number of nearest clusters for k-means clustering. The default of 12 was chosen because that\'s how many class labels Gerald indicated.')
	parser.add_argument('-g', '--graph', help='Path to adjacency matrix npy file.')
	parser.add_argument('--eigplot', action='store_true', help='Only plot eigenvalues. Don\'t run clustering.')
	args = parser.parse_args()
	if args.eigplot:
		title = lambda lap_name: 'Eigenvalues %s (connectivity %d)' % (lap_name, args.knn)
		plot_eigvals(args.graph, args.km, title, args.savedir)
	else:
		run(args.graph, args.km)
