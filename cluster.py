#!/usr/bin/env python3

# Given a sparse graph:
# Build Laplacian
# Compute eigenvectors, eigenvalues of Laplacian

import os, argparse
import pickle
import numpy as np
import scipy.linalg
import sklearn.cluster
import util
from util import tictoc

MAX_EIGVAL = 300

class Clustering(object):
	def __init__(self, graph, dim, savedir=None):
		if isinstance(graph, str):
			graph = np.load(graph)
		self.savedir = savedir
		self.graph   = graph # Adjacency matrix
		self.dim     = dim   # Dimensionality, i.e. number of eigenvectors to use in embedding. If 'None', then all will be used
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
	def cluster_n(self, k):
		embedding = self._embedding( self.normalized_laplacian(), 'normalized' )
		return self._cluster(embedding, k)
	def cluster_rw(self, k):
		embedding = self._embedding( self.random_walk_laplacian(), 'random-walk' )
		return self._cluster(embedding, k)
	def _embedding(self, laplacian, name):
		vals, vecs = eigen(laplacian)
		if self.dim is not None:
			I = vals.argsort()
			vecs = vecs[I, :self.dim]
		return vecs
	def _cluster(self, embedding, k):
		tic = util.tictoc('clustering...')
		kmeans = sklearn.cluster.KMeans(k).fit(embedding)
		util.tictoc('clustered.', tic)
		self._save('kmeans-dim-%s-k-%d.pkl' % (str(self.dim), k), kmeans)
		return kmeans
	def _save(self, name, obj):
		if self.savedir is not None:
			path = os.path.join(self.savedir, name)
			util.debug('Saving %s...' % path)
			with open(path, 'wb') as fout:
				pickle.dump(obj, fout)

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
	clustering = Clustering(graph, None)
	lN  = clustering.normalized_laplacian()
	_plot_eigvals(lN, title('Normalized'))
	lRw = clustering.random_walk_laplacian()
	_plot_eigvals(lN, title('Random-walk'))


def plot_clusters(xy_npy, kmeans):
	xy = np.load(xy_npy)
	print(len(kmeans.labels_))
	print(np.histogram(kmeans.labels_, bins=np.arange(0,13)-0.5))
	import matplotlib.pyplot as plt
	clr = kmeans.labels_
	# clr = np.arange(0, xy.shape[0]) % 3
	# print(clr)
	# import IPython; IPython.embed(); # to do: delete
	plt.scatter(xy[:,0], xy[:,1], c=clr)
	plt.show()
	# import IPython; IPython.embed(); # to do: delete


def run(graph, dim, k, do_plot, xy_npy, savedir=None):
	print(dim, k)
	clustering = Clustering(graph, dim, savedir)
	kmeans = clustering.cluster_n(k)
	if do_plot: plot_clusters(xy_npy, kmeans)
	# clustering.cluster_rw(k)
	# if do_plot: plot_clusters(xy_npy, kmeans)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	util.add_default_args(parser)
	parser.add_argument('--knn', type=int, help='Number of nearest neighbours to preserve in sparse graph')
	parser.add_argument('--km', type=int, default=12, help='Number of nearest clusters for k-means clustering. The default of 12 was chosen because that\'s how many class labels Gerald indicated.')
	parser.add_argument('--dim', type=int, help='Number of dimensions (eigenvectors) to use in embedding.')
	parser.add_argument('--scatterplot', action='store_true', help='Make a 2D scatterplot of the cells.')
	parser.add_argument('--eigplot', action='store_true', help='Only plot eigenvalues. Don\'t run clustering.')
	args = parser.parse_args()
	if args.eigplot:
		title = lambda lap_name: 'Eigenvalues %s (connectivity %d)' % (lap_name, args.knn)
		plot_eigvals(args.graph, args.km, title, args.savedir)
	else:
		run(args.graph, args.dim, args.km, args.scatterplot, args.xy_npy, args.savedir)
