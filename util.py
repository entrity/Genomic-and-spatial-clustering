import datetime
import argparse
import logging

def init_logger(path):
	logging.basicConfig(filename=path, format='%(message)s')
	logging.getLogger().addHandler(logging.StreamHandler())
	logging.getLogger().setLevel(logging.INFO)

def add_default_args(parser):
	raw = parser.add_argument_group('Raw data')
	intm = parser.add_argument_group('Intermediates data')
	raw.add_argument('--xy-csv', '--xy_csv', default='data/output_centroids.txt', help='path to CSV file containing x,y coords of all cells')
	raw.add_argument('--gen-csv', '--gen_csv', default='data/output_count_matrix.txt', help='path to CSV file containing transcriptome data for all cells')
	intm.add_argument('--xy-npy', '--xy_npy', default='data/xy.npy', help='path to NPY file containing x,y coords of all cells')
	intm.add_argument('--gen-npy', '--gen_npy', default='data/transcriptomes.npy', help='path to NPY file containing whitened transcriptome data for all cells')
	raw.add_argument('--id-csv', '--id_csv', default='data/output_keptCellID.txt', help='path to CSV file containing ID\'s all cells')
	raw.add_argument('--lbl-csv', '--lbl_csv', default='data/class_labels.csv', help='path to CSV file containing class labels of all cells')
	intm.add_argument('--knn-feats', '--knn_feats')
	intm.add_argument('--knn-idxs', '--knn_idxs')
	intm.add_argument('-s', '--savedir', '--save', help='Save directory')
	intm.add_argument('--fc', help='fully-connected graph as adjacency matrix (npy file)')
	intm.add_argument('-g', '--sparse', help='sparse adjacency matrix (npy file)')
	parser.add_argument('--knn', type=int, help='Order of k-nearest neighbours graph')
	parser.add_argument('--km', type=int, help='Number of clusters')
	parser.add_argument('--pca', type=int, help='Number of dimensions to use for PCA from raw inputs')
	parser.add_argument('--dim', type=int, help='Number of eigenvectors to use for embedding')
	parser.add_argument('--kmobj', help='Path to pickle of kmeans object')
	parser.add_argument('--embedding', help='Path to embedding as npy file')
	parser.add_argument('--lapmode', help='Indicates whether to use normalized or random-walk Laplacian')
	parser.add_argument('--no-spatial', '--no-spatial', action='store_true', help='Don\'t use xy data when clustering in graphs.BaseGraph.fc_graph')
	parser.add_argument('--cluster-membership-dir', '--cluster_membership_dir', help='Directory in which to store text files showing cell ids belonging to each cluster')


def default_args():
	parser = argparse.ArgumentParser()
	add_default_args(parser)
	return parser.parse_args()

def debug(*msg):
	print(*msg)

def tictoc(msg='', tic=None):
	now = datetime.datetime.now()
	if tic is not None:
		diff = now - tic
		debug('%s %s' % (msg, str(diff)))
	else:
		debug(msg)
		return now

def info(*args):
	logging.getLogger.info(args)
