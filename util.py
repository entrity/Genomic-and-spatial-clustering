import datetime
import argparse

def add_default_args(parser):
	raw = parser.add_argument_group('Raw data')
	intm = parser.add_argument_group('Intermediates data')
	raw.add_argument('--xy-csv', default='data/output_centroids.txt', help='path to CSV file containing x,y coords of all cells')
	raw.add_argument('--gen-csv', default='data/output_count_matrix.txt', help='path to CSV file containing transcriptome data for all cells')
	intm.add_argument('--xy-npy', default='data/xy.npy', help='path to NPY file containing x,y coords of all cells')
	intm.add_argument('--gen-npy', default='data/transcriptomes.npy', help='path to NPY file containing whitened transcriptome data for all cells')
	raw.add_argument('--id_csv', default='data/output_keptCellID.txt', help='path to CSV file containing ID\'s all cells')
	raw.add_argument('--lbl_csv', default='data/class_labels.csv', help='path to CSV file containing class labels of all cells')
	intm.add_argument('--fc', help='path to NPY file containing fully-connected graph as adjacency matrix')
	intm.add_argument('-s', '--savedir', '--save', help='Save directory')
	intm.add_argument('-g', '--graph', help='Adjacency matrix')

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
