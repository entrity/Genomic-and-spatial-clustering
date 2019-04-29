def add_default_args(parser):
	parser.add_argument('--xy_csv', default='data/output_centroids.txt', help='path to CSV file containing x,y coords of all cells')
	parser.add_argument('--gen_csv', default='data/output_count_matrix.txt', help='path to CSV file containing transcriptome data for all cells')
	parser.add_argument('--xy_npy', default='data/xy.npy', help='path to NPY file containing x,y coords of all cells')
	parser.add_argument('--gen_npy', default='data/transcriptomes.npy', help='path to NPY file containing whitened transcriptome data for all cells')
	parser.add_argument('--id_csv', default='data/output_keptCellID.txt', help='path to CSV file containing ID\'s all cells')
	parser.add_argument('--lbl_csv', default='data/class_labels.csv', help='path to CSV file containing class labels of all cells')
