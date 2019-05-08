import os, pickle, argparse, numpy as np
import util
import preprocess
import graphs
import cluster
import evaluation
import my_loss

# args : xy_npy, xy_csv, gen_npy, gen_csv, pca
def _preprocess():
	if os.path.exists(args.xy_npy):
		xy = np.load(args.xy_npy)
	else:
		xy = preprocess.load_xy_csv(args.xy_csv)
		np.save(args.xy_npy, xy)
	if os.path.exists(args.gen_npy):
		gens = np.load(args.gen_npy)
	else:
		gens = preprocess.load_transcriptome_csv(args.gen_csv)
		preprocess.preprocess_transcriptome(gens, args.pca, args.gen_npy)
	return xy, gens

# args : fc, sparse, knn
def _graphs(xy, gens):
	if os.path.exists(args.sparse):
		sparse_graph = np.load(args.sparse)
	else:
		g = graphs.BaseGraph(args.fc, None)
		if g.A is None:
			g.fc_graph(xy, gens, args.fc)
		g.sparsify(args.knn, args.sparse)
		sparse_graph = g.A
	util.debug('RUN\tnonzeros in graph\t%d' % np.count_nonzero(sparse_graph))
	return sparse_graph

# args : kmobj, lapmode, sparse, dim, embedding, km
def _cluster():
	if os.path.exists(args.kmobj) and os.path.exists(args.embedding):
		with open(args.kmobj, 'rb') as fin:
			kmobj = pickle.load(fin)
		embedding = np.load(args.embedding)
	else:
		if args.lapmode == 'n':
			clustering = cluster.NormalizedClustering(args.sparse, args.dim, None)
		elif args.lapmode == 'rw':
			clustering = cluster.RandomWalkClustering(args.sparse, args.dim, None)
		else:
			raise Exception('Illegal value for laplacian mode')
		kmobj = clustering.cluster(args.km)
		with open(args.kmobj, 'wb') as fout:
			pickle.dump(kmobj, fout)
		np.save(args.embedding, clustering.embedding)
		embedding = clustering.embedding
	return kmobj, embedding

# args : id_csv, lbl_csv
def _evaluation(sparse_graph, kmobj, embeddings):
	# Compute 'Accuracy' based on Gerald's labels
	pos_mask, neg_mask    = evaluation.get_ground_truth(sparse_graph, args.id_csv, args.lbl_csv, args.cluster_membership_dir)
	bin_graph             = evaluation.graph_from_clusters(kmobj.labels_)
	pos_ct, neg_ct, score = evaluation.acc(bin_graph, pos_mask, neg_mask)
	# Compute error based on KL Divergence
	kl_div_err = my_loss.compute_kl_div_loss_from_numpy(embeddings, kmobj).item()
	# Print
	print('>>>\tKL D\tpos\tneg\tscore\tbinNz\tposNz\tnegNz')
	print('>>>\t%f\t%d\t%d\t%f\t%d\t%d\t%d' % (kl_div_err, pos_ct, neg_ct, score, np.count_nonzero(bin_graph), np.count_nonzero(pos_mask), np.count_nonzero(neg_mask)))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	util.add_default_args(parser)
	args = parser.parse_args()
	print(args)
	print('>>>\tPCA\tDIM\tKM\tKNN\tLAP')
	print('>>>\t%d\t%d\t%d\t%d\t%s' % (args.pca, args.dim, args.km, args.knn, args.lapmode))
	# Preprocess: transcriptomes, xy
	xy, gens = _preprocess()
	# Graphs: fc-graph, sparse-graph
	sparse_graph = _graphs(xy, gens)
	# Cluster: embedding, kmeans
	kmobj, embeddings = _cluster()
	# Evaluation: (print)
	_evaluation(sparse_graph, kmobj, embeddings)
