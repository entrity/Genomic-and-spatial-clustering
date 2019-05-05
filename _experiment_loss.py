import torch
import numpy as np
from sklearn.cluster import KMeans
import util

# Ref basic operations
# https://jhui.github.io/2018/02/09/PyTorch-Basic-operations/

embeddings = torch.tensor([
	[1,2,3],
	[0,0.5,4],
	[9,0,0],
	[2,3,0.1],
	]).float()
embeddings.requires_grad_()
# centroids = torch.tensor([
# 	[2,2.5,2],
# 	[7,0.3,0.15],
# 	])

assert isinstance(embeddings, torch.Tensor)

# Cluster
kmeans = 2
if isinstance(kmeans, int):
	kmeans = KMeans(kmeans)
	util.debug('Running k-means on matrix of %d by %d...' % tuple(embeddings.shape))
	kmeans.fit(embeddings.data.numpy())
centroids = kmeans.cluster_centers_

# Get dimensions
I = torch.tensor(kmeans.labels_.shape).prod() # Number of data points
J = centroids.shape[0] # Number of clusters
D = centroids.shape[1] # Number of dimensions for data pts
assert D == embeddings.shape[1] # Dimension of centroids is same as dimensions of embedding

# Expand embeddings
embeddings_tensor = embeddings.view(I,1,D).expand(I,J,D)

# Compute q numerator
q_centroids = torch.from_numpy(centroids)
q_centroids = q_centroids.view(1,J,D).expand(I,-1,-1)
q_centroids.requires_grad_(False)
q_numerator_differences = torch.add(embeddings_tensor, -1, q_centroids)
q_numerator_sq_distances = torch.norm(q_numerator_differences, p=2, dim=2)
print(' CENTROIDS', centroids.shape)
print('QCENTROIDS', q_centroids.shape)
print(' EMBTENSOR', embeddings_tensor.shape)
print('     DIFFS', q_numerator_differences.shape)
print('  SQ DISTS', q_numerator_sq_distances.shape)
print('EMBEDDINGS', embeddings.shape)
assert q_numerator_sq_distances.shape[0] == I
assert q_numerator_sq_distances.shape[1] == J
q_numerators = torch.add(q_numerator_sq_distances, 1)
assert q_numerators.dim() == 2, q_numerators.shape



# End
print('DONE A-OK')
