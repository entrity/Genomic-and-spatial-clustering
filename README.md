See preliminary [Project writeup (0.5 pages, including data source)](https://www.overleaf.com/project/5cb7a3821f07a5705c5aa3ae), written prior to the start of work.

## Aim

My project is an unsupervised approach at identifying patterns of cell types in a 2D layout. It has come to my attention that in the last one or two years, datasets have become available which represent not only the genomic data of sampled cells but also the cells’ location in a 2D surface.

## Approach

1. **Fully-connected graph:** The aforementioned representation comprising both spatial and genomic information will enable me to construct a fully-connected graph, with the edge weight between cell `i` and cell `j` calculated as `d(i, j) × c(i, j)`, where `d` represents the spatial distance and `c` represents the feature distance between cells `i` and `j`. (For these two distances, I expect to work with Gaussian distance, but given time, I may also experiment with cosine distance or Euclidean distance for either one. Furthermore, I might experiment with variants of the distance calculation stated above so as to favour spatial and feature distance differently.)
2. **Sparsify graph:** To sparsify the graph, I will run k-nearest-neighbours and keep only the edges for these neighbourhoods.
3. **GCN:** I intend to train a graph convolution network (GCN) as an autoencoder on the graph data. Being that k-nearest-neighbours will give me a fixed neighbourhood size, my GCN’s aggregator function can be to simply concatenate genomic features of all of the cells in each neighbourhood. (This has shortcomings, but GCN’s are new to me, so I intend this to be at least my first approach. If all proceeds quickly, I may experiment with other aggregator functions.) The embedding produced by the GCN’s encoder will then serve as a new feature representation (replacing the former spatial and feature data) for each cell (or, rather, each neighbourhood, with a neighbourhood being focused on a single cell).
4. **Spectral clustering:** In this new feature space, I will perform spectral clustering. I intend to use the normalized Laplacian to this end, but given time, I may also experiment with a random-walk Lalacian. Moreover, I may experiment with clustering on the feature space in addition to clustering on the eigenvectors of the Laplacians. (GCN’s and spectral clustering are both new to me, so I anticipate the possibility that I will not move quickly enough to compare results with alternative designs.)
5. **Qualitative inspection:** I will perform qualitative examination of neighbourhoods nearest to the cluster centroids, looking for cell-type consistency within clusters and differentiation between clusters.

## Baseline/strawman approach

I plan to apply spectral clustering directly to the graph data (omitting the aforementioned GCN.

1. Fully-connected graph
2. Sparse graph
3. Spectral clustering
4. Qualitative inspection
