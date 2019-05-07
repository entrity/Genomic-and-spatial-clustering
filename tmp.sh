#!/bin/bash

PCA=50
KNN=4
DIM=64
KM=12
LAP=rw

CLUSTER_DIR="debug/pca_$PCA-knn_$KNN-dim_$DIM-km_$KM"

mkdir -p $CLUSTER_DIR
mkdir -p artefacts

python run.py \
	--id_csv  data/output_keptCellID.txt \
	--lbl_csv data/class_labels.csv \
	--xy_csv  data/output_centroids.txt \
	--gen_csv data/output_count_matrix.txt \
	--xy_npy  artefacts/xy.npy \
	--gen_npy artefacts/gen-$PCA.npy \
	--km $KM \
	--pca $PCA \
	--knn $KNN \
	--dim $DIM \
	--lapmode $LAP \
	--fc artefacts/fc-pca_$PCA-knn_$KNN.npy \
	--sparse artefacts/sparse-pca_$PCA-knn_$KNN.npy \
	--kmobj artefacts/kmeans-pca_$PCA-knn_$KNN-dim_$DIM-km_$KM.pkl \
	--embedding artefacts/embedding-pca_$PCA-knn_$KNN-dim_$DIM-km_$KM.npy \
	--cluster-membership-dir "$CLUSTER_DIR"
