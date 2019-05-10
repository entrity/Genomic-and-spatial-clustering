#!/bin/bash

PCA=0
KNN=4
DIM=64
KM=12
LAP=n
SPATIAL_ARG=--no-spatial

SPA=$SPATIAL_ARG


CLUSTER_DIR="debug/pca_$PCA${SPA}-knn_$KNN-dim_$DIM-km_$KM-lap_$LAP"

mkdir -p $CLUSTER_DIR
mkdir -p artefacts
mkdir -p logs

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
	--fc artefacts/fc-pca_$PCA${SPA}-knn_$KNN.npy \
	--sparse artefacts/sparse-pca_$PCA${SPA}-knn_$KNN.npy \
	--kmobj artefacts/kmeans-pca_$PCA${SPA}-knn_$KNN-dim_$DIM-km_$KM-lap_$LAP.pkl \
	--embedding artefacts/embedding-pca_$PCA${SPA}-knn_$KNN-dim_$DIM-km_$KM-lap_$LAP.npy \
	--cluster-membership-dir "$CLUSTER_DIR" \
	$SPATIAL_ARG \
	#> logs/pca_$PCA${SPA}-knn_$KNN-dim_$DIM-km_$KM-lap_$LAP.log

