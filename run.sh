#!/bin/bash

PCA=$1
KNN=$2
DIM=$3
KM=$4
LAP=$5
SPATIAL=$6

N_ARGS=5
if [[ $# -lt $N_ARGS ]]; then
	>&2 echo Bad args. Need $N_ARGS. Got $#
	exit 8
fi

CLUSTER_DIR="debug/pca_$PCA${SPATIAL}-knn_$KNN-dim_$DIM-km_$KM-lap_$LAP"

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
	--fc artefacts/fc-pca_$PCA${SPATIAL}-knn_$KNN.npy \
	--sparse artefacts/sparse-pca_$PCA${SPATIAL}-knn_$KNN.npy \
	--kmobj artefacts/kmeans-pca_$PCA${SPATIAL}-knn_$KNN-dim_$DIM-km_$KM-lap_$LAP.pkl \
	--embedding artefacts/embedding-pca_$PCA${SPATIAL}-knn_$KNN-dim_$DIM-km_$KM-lap_$LAP.npy \
	--cluster-membership-dir "$CLUSTER_DIR" \
	$SPATIAL \
	> logs/pca_$PCA${SPATIAL}-knn_$KNN-dim_$DIM-km_$KM-lap_$LAP.log

