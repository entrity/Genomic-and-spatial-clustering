
# PCA	0 50
# KNN	4 8 16 32
# DIM	512 4 256 8 0 128 64 32
# KM	12	24
# LAP	n	rw

# PCA	KNN	DIM	KM	LAP
echo -n > batch.sh
echo -e {0,50}\\t{4,8,16,32}\\t{512,4,256,8,0,128,64,32}\\t{12,24}\\t{n,rw}\\t{--no-spatial,}\\n \
| sed -e 's/^ //' -e '/^$/d' \
| while read PCA KNN DIM KM LAP; do
	if [[ -z $LAP ]]; then
		>&2 echo Bad line P $PCA K $KNN D $DIM K $KM L $LAP S $SPAT
		exit 8
	fi
	echo bash run.sh $PCA $KNN $DIM $KM $LAP $SPAT >> batch.sh
done

split -n r/20 -d -e -a 3 --additional-suffix=.sh batch.sh batch-
