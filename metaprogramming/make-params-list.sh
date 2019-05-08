
# PCA	0 50
# KNN	4 8 16 32
# DIM	512 4 256 8 0 128 64 32
# KM	12	24
# LAP	n	rw

# PCA	KNN	DIM	KM	LAP
echo -e {0,50}\\t{4,8,16,32}\\t{512,4,256,8,0,128,64,32}\\t{12,24}\\t{n,rw}\\n \
	| sed -e 's/^ //' -e '/^$/d' > params.txt
echo -n > batch.sh

while read PCA KNN DIM KM LAP; do
	 echo P $PCA K $KNN D $DIM K $KM L $LAP
	if [[ -z $LAP ]]; then
		>&2 echo Bad line P $PCA K $KNN D $DIM K $KM L $LAP
		exit 8
	fi
	echo bash run.sh $PCA $KNN $DIM $KM $LAP >> batch.sh
done < params.txt

split -n r/20 -d -e -a 3 --additional-suffix=.sh batch.sh batch-

