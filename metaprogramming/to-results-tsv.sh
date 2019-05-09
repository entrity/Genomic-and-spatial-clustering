#!/bin/bash

find logs -type f -name \*.log | while read -r f; do
	grep -P '^>>>' "$f" \
	| xargs echo \
	| sed -e 's/>>>/\t/g' -e 's/ \t/\t/g' -e 's/\t /\t/g' -e 's/^\t//' -e 's/ /\t/g' \
	| tr -s $'\t' \
	| tr -d $'\n'
	if [[ $f =~ no-spatial ]]; then
		echo -e "\t1"
	else
		echo -e "\t0"
	fi
done \
| cut -f 6-10,19- \
| sed -e 's/\tn\t/\t0\t/' -e 's/\trw\t/\t1\t/' \
> results.tsv
