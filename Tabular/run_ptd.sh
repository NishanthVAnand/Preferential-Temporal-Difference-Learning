#!/bin/bash

while read lam;
do
	while read lr;
	do
		while read seed;
		do
			python ptd.py --seed="$seed" --lr="$lr" --lamb="$lam" --env=$1 --episodes=$2
		done < seed.txt
	done < lr.txt
done < lamb.txt
