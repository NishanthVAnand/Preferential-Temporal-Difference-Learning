#!/bin/bash

while read intr;
do
	while read lam;
	do
		while read lr;
		do
			while read seed;
			do
				python etd.py --seed="$seed" --lr="$lr" --lamb="$lam" --intrst="$intr" --env=$1 --episodes=$2
			done < seed.txt
		done < lr.txt
	done < lamb.txt
done < interest.txt