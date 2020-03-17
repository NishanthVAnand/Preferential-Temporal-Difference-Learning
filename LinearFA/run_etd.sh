#!/bin/bash

while read intr;
do
	while read lr;
	do
		while read seed;
		do
			python etd.py --seed="$seed" --lr="$lr" --intrst="$intr" --env=$1 --episodes=$2
		done < seed.txt
	done < lr.txt
done < interest.txt
