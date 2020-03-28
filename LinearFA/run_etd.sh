#!/bin/bash

while read len;
do
	while read intr;
	do
		while read lr;
		do
			while read seed;
			do
				echo "--seed="$seed" --lr="$lr" --intrst="$intr" --len="$len" --env=$1 --episodes=$2"
				python etd.py --seed="$seed" --lr="$lr" --intrst="$intr" --len="$len" --env=$1 --episodes=$2
			done < seed.txt
		done < lr.txt
	done < interest.txt
done < len.txt