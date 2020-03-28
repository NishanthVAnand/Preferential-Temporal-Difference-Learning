#!/bin/bash

while read len;
do
	while read lr;
	do
		while read seed;
		do
			echo "--seed="$seed" --lr="$lr" --len="$len" --env=$1 --episodes=$2"
			python gated_trace.py --seed="$seed" --lr="$lr" --len="$len" --env=$1 --episodes=$2
		done < seed.txt
	done < lr.txt
done < len.txt