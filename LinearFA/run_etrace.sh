#!/bin/bash

while read lr;
do
	while read seed;
	do
		python etrace.py --seed="$seed" --lr="$lr" --env=$1 --episodes=$2
	done < seed.txt
done < lr.txt
