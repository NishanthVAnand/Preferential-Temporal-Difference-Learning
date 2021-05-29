#!/bin/bash

while read lr;
do
	while IFS=, read alg len;
	do
		while read seed;
		do
			echo "$alg.py --seed="$seed" --lr="$lr" --len="$len" --env="elevator" --episodes=200"
			python $alg.py --seed="$seed" --lr="$lr" --len="$len" --env="elevator" --episodes=200
		done < seed.txt
	done < v2_ychain.txt
done < lr.txt