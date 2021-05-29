#!/bin/bash

while read lr;
do
	while IFS=, read alg len;
	do
		while read seed;
		do
			echo "$alg.py --seed="$seed" --lr="$lr" --len="$len" --env="YChain" --episodes=100"
			python $alg.py --seed="$seed" --lr="$lr" --len="$len" --env="YChain" --episodes=100
		done < seed.txt
	done < v2_ychain.txt
done < lr.txt
