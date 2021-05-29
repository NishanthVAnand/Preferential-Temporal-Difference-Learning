#!/bin/bash

while read lamb;
do
	while read lr;
	do
		while read len;
		do
			while read seed;
			do
				echo "etd_fixed_v2.py --seed="$seed" --lr="$lr" --len="$len" --lamb=$lamb --env="YChain" --episodes=100"
				python etd_fixed_v2.py --seed="$seed" --lr="$lr" --len="$len" --lamb=$lamb --env="YChain" --episodes=100
			done < seed.txt
		done < len.txt
	done < lr.txt
done < lamb.txt