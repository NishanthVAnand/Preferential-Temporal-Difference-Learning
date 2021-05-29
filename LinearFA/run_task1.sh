#!/bin/bash

mkdir -p results_YChain

while IFS=, read alg len lr;
do
	while read seed;
	do
		echo "$alg.py --seed="$seed" --lr="$lr" --len="$len" --env="YChain" --episodes=100"
		python $alg.py --seed="$seed" --lr="$lr" --len="$len" --env="YChain" --episodes=100
	done < seed.txt
done < YChain.txt

python plots/plot_ychain.py
