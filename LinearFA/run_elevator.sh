#!/bin/bash

while IFS=, read alg len lr;
do
	while read seed;
	do
		echo "$alg.py --seed="$seed" --lr="$lr" --len="$len" --env="elevator" --episodes=200"
		python $alg.py --seed="$seed" --lr="$lr" --len="$len" --env="elevator" --episodes=200
	done < seed.txt
done < elevator.txt

python plots/plot_elevator.py