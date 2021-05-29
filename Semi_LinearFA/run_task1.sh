#!/bin/bash

mkdir -p results_gridWorld

while IFS=, read size trace interest lr epi;
do
	echo --n="$size" --trace_type="$trace" --intrst="$interest" --lr="$lr" --env="gridWorld" --episodes="$epi"
	python drl.py --n="$size" --trace_type="$trace" --intrst="$interest" --lr="$lr" --env="gridWorld" --episodes="$epi" --t_seeds=50
done < task1.txt

python plots/plot_task1.py
