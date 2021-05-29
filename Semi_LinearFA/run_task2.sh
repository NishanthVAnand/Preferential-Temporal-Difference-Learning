#!/bin/bash

while IFS=, read size trace interest lr epi;
do
	echo --n="$size" --trace_type="$trace" --intrst="$interest" --lr="$lr" --env="gridWorld2" --episodes="$epi"
	python drl.py --n="$size" --trace_type="$trace" --intrst="$interest" --lr="$lr" --env="gridWorld2" --episodes="$epi" --t_seeds=50
done < task2.txt

python plots/plot_task2.py
