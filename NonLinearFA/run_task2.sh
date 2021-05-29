#!/bin/bash

mkdir -p data

python data_generate.py --n=8 --env="gridWorld2"
python data_generate.py --n=12 --env="gridWorld2"
python data_generate.py --n=16 --env="gridWorld2"

mkdir -p results_gridWorld2

hunits = ( 1 2 4 8 16 )
for units in ${hunits[*]};
do
	while IFS=, read size trace interest lr;
	do
		echo --n="$size" --hidden="$h" --trace_type="$trace" --intrst="$interest" --lr="$lr" --env="gridWorld2"
		python drl_data.py --n="$size" --hidden="$h" --trace_type="$trace" --intrst="$interest" --lr="$lr" --env="gridWorld2" --episodes=250 --t_seeds=50
	done < task2.txt
done

python plots/plot_task2.py