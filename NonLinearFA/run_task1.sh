#!/bin/bash

mkdir -p data

python data_generate.py --n=8 --env="gridWorld"
python data_generate.py --n=12 --env="gridWorld"
python data_generate.py --n=16 --env="gridWorld"

mkdir -p results_gridWorld

hunits = ( 1 2 4 8 16 )
for units in ${hunits[*]};
do
	while IFS=, read size trace interest lr;
	do
		echo --n="$size" --hidden="$h" --trace_type="$trace" --intrst="$interest" --lr="$lr" --env="gridWorld"
		python drl_data.py --n="$size" --hidden="$h" --trace_type="$trace" --intrst="$interest" --lr="$lr" --env="gridWorld" --episodes=250 --t_seeds=50
	done < task1.txt
done

python plots/plot_task1.py