#!/bin/bash

mkdir -p data

python data_generate.py --n=8 --env="gridWorld"
python data_generate.py --n=12 --env="gridWorld"
python data_generate.py --n=16 --env="gridWorld"

mkdir -p results_gridWorld

while IFS=, read size trace h lr interest;
do
	echo --n="$size" --hidden="$h" --trace_type="$trace" --intrst="$interest" --lr="$lr" --env="gridWorld"
	python drl_data_online_via_traces.py --n="$size" --hidden="$h" --trace_type="$trace" --intrst="$interest" --lr="$lr" --env="gridWorld" --episodes=250 --t_seeds=50
done < task1_traces.txt

python plots/plot_task1_traces.py