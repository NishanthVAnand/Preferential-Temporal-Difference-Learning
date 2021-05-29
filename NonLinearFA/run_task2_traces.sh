#!/bin/bash

mkdir -p data

python data_generate.py --n=8 --env="gridWorld2"
python data_generate.py --n=12 --env="gridWorld2"
python data_generate.py --n=16 --env="gridWorld2"

mkdir -p results_gridWorld2

while IFS=, read size trace h lr interest;
do
	echo --n="$size" --hidden="$h" --trace_type="$trace" --intrst="$interest" --lr="$lr" --env="gridWorld2"
	python drl_data_online_via_traces.py --n="$size" --hidden="$h" --trace_type="$trace" --intrst="$interest" --lr="$lr" --env="gridWorld2" --episodes=250 --t_seeds=50
done < task2_traces.txt

python plots/plot_task2_traces.py