#!/bin/bash

while read n;
do
	while IFS=, read trace interest;
	do
		while read lr;
		do
			echo --trace_type="$trace" --intrst="$interest" --n="$n" --lr="$lr" --env=$1 --episodes=$2
			python drl.py --trace_type="$trace" --intrst="$interest" --n="$n" --lr="$lr" --env=$1 --episodes=$2
		done < lr.txt
	done < trace.txt
done < len.txt