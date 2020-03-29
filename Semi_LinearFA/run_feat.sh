#!/bin/bash

while read n;
do
	while read lr;
	do
		echo --n="$n" --lr="$lr" --env=$1 --episodes=$2
		python drl.py --n="$n" --lr="$lr" --env=$1 --episodes=$2
	done < lr.txt
done < len.txt