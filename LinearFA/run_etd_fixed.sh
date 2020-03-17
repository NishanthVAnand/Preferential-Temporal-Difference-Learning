#!/bin/bash

while read len;
do
	while read int_fo;
	do
		while read int_po;
		do
			while read lamb;
			do
				while read lr;
				do
					while read seed;
					do
						python etd_fixed.py --seed="$seed" --lr="$lr" --len="$len" --lamb="$lamb" --interest_fo="$int_fo" --interest_po="$int_po" --env=$1 --episodes=$2
					done < seed.txt
				done < lr.txt
			done < lamb.txt
		done < int_po.txt
	done < int_fo.txt
done < len.txt
