#!/bin/bash

./run_etd.sh randomWalk 10
./run_etrace.sh randomWalk 10
./run_ptd.sh randomWalk 10

python plots/plot.py
