# Preferential Temporal Difference Learning

## Tabular

To run the tabular experiments (Figure 2) run the following command:

```
cd Tabular
./run_tabular.sh
```

## Linear

Run the following commands to reproduce Figure 3 and A.2.

```
cd LinearFA
./run_task1.sh
./run_task2.sh
```

The next two settings requires `pytorch` to reproduce the figures.

## Semi-Linear

First run the feature net by executing the following command:
```
cd Semi_LinearFA
./run_MC.sh
```
Running the following commands will reproduce results presented in Figure 4 and A.8 by loading the trained feature net models.

```
./run_task1.sh
./run_task2.sh
```

## Non-Linear

Running the following commands will generate Figures 5 and 6 from the paper. The first two commands correspond to the forward-view results and the next two commands correspond to the backward-view results.

*Comment data generation part inside each script file if your Data folder is not empty*

```
cd NonLinearFA
./run_task1.sh
./run_task2.sh
./run_task1_traces.sh
./run_task2_traces.sh
```

All figures are saved in `plots` directory of the respective settings. For example, the tabular figures are saved in `Tabular/plots/` directory.

Results are saved in `results_` directories after and you can delete them after generating the plot.