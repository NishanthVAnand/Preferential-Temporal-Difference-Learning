# Task 1 -  gridWorld

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.backends.backend_pdf
from scipy.integrate import trapz

hunits = [1, 2, 4, 8, 16]
opt_comb = {"gated":{8:0.01, 12:0.01, 16:0.01}, "accumulating":{8:0.005, 12:0.001, 16:0.001},\
            "etd":{8:0.05, 12:0.01, 16:0.01}, "etd_adaptive":{8:0.001, 12:0.001, 16:0.001}}
colors = {"etd": "magenta", "etd_adaptive": "blue",\
          "accumulating": "green", "gated": "red"}
markers = {"etd": "*", "etd_adaptive":"D",\
          "accumulating": "x", "gated": "o"}
name = {"etd": "ETD fixed", "etd_adaptive":"ETD variable",\
          "accumulating": r"TD($\lambda$)", "gated": "PTD"}
intrst={"etd": 0.01, "etd_adaptive": 0.5}

env = "gridWorld"
t_seeds = 50
episodes = 250

z_star = 1.96
fig, ax = plt.subplots(1, 3, figsize=(18,6))
pdf = matplotlib.backends.backend_pdf.PdfPages("plots/End2End_Grid1_hidden_units_test.pdf")

for t, size_lr in opt_comb.items():
    cnt = 0
    for size, lrate in size_lr.items():
        hunits_mean = []
        hunits_std = []
        for hidden in hunits:
            if t in ["etd", "etd_adaptive"]:
                filename = "drl_"+t+"_int_"+str(intrst[t])+"_env_"+str(env)+"_size_"+str(size)+"_hidden_"+str(hidden)+"_lr_"+str(lrate)+"_seeds_"+str(t_seeds)+"_epi_"+str(episodes)
            else:
                filename = "drl_"+t+"_env_"+str(env)+"_size_"+str(size)+"_hidden_"+str(hidden)+"_lr_"+str(lrate)+"_seeds_"+str(t_seeds)+"_epi_"+str(episodes)

            with open("results_"+str(env)+"/"+filename+"_errors_mean.pkl", "rb") as f:
                avg_error = pickle.load(f)
                
            with open("results_"+str(env)+"/"+filename+"_errors_std.pkl", "rb") as f:
                avg_std = pickle.load(f)

            auc = trapz(avg_error, np.linspace(0,1,episodes))
            auc_err = trapz(avg_error+z_star*(avg_std/t_seeds**0.5), np.linspace(0,1,episodes))
            hunits_mean.append(auc)
            hunits_std.append(auc_err - auc)
        #ax[cnt].plot(hunits, hunits_mean, color=colors[t], marker=markers[t])
        ax[cnt].errorbar(hunits, hunits_mean, hunits_std, elinewidth=0.8, color=colors[t], marker=markers[t])
        ax[cnt].set(xscale="log")
        ax[cnt].set_xticks(hunits)
        ax[cnt].set_xticklabels(hunits)
        ax[cnt].set_title("Grid size = "+str(size), fontsize=16)
        ax[cnt].set_xlabel("Number of hidden units", fontsize=16)
        ax[cnt].set_ylabel("AUC of averaged MSE", fontsize=16)
        cnt += 1
fig.legend([name[t] for t, _ in opt_comb.items()], bbox_to_anchor=(0.5, 0., 0.1, 0.8), frameon=False, fontsize=16)
fig.tight_layout()
pdf.savefig(fig, bbox_inches = 'tight')
pdf.close()        
