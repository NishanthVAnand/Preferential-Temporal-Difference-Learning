# Task 1 -  gridWorld

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.backends.backend_pdf
from scipy.integrate import trapz

hunits = [1, 2, 4, 8, 16]
opt_comb = {"gated":{8:{1:0.01, 2:0.01, 4:0.01, 8:0.01, 16:0.01}, 12:{1:0.01, 2:0.01, 4:0.01, 8:0.01, 16:0.01}, 16:{1:0.01, 2:0.01, 4:0.01, 8:0.01, 16:0.01}},\
            "accumulating":{8:{1:0.05, 2:0.05, 4:0.05, 8:0.01, 16:0.01}, 12:{1:0.01, 2:0.01, 4:0.01, 8:0.01, 16:0.01}, 16:{1:0.01, 2:0.01, 4:0.01, 8:0.01, 16:0.01}},\
            "etd_int_0.01":{8:{1:0.05, 2:0.05, 4:0.05, 8:0.05, 16:0.05}, 12:{1:0.01, 2:0.01, 4:0.01, 8:0.01, 16:0.05}, 16:{1:0.05, 2:0.01, 4:0.01, 8:0.01, 16:0.05}},\
            "etd_adaptive_int_0.01":{8:{1:0.001, 2:0.001, 4:0.001, 8:0.001, 16:0.001}, 12:{1:0.0005, 2:0.0005, 4:0.001, 8:0.001, 16:0.001}, 16:{1:0.0005, 2:0.0005, 4:0.0005, 8:0.0005, 16:0.0005}}}
colors = {"etd_int_0.01": "magenta", "etd_adaptive_int_0.5": "blue",\
          "accumulating": "green", "gated": "red"}
markers = {"etd_int_0.01": "*", "etd_adaptive_int_0.5":"D",\
          "accumulating": "x", "gated": "o"}
name = {"etd_int_0.01": "ETD fixed", "etd_adaptive_int_0.5":"ETD variable",\
          "accumulating": r"TD($\lambda$)", "gated": "PTD"}

env = "gridWorld"
t_seeds = 50
episodes = 250

z_star = 1.96
fig, ax = plt.subplots(1, 3, figsize=(18,6))
pdf = matplotlib.backends.backend_pdf.PdfPages("plots/End2End_traces_Grid1_hidden_units_test.pdf")

for t, size_h_lr in opt_comb.items():
    cnt = 0
    for size, h_lrate in size_h_lr.items():
        hunits_mean = []
        hunits_std = []
        for hidden, lrate in h_lrate.items():
            filename = "drl_online_traces_"+t+"_env_"+str(env)+"_size_"+str(size)+"_hidden_"+str(hidden)+"_lr_"+str(lrate)+"_seeds_"+str(t_seeds)+"_epi_"+str(episodes)

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
        ax[cnt].set_ylabel("AUC of Mean error", fontsize=16)
        cnt += 1
        
opt_comb_2 = {"gated":{8:0.01, 12:0.01, 16:0.01}}
colors = {"gated": "k"}

for t, size_lr in opt_comb_2.items():
    cnt = 0
    for size, lrate in size_lr.items():
        hunits_mean = []
        hunits_std = []
        for hidden in hunits:
            if t in ["etd", "etd_adaptive"]:
                filename = "drl_"+t+"_int_"+str(intrst)+"_env_"+str(env)+"_size_"+str(size)+"_hidden_"+str(hidden)+"_lr_"+str(lrate)+"_seeds_"+str(t_seeds)+"_epi_"+str(episodes)
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
        cnt += 1

fig.legend([name[t] for t, _ in opt_comb.items()]+["PTD (forward)"], bbox_to_anchor=(0.5, 0., 0.1, 0.83), frameon=False, fontsize=16)
fig.tight_layout()
pdf.savefig(fig, bbox_inches = 'tight')
pdf.close()