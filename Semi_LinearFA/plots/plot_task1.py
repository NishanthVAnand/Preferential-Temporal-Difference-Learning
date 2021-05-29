# Task 1 -  gridWorld

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.backends.backend_pdf

# Task 1 -  gridWorld

def moving_average(a, n=3) :
    return np.convolve(a, np.ones((n,))/n, mode='valid')

t_seeds=50
seed = list(range(t_seeds))
length = [8, 12, 16]
env = "gridWorld"
epi = {8:100, 12:100, 16:50}
z_star = 0.674 # 0.674-50% 1.28-80%, 1.645-90%, 1.96-95%, 2.33-98%, 2.58-99%
pdf = matplotlib.backends.backend_pdf.PdfPages("plots/Semi_Grid_task1_test.pdf")
types = ["accumulating", "etd_int_0.01", "etd_adaptive_int_0.5", "ptd"]

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax = ax.reshape(1,-1)
fig.tight_layout(w_pad=3, h_pad=4)


learn_rate = {"etd_int_0.01":{8:0.007, 12:0.003, 16:0.0003} , "etd_int_0.05":{8:0.001, 12:0.001, 16:0.001} , "etd_int_0.1":{8:0.0007, 16:0.01, 12:0.0003} ,\
        "accumulating":{8:0.001, 12:0.0007, 16:0.0001} , "ptd":{8:0.003, 12:0.001, 16:0.001},\
            "etd_adaptive_int_0.5":{8:0.0003, 12:0.0001, 16:0.00005} }
name = {"etd_int_0.01": "ETD fixed", "etd_int_0.05": "ETD interest: 0.05", "etd_int_0.1": "ETD interest: 0.1",\
        "accumulating": r"TD($\lambda$)", "ptd": "PTD", "etd_adaptive_int_0.5":"ETD variable"}
colors = {"etd_int_0.01": "magenta", "etd_int_0.05": "magenta", "etd_int_0.1": "teal",\
          "accumulating": "green", "ptd": "red", "etd_adaptive_int_0.5":"blue"}
marker = {"etd_int_0.01": "*", "etd_int_0.05":"p", "etd_int_0.1": "^",\
          "accumulating": "x", "ptd": "o", "etd_adaptive_int_0.5":"D"}

row_plt = 0
col_plt = 0
no_labels = []

for c_l in length:
    for t in types:
        seed_error = []
        for se in seed:
            try:
                with open("results_gridWorld/drl_"+t+"_env_"+env+"_size_"+str(c_l)+"_lr_"+str(learn_rate[t][c_l])+"_seed_"+str(se)+"_epi_"+str(epi[c_l])+"_all_errors.pkl", "rb") as f:
                    tmpp = list(pickle.load(f))
                    seed_error.append(tmpp)
            except:
                print("drl_"+t+"_env_"+env+"_size_"+str(c_l)+"_lr_"+str(learn_rate[t][c_l])+"_seed_"+str(se)+"_epi_"+str(epi[c_l]))
        
        mean = np.mean(np.array(seed_error), axis=0)
        std = np.std(np.array(seed_error), axis=0)
        
        n = 1
            
        ax[row_plt, col_plt].plot(range(epi[c_l]-n+1), mean, c=colors[t])
        ax[row_plt, col_plt].fill_between(range(epi[c_l]-n+1), mean+z_star*(std/t_seeds**0.5), mean-z_star*(std/t_seeds**0.5), color=colors[t], alpha=0.3)
        ax[row_plt, col_plt].set_title("Grid size = "+str(c_l), fontsize=14)
        #ax[row_plt, col_plt].set_ylim(0,3)
        ax[row_plt, col_plt].set_xlabel("Episodes", fontsize=16)
        ax[row_plt, col_plt].set_ylabel("MSE on FO states", fontsize=16)
        
        
    col_plt += 1
    if col_plt == 3:
        row_plt = 1
        col_plt = 0

    
fig.legend([name[t] for t in types], loc="right", bbox_to_anchor=(0.5, 0., 0.1, 1.5), prop={'size': 15})#loc="lower center", bbox_to_anchor=(0.4, 0., 0.5, -5), ncol=3)
pdf.savefig(fig, bbox_inches = 'tight')
pdf.close()
    