# Task 2 - elevator

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.backends.backend_pdf

# elevator

# elevator

def moving_average(a, n=3) :
    return np.convolve(a, np.ones((n,))/n, mode='valid')

n=1

plt.style.use('seaborn-white')

t_seeds=25
seed = list(range(t_seeds))
length = range(5,30,5)
env = "elevator"
epi = 200
z_star = 1.96 # 0.674-50% 1.28-80%, 1.645-90%, 1.96-95%, 2.33-98%, 2.58-99%
pdf = matplotlib.backends.backend_pdf.PdfPages("plots/task2_learning_curve_test.pdf")
types = ["etrace", "etd_int_0.01", "etd_adaptive_int_0.5", "ptd"]

#fig, ax = plt.subplots(2, 3, figsize=(10, 6))
#fig.delaxes(ax[1,2])
fig, ax = plt.subplots(1, 5, figsize=(18, 3.5))
ax = ax.reshape(1,-1)


learn_rate = {"etd_adaptive_int_0.5":{5:0.1, 10:0.1, 15:0.1, 20:0.08, 25:0.08},\
              "etd_int_0.01":{5:0.8, 10:0.8, 15:0.8, 20:0.8, 25:0.8},\
              "etrace":{5:0.05, 10:0.01, 15:0.01, 20:0.004, 25:0.004} ,\
              "ptd":{5:0.1, 10:0.1, 15:0.1, 20:0.08, 25:0.08} }
name = {"etd_int_0.01": "ETD fixed", "etd_int_0.05": "ETD interest: 0.05", "etd_int_0.1": "ETD interest: 0.1",\
        "etrace": r"TD($\lambda$)", "ptd": "PTD", "etd_adaptive_int_0.5":"ETD variable"}
colors = {"etd_int_0.01": "magenta", "etd_int_0.05": "magenta", "etd_int_0.1": "teal",\
          "etrace": "green", "ptd": "red", "etd_adaptive_int_0.5":"blue"}
marker = {"etd_int_0.01": "*", "etd_int_0.05":"p", "etd_int_0.1": "^",\
          "etrace": "x", "ptd": "o", "etd_adaptive_int_0.5":"D"}

row_plt = 0
col_plt = 0
no_labels = []

for c_l in length:
    for t in types:
        seed_error = []
        for se in seed:
            try:
                with open("results_elevator/"+t+"_env_"+env+"_len_"+str(c_l)+"_lr_"+str(learn_rate[t][c_l])+"_seed_"+str(se)+"_all_errors.pkl", "rb") as f:
                    tmpp = list(pickle.load(f))
                    seed_error.append(tmpp)
            except:
                pass
        
        mean = np.mean(np.array(seed_error), axis=0)
        std = np.std(np.array(seed_error), axis=0)    
        ax[row_plt, col_plt].plot(range(epi-n+1), mean, c=colors[t])
        ax[row_plt, col_plt].fill_between(range(epi-n+1), mean+z_star*(std/t_seeds**0.5), mean-z_star*(std/t_seeds**0.5), color=colors[t], alpha=0.3)
        ax[row_plt, col_plt].set_title("Length = "+str(c_l), fontsize=14)
        ax[row_plt, col_plt].set_xlabel("Episodes", fontsize=14)
        ax[row_plt, col_plt].set_ylabel("MSE on FO states", fontsize=14)
        
        
    col_plt += 1
    if col_plt == 5:
        row_plt = 1
        col_plt = 0

ax[0, 4].set_ylim(top=4.5)   
fig.legend([name[t] for t in types], loc="right", ncol=4, bbox_to_anchor=(0.3, 0., 0.4, 0.04), prop={'size': 14}, frameon=False)
fig.tight_layout(w_pad=3, h_pad=4)
pdf.savefig(fig, bbox_inches = 'tight')
pdf.close()
