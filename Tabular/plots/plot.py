import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.backends.backend_pdf

np.seterr(divide='ignore', invalid='ignore')

# 19-State Random Walk

plt.style.use('seaborn-white')

seed = list(range(25))
intrst = [0.01]
lamb = [0.0, 0.1, 0.2, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1.0]
env = "randomWalk"
ci = 0.
pdf = matplotlib.backends.backend_pdf.PdfPages("plots/lr_19stateRW_hp_tuning_test.pdf")
types = ["etd", "ptd", "etrace"]

fig, ax = plt.subplots(1, 3, figsize=(10, 3.3))
ax = ax.reshape(1,-1)
#fig.delaxes(ax[1,2])

learn_rate = [3.0, 2.5, 2.2, 1.75, 1.3, 1.0, 0.8, 0.5, 0.3, 0.1, 0.08, 0.05,\
              0.03, 0.01, 0.007, 0.003, 0.001, 0.0007, 0.0003, 0.0001, 0.00005]
              

name = {"etd": "Emphatic TD", "etrace": r"TD($\lambda$)", "ptd": "Preferential TD"}
colors = {"etd": "blue", "etrace": "green", "ptd": "red"}
marker = {"etd": "o", "etrace": "o", "ptd": "o"}

row_plt = 0
col_plt = 0
no_labels = []

for t in types:
    if t == "etd":
        for itrst in intrst:
            for lam in lamb:
                lr_list = []
                avg_error_list = []
                std_error_list = []
                for lr in learn_rate:
                    seed_error = []
                    for se in seed:
                        flag = False
                        try:
                            with open("results_randomWalk/"+t+"_env_"+env+"_intrst_"+str(itrst)+"_lamb_"+str(lam)+"_lr_"+str(lr)+"_seed_"+str(se)+"_all_errors.pkl", "rb") as f:
                                tmpp = list(pickle.load(f))
                                tmp_avg = np.mean(np.sqrt(np.array(tmpp)))
                                if tmp_avg > 10 or np.isnan(tmp_avg):
                                    tmp_avg = 10
                                    flag = True
                                seed_error.append(tmp_avg)
                        except:
                            continue

                    mean = np.mean(np.array(seed_error))
                    avg_error_list.append(mean)
                    if flag:
                        std_error_list.append(np.zeros_like(mean))
                    else:
                        std_error_list.append(np.std(np.array(seed_error)))
                    lr_list.append(lr)

                ax[row_plt, col_plt].plot(lr_list, avg_error_list, marker=marker[t], markersize=4)
                ax[row_plt, col_plt].fill_between(lr_list, np.array(avg_error_list)+ci*np.array(std_error_list),\
                                          np.array(avg_error_list)-ci*np.array(std_error_list), alpha=0.3)
                #ax[col_plt].set(xscale="log")
                ax[row_plt, col_plt].set_title(name[t]+" (interest: "+str(itrst)+")", fontsize=14)
                ax[row_plt, col_plt].set_ylim(0.2,0.6)
                if itrst == 0.25:
                    ax[row_plt, col_plt].set_xlim(0.2,0.6)
                ax[row_plt, col_plt].set_xlabel("Learning rate", fontsize=14)
                ax[row_plt, col_plt].set_ylabel("Mean RMSE over 10 episodes", fontsize=12)
            col_plt += 1
            if col_plt == 3:
                row_plt = 1
                col_plt = 0
                
    else:
        for lam in lamb:
            lr_list = []
            avg_error_list = []
            std_error_list = []
            for lr in learn_rate:
                seed_error = []
                for se in seed:
                    flag = False
                    try:
                        with open("results_randomWalk/"+t+"_env_"+env+"_lamb_"+str(lam)+"_lr_"+str(lr)+"_seed_"+str(se)+"_all_errors.pkl", "rb") as f:
                            tmpp = list(pickle.load(f))
                            tmp_avg = np.mean(np.sqrt(np.array(tmpp)))
                            if tmp_avg > 10 or np.isnan(tmp_avg):
                                tmp_avg = 10
                                flag = True
                            seed_error.append(tmp_avg)
                    except:
                        continue

                mean = np.mean(np.array(seed_error))
                avg_error_list.append(mean)
                if flag:
                    std_error_list.append(np.zeros_like(mean))
                else:
                    std_error_list.append(np.std(np.array(seed_error)))
                lr_list.append(lr)

            ax[row_plt, col_plt].plot(lr_list, avg_error_list, marker=marker[t], markersize=4)
            ax[row_plt, col_plt].fill_between(lr_list, np.array(avg_error_list)+ci*np.array(std_error_list),\
                                      np.array(avg_error_list)-ci*np.array(std_error_list), alpha=0.3)
            ax[row_plt, col_plt].set_title(name[t], fontsize=14)
            ax[row_plt, col_plt].set_ylim(0.2,0.6)
            if t == "etrace":
                ax[row_plt, col_plt].set_xlim(0,1.5)
            ax[row_plt, col_plt].set_xlabel("Learning rate", fontsize=14)
            ax[row_plt, col_plt].set_ylabel("Mean RMSE over 10 episodes", fontsize=12)
        
        col_plt += 1
        if col_plt == 3:
            row_plt = 1
            col_plt = 0

plt.legend(lamb, ncol=1, loc="right",bbox_to_anchor=(0.5, 0., 1., 1), frameon=True) # bbox_to_anchor=(0.33, 0.15, 0.4, 0.)
fig.tight_layout(w_pad=2, h_pad=2)
pdf.savefig(fig, bbox_inches = 'tight')
pdf.close()
