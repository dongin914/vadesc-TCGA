import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

CB_COLOR_CYCLE = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

def plot_bigroup_kaplan_meier(t, d, c_probs, legend=False, legend_outside=False):
    labels = range(c_probs.shape[1])
    for l in labels:
        kmf = KaplanMeierFitter()
        weights = c_probs[:, l]
        kmf.fit(t, d, weights=weights, label=f"Cluster {l+1}")
        kmf.plot(ci_show=True, alpha=0.75, color=CB_COLOR_CYCLE[l], linewidth=2)

    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title("vadesc_TCGA")
    plt.legend(title='Clusters')
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)
