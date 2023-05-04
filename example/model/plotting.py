import os

import numpy as np

from lifelines import KaplanMeierFitter

import matplotlib.pyplot as plt

import sys

sys.path.insert(0, '../')

CB_COLOR_CYCLE = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
GRAY_COLOR_CYCLE = ['black', 'dimgray', 'darkgray', 'gainsboro', 'whitesmoke']
LINE_TYPES = ['solid', 'dashed', 'dashdot', 'dotted', 'dashed']
MARKER_STYLES = ['', '', '', '', '']
DASH_STYLES = [[], [4, 4], [4, 1], [1, 1, 1], [2, 1, 2]]


#plotting kmf each clusters
def plot_bigroup_kaplan_meier(t, d,c_, dir=None, postfix=None, legend=False, legend_outside=False):
    labels = np.unique(c_)
    for l in labels:
        kmf = KaplanMeierFitter()
        kmf.fit(t[c_ == l], d[c_ == l], label="cluster " + str(int(l + 1)))
        kmf.plot(ci_show=True, alpha=0.75, color=CB_COLOR_CYCLE[int(l)], linewidth=2)

    plt.xlabel("Time")
    plt.ylabel("Survival Probability")