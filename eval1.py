import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.cluster import KMeans
from plotting import plot_bigroup_kaplan_meier
from train import setup_seed

import tensorflow as tf
 
def ensure_results_folder_exists():
    if not os.path.exists('results'):
        os.makedirs('results')

def evaluate_vadesc(model, X_train, Y_train, seed):
    setup_seed(seed)

    tf.keras.backend.set_value(model.use_t, np.array([0.0]))
    model.sample_surv = False

    rec, z_sample, p_z_c, p_c_z, risk_scores, lambdas = model.predict((X_train, Y_train), batch_size=16)
    risk_scores = np.squeeze(risk_scores)
    c_hat = np.argmax(p_c_z, axis=-1)

    plot_bigroup_kaplan_meier(t=Y_train[:,0], d=Y_train[:,1], c_=c_hat, legend=True, legend_outside=True)
    ensure_results_folder_exists()
    plt.savefig('./results/vadesc.png')
    plt.close()


def evaluate_coxPH():
    with open('./processedData/TotalData1.pkl', 'rb') as f:
        Totaldata = pickle.load(f)
    data = Totaldata.iloc[:, 2:]
    cph = CoxPHFitter()
    cph.fit(data, duration_col='time', event_col='event')

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(cph.predict_survival_function(data).T)

    np.unique(kmeans.labels_, return_counts=True)

    cluster_labels = kmeans.labels_
    kmf = KaplanMeierFitter()

    plt.figure()

    for cluster in np.unique(cluster_labels):
        idx = cluster_labels == cluster
        kmf.fit(data.loc[idx, 'time'], event_observed=data.loc[idx, 'event'], label=f'Cluster {cluster}')
        kmf.plot(ci_show=False)

    plt.title('Kaplan-Meier Survival Curves by K-means Cluster')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.legend()
    ensure_results_folder_exists()
    plt.savefig('./results/coxph.png')
    plt.close()

def evaluate_kmeans(X_train, Y_train):
    
    data = X_train

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X_train)

    # Get the cluster labels for each sample
    labels = kmeans.labels_
    survival_curves = []

    for label in np.unique(labels):
        cluster_indices = np.where(labels == label)

        tte = Y_train[:,0]
        event = Y_train[:,1]
        # Fit the Kaplan-Meier estimator to the data in the current cluster
        kmf = KaplanMeierFitter()
        kmf.fit(tte[cluster_indices], event[cluster_indices], label=f"Cluster {label}")

        # Add the fitted Kaplan-Meier curve to the list of survival curves
        survival_curves.append(kmf)

    # Plot the survival curves for each cluster
    for kmf in survival_curves:
        kmf.plot()

    plt.title("Survival Curve by cluster")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    ensure_results_folder_exists()
    plt.savefig('./results/kmeans.png')
    plt.close()