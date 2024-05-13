import sys
import os
import yaml

from pathlib import Path
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from model import GMM_Survival
from utils import get_gen
from loss import Losses
from plot import plot_bigroup_kaplan_meier

if __name__ == "__main__":
    project_dir = os.path.dirname(os.getcwd())
    config_path = Path(os.path.join(project_dir, '../configs/config.yml'))
    with config_path.open(mode='r') as yamlfile:
        configs = yaml.safe_load(yamlfile)
    print(configs)

    losses = Losses(configs)
    rec_loss = losses.loss_reconstruction_mse

    #load TCGA data
    GeneCount = np.load('../../processedData/GeneCount.npy',allow_pickle=True) 
    TTE = np.load('../../processedData/TTE.npy',allow_pickle=True)
    EVENT = np.load('../../processedData/Event.npy',allow_pickle=True)

    X_train = GeneCount
    Y_train = np.stack([TTE,EVENT],axis=1)

    X_train,X_test,Y_train,Y_test=train_test_split(X_train,Y_train,test_size=0.2,shuffle=True, random_state=44)

    # Construct the model & optimizer 
    model = GMM_Survival(**configs['training'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer, loss={"output_1": rec_loss})

    # Use survival times during training
    tf.keras.backend.set_value(model.use_t, np.array([1.0]))

    # Data generators
    gen_train = get_gen(X_train, Y_train,configs, 16)  
    gen_test = get_gen(X_test, Y_test, configs, 16, validation=True)

    # Fit the model
    model.fit(gen_train,validation_data=gen_test, epochs=11, verbose=1)
    model.summary()

    tf.keras.backend.set_value(model.use_t, np.array([0.0]))
    model.sample_surv = False
    rec, z_sample, p_z_c, p_c_z, risk_scores, lambdas = model.predict((X_train, Y_train), batch_size=16)
    risk_scores = np.squeeze(risk_scores)
    
    c_hat = p_c_z            

    plot_bigroup_kaplan_meier(t=Y_train[:,0], d=Y_train[:,1], c_probs=c_hat, legend=True, legend_outside=True)

    #Heatmap plotting by cluster of genes data  
    grid_size = 2
    inp_size =[11684]

    for j in range(model.num_clusters):
        samples = model.generate_samples(j=j, n_samples=grid_size**2)
        cnt = 0
        img = None
        for k in range(grid_size):
            row_k = []
            for l in range(grid_size):
                row_k.append(np.reshape(samples[0, cnt, :], (inp_size[0])))
                cnt = cnt + 1

    plt.figure(figsize=(20, 2))
    ax = sns.heatmap(row_k,cmap='viridis')
    plt.xlabel('Genes')
    plt.ylabel('Clusters')
    plt.title('Heatmap of Average Gene Expression by Cluster')
    plt.show()