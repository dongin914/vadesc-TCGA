import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

from model.data_utils import DataGen, get_gen
from model.model import GMM_Survival
from model.losses import Losses
from model.eval_utils import cindex_metric
from sklearn.model_selection import train_test_split

def setup_seed(seed):
    random.seed(seed)  
    np.random.seed(seed) 
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def save_weights(model, weights_path):
    model.save_weights(weights_path)

def load_weights(model, weights_path):
    model.load_weights(weights_path)

def train_model(configs, X_train, Y_train, X_test, Y_test, seed):
    setup_seed(seed)

    losses = Losses(configs)
    rec_loss = losses.loss_reconstruction_mse

    model = GMM_Survival(**configs['training'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer, loss={"output_1": rec_loss},
                  metrics={"output_2": cindex_metric})

    tf.keras.backend.set_value(model.use_t, np.array([1.0]))

    gen_train = get_gen(X_train, Y_train, configs, 16)
    gen_test = get_gen(X_test, Y_test, configs, 16, validation=True)

    model.fit(gen_train, validation_data=gen_test, epochs=55, verbose=1)
    model.summary()

    return model
