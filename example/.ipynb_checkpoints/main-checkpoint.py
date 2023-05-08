import os
import yaml
from pathlib import Path
import numpy as np

from train import train_model
from train import setup_seed
from sklearn.model_selection import train_test_split
from eval import evaluate_vadesc, evaluate_coxPH, evaluate_kmeans

project_dir = os.path.dirname(os.getcwd())
config_path = Path(os.path.join(project_dir, 'configs/test.yml'))
with config_path.open(mode='r') as yamlfile:
    configs = yaml.safe_load(yamlfile)
print(configs)

seed = 20220229
setup_seed(seed)

GeneCount = np.load('../processedData/GeneCount.npy',allow_pickle=True)
TTE = np.load('../processedData/TTE.npy',allow_pickle=True)
EVENT = np.load('../processedData/Event.npy',allow_pickle=True)

X_train = GeneCount
Y_train = np.stack([TTE,EVENT],axis=1)

evaluate_kmeans(X_train,Y_train)

X_train,X_test,Y_train,Y_test=train_test_split(X_train,Y_train,test_size=0.2,shuffle=True,random_state=44)

model = train_model(configs, X_train, Y_train, X_test, Y_test, seed)

model.save_weights('./model_weights/weights.h5')

evaluate_vadesc(model, X_train, Y_train, seed)
evaluate_coxPH()