import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical

class DataGen(tf.keras.utils.Sequence):

    def __init__(self, X, y, num_classes, ae=False, ae_class=False, batch_size=32, shuffle=True, augment=False):
        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.ae = ae
        self.ae_class = ae_class
        self.num_classes = num_classes
        self.augment = augment
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            inds = np.arange(len(self.X))
            np.random.shuffle(inds)
            self.X = self.X[inds]
            self.y = self.y[inds]

    def __getitem__(self, index):
        X = self.X[index * self.batch_size:(index + 1) * self.batch_size]
        y = self.y[index * self.batch_size:(index + 1) * self.batch_size]
        # augmentation
        if self.augment:
            X = augment_images(X)
        if self.ae:
            return X, {'dec': X}
        elif self.ae_class:
            c = to_categorical(y[:, 2], self.num_classes)
            return X, {'dec': X, 'classifier': c}
        else:
            return (X, y), {"output_1": X, "output_4": y, "output_5": y}

    def __len__(self):
        return len(self.X) // self.batch_size


def get_gen(X, y, configs, batch_size, validation=False, ae=False, ae_class=False):
    num_clusters = 2
    input_dim = 2
    if isinstance(input_dim, list) and validation==False:
        if ae_class:
            data_gen = DataGen(X, y, 4, augment=True, ae=ae, ae_class=ae_class, batch_size=batch_size)
        else:
            data_gen = DataGen(X, y, num_clusters, augment=True, ae=ae, ae_class=ae_class, batch_size=batch_size)
    else:
        if ae_class:
            data_gen = DataGen(X, y, 4, ae=ae, ae_class=ae_class, batch_size=batch_size)
        else:
            data_gen = DataGen(X, y, num_clusters, ae=ae, ae_class=ae_class, batch_size=batch_size)
    return data_gen


