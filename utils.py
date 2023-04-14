"""
miscellaneous utility functions.
"""
import matplotlib

import sys

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras

matplotlib.use('Agg')

sys.path.insert(0, '../../')

# Weibull(lmbd, k) log-pdf
def weibull_log_pdf(t, d, lmbd, k):
    t_ = tf.ones_like(lmbd) * tf.cast(t, tf.float64)
    d_ = tf.ones_like(lmbd) * tf.cast(d, tf.float64)
    k = tf.cast(k, tf.float64)
    a = t_ / (1e-60 + tf.cast(lmbd, tf.float64))
    tf.debugging.check_numerics(a, message="weibull_log_pdf")

    return tf.cast(d_, tf.float64) * (tf.math.log(1e-60 + k) - tf.math.log(1e-60 + tf.cast(lmbd, tf.float64)) +
                                     (k - 1) * tf.math.log(1e-60 + tf.cast(t_, tf.float64)) - (k - 1) *
                                     tf.math.log(1e-60 + tf.cast(lmbd, tf.float64))) - (a) ** k


def weibull_scale(x, beta):
    beta_ = tf.cast(beta, tf.float64)
    beta_ = tf.cast(tf.ones([tf.shape(x)[0], tf.shape(x)[1], beta.shape[0]]), tf.float64) * beta_
    return tf.clip_by_value(tf.math.log(1e-60 + 1.0 + tf.math.exp(tf.reduce_sum(-tf.cast(x, tf.float64) * beta_[:, :, :-1], axis=2) -
                                                 tf.cast(beta[-1], tf.float64))), -1e+64, 1e+64)


def tensor_slice(target_tensor, index_tensor):
    indices = tf.stack([tf.range(tf.shape(index_tensor)[0]), index_tensor], 1)
    return tf.gather_nd(target_tensor, indices)