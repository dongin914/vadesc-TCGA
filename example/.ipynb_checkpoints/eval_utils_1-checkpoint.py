import numpy as np

import sys

import tensorflow as tf

sys.path.insert(0, '../')

def cindex_metric(inp, risk_scores):
    # Evaluates the concordance index based on provided predicted risk scores, computed using hard clustering
    # assignments.
    t = inp[:, 0]
    d = inp[:, 1]
    risk_scores = tf.squeeze(risk_scores)
    return tf.cond(tf.reduce_any(tf.math.is_nan(risk_scores)),
                   lambda: tf.numpy_function(cindex, [t, d, tf.zeros_like(risk_scores)], tf.float64),
                   lambda: tf.numpy_function(cindex, [t, d, risk_scores], tf.float64))

