from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf


def class_wise_pooling(x):
    m = 8
    _, _, _, n = x.get_shape().as_list()
    n_classes = n // m
    ops = []
    for i in range(n_classes):
        class_avg_op = tf.reduce_mean(x[:, :, :, :], axis=3, keep_dims=True)
        ops.append(class_avg_op)
    final_op = tf.concat(ops, axis=3)
    return final_op


def spatial_pooling(x):
    k = 1
    alpha = 0.7

    batch_size, w, h, n_classes = x.get_shape().as_list()
    x_flat = tf.reshape(x, shape=(-1, w * h, n_classes))
    x_transp = tf.transpose(x_flat, perm=(0, 2, 1))
    k_maxs, _ = tf.nn.top_k(x_transp, k, sorted=False)
    k_maxs_mean = tf.reduce_mean(k_maxs, axis=2)
    result = k_maxs_mean
    if alpha:
        # top -x_flat to retrieve the k smallest values
        k_mins, _ = tf.nn.top_k(-x_transp, k, sorted=False)
        # flip back
        k_mins = -k_mins
        k_mins_mean = tf.reduce_mean(k_mins, axis=2)
        alpha = tf.constant(alpha, name='alpha', dtype=tf.float32)
        result += alpha * k_mins_mean
    return result
