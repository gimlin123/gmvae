import tensorflow as tf
from tensorbayes import Constant, Placeholder, Dense, GaussianSample, log_bernoulli_with_logits, log_normal, cross_entropy_with_logits
import numpy as np
import sys

# vae subgraphs
def qy_graph(x, k=10):
    reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='qy')) > 0
    # -- q(y)
    with tf.variable_scope('qy'):
        h1 = Dense(x, 512, 'layer1', tf.nn.relu, reuse=reuse)
        h2 = Dense(h1, 512, 'layer2', tf.nn.relu, reuse=reuse)
        qy_logit = Dense(h2, k, 'logit', reuse=reuse)
        qy = tf.nn.softmax(qy_logit, name='prob')
    return qy_logit, qy

def qz_graph(x, y):
    reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='qz')) > 0
    # -- q(z)
    with tf.variable_scope('qz'):
        xy = tf.concat(1, (x, y), name='xy/concat')
        h1 = Dense(xy, 512, 'layer1', tf.nn.relu, reuse=reuse)
        h2 = Dense(h1, 512, 'layer2', tf.nn.relu, reuse=reuse)
        zm = Dense(h2, 64, 'zm', reuse=reuse)
        zv = Dense(h2, 64, 'zv', tf.nn.softplus, reuse=reuse)
        z = GaussianSample(zm, zv, 'z')
    return z, zm, zv

def ql_graph(y):
    reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='ql')) > 0
    # -- q(z)
    with tf.variable_scope('ql'):
        h1 = Dense(y, 512, 'layer1', tf.nn.relu, reuse=reuse)
        h2 = Dense(h1, 512, 'layer2', tf.nn.relu, reuse=reuse)
        lm = Dense(h2, 1, 'lm', reuse=reuse)
        lv = Dense(h2, 1, 'lv', tf.nn.softplus, reuse=reuse)
        l = GaussianSample(lm, lv, 'l')
    return l

def labeled_loss(x, px_logit, z, zm, zv, zm_prior, zv_prior):
    xy_loss = -log_bernoulli_with_logits(x, px_logit)
    xy_loss += (log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior))
    return xy_loss - np.log(0.1)
