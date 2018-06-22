import tensorflow as tf
from tensorbayes import Constant, Placeholder, Dense, GaussianSample, log_bernoulli_with_logits, log_normal, cross_entropy_with_logits
import numpy as np
import sys
import configparser

config = configparser.ConfigParser()
config.read('gmvae.ini')
config = config['gmvae_k']

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
    xy_loss = (-log_bernoulli_with_logits(x, px_logit)) * float(config['reconstruct_loss_lambda'])
    xy_loss += (log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)) * float(config['kl_loss_lambda'])
    return xy_loss - np.log(0.1)

def labeled_loss_real(x, px_logit, xv, z, zm, zv, zm_prior, zv_prior):
    xy_loss = (-log_normal(x, tf.sigmoid(px_logit), tf.clip_by_value(xv, 0.1, 1))) * float(config['reconstruct_loss_lambda'])
    xy_loss += (log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)) * float(config['kl_loss_lambda'])
    return xy_loss - np.log(0.1)

def labeled_loss_custom(x, x_reconstruct, xv, z, zm, zv, zm_prior, zv_prior):
    xy_loss = (-log_normal(x, x_reconstruct, tf.clip_by_value(xv, 0.1, 1))) * float(config['reconstruct_loss_lambda'])
    xy_loss += (log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)) * float(config['kl_loss_lambda'])
    return xy_loss - np.log(0.1)

def triplet_loss(z, qy, k):
    alpha = float(config['tl_margin'])
    z_normalized = tf.nn.l2_normalize(z, 2)
    a, p, n = tf.split(1, 3, z_normalized)
    a_qy, p_qy, n_qy = tf.split(0, 3, qy)

    return tf.add_n([a_qy[:, i] * p_qy[:, j] * n_qy[:, z] * tf.reduce_sum(tf.maximum(0.0, alpha + tf.multiply(a[i], n[z]) - tf.multiply(a[i], p[j])), axis=1)
      for i in xrange(k) for j in xrange(k) for z in xrange(k)])
