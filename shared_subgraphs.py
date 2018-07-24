import tensorflow as tf
from tensorbayes import Constant, Placeholder, Dense, GaussianSample, log_bernoulli_with_logits, log_normal, cross_entropy_with_logits, conv2d
import numpy as np
import sys
import configparser

config = configparser.ConfigParser()
config.read('gmvae.ini')
config = config['gmvae_k']

# old_code
# if config.getboolean('conv_layer'):
#             with tf.variable_scope('convl', 'conv', reuse=reuse):
#                 inp = tf.reshape(inp, [-1, 4096 + 6, 3])
#                 conv1 = tf.layers.conv1d(
#                   inputs=inp,
#                   kernel_size=int(config['kernel_size']),
#                   filters=int(config['filters']),
#                   padding="same",
#                   activation=tf.nn.relu)
#                 pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=int(config['pooling_size']), strides=int(config['pooling_stride']))
#                 inp = tf.contrib.layers.flatten(pool1)

# vae subgraphs
def qy_graph(x, k=10):
    reuse = tf.AUTO_REUSE
#     reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='qy')) > 0
    layer_size = int(config['layer_size'])
    # -- q(y)
    with tf.variable_scope('qy'):
        h1 = Dense(x, layer_size, 'layer1', tf.nn.relu, reuse=reuse)
        h2 = Dense(h1, layer_size, 'layer2', tf.nn.relu, reuse=reuse)
        qy_logit = Dense(h2, k, 'logit', reuse=reuse)
        qy = tf.nn.softmax(qy_logit, name='prob')
    return qy_logit, qy

def conv_qy_graph(x, k=10):
#     reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='qy')) > 0
    reuse = tf.AUTO_REUSE
    layer_size = int(config['layer_size'])
    # -- q(y)
    with tf.variable_scope('qy'):
        inp = tf.reshape(x, [-1, 128, 128,3])
        h1 = conv2d(inp, 64, 'layer1', tf.nn.relu, reuse=reuse)
        h2 = conv2d(h1, 128, 'layer2', tf.nn.relu, reuse=reuse)
        h3 = conv2d(h2, 256, 'layer3', tf.nn.relu, reuse=reuse)
        h4 = conv2d(h3, 512, 'layer4', tf.nn.relu, reuse=reuse)
        h5 = conv2d(h4, 1024, 'layer5', tf.nn.relu, reuse=reuse)
        h6 = tf.contrib.layers.flatten(h5)
        qy_logit = Dense(h6, k, 'logit', reuse=reuse)
        qy = tf.nn.softmax(qy_logit, name='prob')
    return qy_logit, qy

def qz_graph(x, y):
    reuse = tf.AUTO_REUSE
#     reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='qz')) > 0
    # -- q(z)
    layer_size = int(config['layer_size'])
    embedding_size = int(config['embedding_size'])
    with tf.variable_scope('qz'):
        xy = tf.concat((x, y), 1, name='xy/concat')
        h1 = Dense(xy, layer_size , 'layer1', tf.nn.relu, reuse=reuse)
        h2 = Dense(h1, layer_size, 'layer2', tf.nn.relu, reuse=reuse)
        zm = Dense(h2, embedding_size, 'zm', reuse=reuse)
        zv = Dense(h2, embedding_size, 'zv', tf.nn.softplus, reuse=reuse)
        z = GaussianSample(zm, zv, 'z')
    return z, zm, zv

def conv_qz_graph(x, y):
#     reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='qz')) > 0
    # -- q(z)
    reuse = tf.AUTO_REUSE
    layer_size = int(config['layer_size'])
    embedding_size = int(config['embedding_size'])
    with tf.variable_scope('qz'):
        inp = tf.reshape(x, [-1, 128, 128,3])
        h1 = conv2d(inp, 64, 'layer1', tf.nn.relu, reuse=reuse)
        h2 = conv2d(h1, 128, 'layer2', tf.nn.relu, reuse=reuse)
        h3 = conv2d(h2, 256, 'layer3', tf.nn.relu, reuse=reuse)
        h4 = conv2d(h3, 512, 'layer4', tf.nn.relu, reuse=reuse)
        h5 = conv2d(h4, 1024, 'layer5', tf.nn.relu, reuse=reuse)
        h6 = tf.contrib.layers.flatten(h5)
        xy = tf.concat((h6, y*1000), 1, name='xy/concat')
        zm = Dense(xy, embedding_size, 'zm', reuse=reuse)
        zv = Dense(xy, embedding_size, 'zv', tf.nn.softplus, reuse=reuse)
        z = GaussianSample(zm, zv, 'z')
    return z, zm, zv

def ql_graph(y):
    reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='ql')) > 0
    layer_size = int(config['layer_size'])
    # -- q(z)
    with tf.variable_scope('ql'):
        h1 = Dense(y, layer_size, 'layer1', tf.nn.relu, reuse=reuse)
        h2 = Dense(h1, layer_size, 'layer2', tf.nn.relu, reuse=reuse)
        lm = Dense(h2, 1, 'lm', reuse=reuse)
        lv = Dense(h2, 1, 'lv', tf.nn.softplus, reuse=reuse)
        l = GaussianSample(lm, lv, 'l')
    return l

def labeled_loss(x, px_logit, z, zm, zv, zm_prior, zv_prior):
    r_loss = (-log_bernoulli_with_logits(x, px_logit)) * float(config['reconstruct_loss_lambda'])
    kl_loss = (log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)) * float(config['kl_loss_lambda'])
#     return xy_loss - np.log(0.1)
    return (r_loss, kl_loss)

def labeled_loss_real(x, px_logit, xv, z, zm, zv, zm_prior, zv_prior):
    r_loss = (-log_normal(x, tf.sigmoid(px_logit), tf.clip_by_value(xv, 0.1, 1))) * float(config['reconstruct_loss_lambda'])
    kl_loss = (log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)) * float(config['kl_loss_lambda'])
#     return xy_loss - np.log(0.1)
    return (r_loss, kl_loss)

def labeled_loss_custom(x, x_reconstruct, xv, z, zm, zv, zm_prior, zv_prior):
    r_loss = (-log_normal(x, x_reconstruct, tf.clip_by_value(xv, 0.1, 1))) * float(config['reconstruct_loss_lambda'])
    kl_loss = (log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)) * float(config['kl_loss_lambda'])
#     return xy_loss - np.log(0.1)
    return (r_loss, kl_loss)

def triplet_loss(z, qy, k):
    alpha = float(config['tl_margin'])
    z_normalized = tf.nn.l2_normalize(z, 2)
    a, p, n = tf.split(z_normalized, 3, axis=1)
    a_qy, p_qy, n_qy = tf.split(qy, 3, axis=0)
    
    out = 0.0
    
    for i in range(k):
        for j in range(k):
            for z in range(k):
                out = tf.add(out, a_qy[:, i] * p_qy[:, j] * n_qy[:, z] * tf.reduce_sum(tf.maximum(0.0, alpha + tf.multiply(a[i], n[z]) - 
                     tf.multiply(a[i], p[j])), axis=1))
                
    return out
#     return tf.add_n([a_qy[:, i] * p_qy[:, j] * n_qy[:, z] * tf.reduce_sum(tf.maximum(0.0, alpha + tf.multiply(a[i], n[z]) - 
#                      tf.multiply(a[i], p[j])), axis=1) for i in range(k) for j in range(k) for z in range(k)])
