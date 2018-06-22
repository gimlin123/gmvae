import tensorflow as tf
import pickle
from tensorflow.examples.tutorials.mnist import input_data
from tensorbayes import Constant, Placeholder, Dense, GaussianSample, log_bernoulli_with_logits, log_normal, cross_entropy_with_logits, show_graph, progbar
import numpy as np
import sys
from shared_subgraphs import qy_graph, qz_graph, ql_graph
from custom_utils import train_custom

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
custom_data = pickle.load(open("custom_data/linear.p", "rb" ))

try:
    k = int(sys.argv[1])
except IndexError:
    raise Exception('Pass an argument specifying the number of mixture components\n'
                    'e.g. python gmvae_k.py 10')

def mean_squared_error(labels, predictions):
    return tf.reduce_mean(tf.squared_difference(labels, predictions))


def labeled_loss(xr, x_reconstruct, xv, z, zm, zv, zm_prior, zv_prior, l, l_predict):
    with tf.name_scope('reconstruction_loss'):
        reconstruction_loss = -log_normal(xr*100, x_reconstruct*100, xv*100**2) / 5
    with tf.name_scope('divergence_loss'):
        divergence_loss = (log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior))
    xy_loss = reconstruction_loss
    xy_loss += divergence_loss
    return xy_loss - np.log(0.1)

def px_graph(z, y):
    reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='px')) > 0
    # -- p(z)
    with tf.variable_scope('pz'):
        # zm = Dense(y, 64, 'zm', reuse=reuse)
        # zv = Dense(y, 64, 'zv', tf.nn.softplus, reuse=reuse)
        zm = Dense(y, 50, 'zm', reuse=reuse)
        zv = Dense(y, 50, 'zv', tf.nn.softplus, reuse=reuse)
    # -- p(x)
    with tf.variable_scope('px'):
        h1 = Dense(z, 512, 'layer1', tf.nn.relu, reuse=reuse)
        h2 = Dense(h1, 512, 'layer2', tf.nn.relu, reuse=reuse)
        # px_logit = Dense(h2, 784, 'logit', reuse=reuse)
        x_reconstruct = Dense(h2, 300, 'logit', reuse=reuse)
        xv = Dense(h2, 300, 'xv', tf.nn.softplus, reuse=reuse)
    return zm, zv, x_reconstruct, xv

tf.reset_default_graph()
# x = Placeholder((None, 784), name='x')
# changing for custom stuff
x = Placeholder((None, 300), name='x')
l = Placeholder((None, None), name='l')

# binarize data and create a y "placeholder"
with tf.name_scope('x_reducedv'):
    xr = x / 100.
#create a y "placeholder"
with tf.name_scope('y_'):
    y_ = tf.fill(tf.pack([tf.shape(x)[0], k]), 0.0)

# propose distribution over y
qy_logit, qy = qy_graph(xr, k)

# for each proposed y, infer z and reconstruct x and l
z, zm, zv, zm_prior, zv_prior, x_reconstruct, l_predict, xv = [[None] * k for i in range(8)]
for i in range(k):
    with tf.name_scope('graphs/hot_at{:d}'.format(i)):
        y = tf.add(y_, Constant(np.eye(k)[i], name='hot_at_{:d}'.format(i)))
        z[i], zm[i], zv[i] = qz_graph(xr, y)
        l_predict[i] = ql_graph(y)
        zm_prior[i], zv_prior[i], x_reconstruct[i], xv[i] = px_graph(z[i], y)

# Aggressive name scoping for pretty graph visualization :P
with tf.name_scope('loss'):
    with tf.name_scope('neg_entropy'):
        nent = -cross_entropy_with_logits(qy_logit, qy)
    losses = [None] * k
    label_losses = [None] * k
    for i in range(k):
        with tf.name_scope('loss_at{:d}'.format(i)):
            losses[i] = labeled_loss(xr, x_reconstruct[i], xv[i], z[i], zm[i], zv[i], zm_prior[i], zv_prior[i], l, l_predict[i])
        with tf.name_scope('label_loss_at{:d}'.format(i)):
            label_losses[i] = mean_squared_error(l, l_predict[i])
    with tf.name_scope('label_loss'):
        label_loss = tf.add_n([qy[:, i] * label_losses[i] for i in range(k)])
    with tf.name_scope('final_loss'):
        loss = tf.add_n([nent, label_loss] + [qy[:, i] * losses[i] for i in range(k)])

train_step = tf.train.AdamOptimizer().minimize(loss)
sess = tf.Session()
# sess.run(tf.initialize_all_variables())
sess.run(tf.global_variables_initializer()) # Change initialization protocol depending on tensorflow version
sess_info = (sess, qy_logit, nent, loss, train_step, x, x_reconstruct)
train_custom('logs/custom_gmvae_k={:d}.log'.format(k), custom_data, sess_info, epochs=1000)
