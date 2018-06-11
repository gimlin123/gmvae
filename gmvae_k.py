import tensorflow as tf
import pickle
from tensorflow.examples.tutorials.mnist import input_data
from tensorbayes import Constant, Placeholder, Dense, GaussianSample, log_bernoulli_with_logits, log_normal, cross_entropy_with_logits, show_graph, progbar
import numpy as np
import sys
from shared_subgraphs import qy_graph, qz_graph, labeled_loss
from custom_utils import train_custom
from utils import train

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# custom_data = pickle.load(open("custom_data/low_variance.p", "rb" ))

try:
    k = int(sys.argv[1])
except IndexError:
    raise Exception('Pass an argument specifying the number of mixture components\n'
                    'e.g. python gmvae_k.py 10')

def generate_n_images(clus, k, n):
    y = tf.fill(tf.pack([1, k]), 0.0)
    y = tf.add(y, Constant(np.eye(k)[clus]))

    z_dummy = tf.fill(tf.pack([1, 64]), 0.0)
    zm, zv, _ = px_graph(z_dummy, y)

    z = GaussianSample(zm, zv, 'z')
    for i in range(n-1):
        z = tf.concat(0, [z, GaussianSample(zm, zv, 'z' + str(i))])

    _, __, x_reconstruct = px_graph(z, y)
    x_reconstruct_b = tf.cast(tf.greater(x_reconstruct, tf.zeros(tf.shape(x_reconstruct))), tf.uint8) * 255
    return x_reconstruct_b

def px_graph(z, y):
    reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='px')) > 0
    # -- p(z)
    with tf.variable_scope('pz'):
        zm = Dense(y, 64, 'zm', reuse=reuse)
        zv = Dense(y, 64, 'zv', tf.nn.softplus, reuse=reuse)
    # -- p(x)
    with tf.variable_scope('px'):
        h1 = Dense(z, 512, 'layer1', tf.nn.relu, reuse=reuse)
        h2 = Dense(h1, 512, 'layer2', tf.nn.relu, reuse=reuse)
        # px_logit = Dense(h2, 784, 'logit', reuse=reuse)
        px_logit = Dense(h2, 784, 'logit', reuse=reuse)
    return zm, zv, px_logit

tf.reset_default_graph()
# x = Placeholder((None, 784), name='x')
# changing for custom stuff
x = Placeholder((None, 784), name='x')
l = Placeholder((None, None), name='l')


# binarize data and create a y "placeholder"
with tf.name_scope('x_binarized'):
    xb = tf.cast(tf.greater(x, tf.random_uniform(tf.shape(x), 0, 1)), tf.float32)
with tf.name_scope('y_'):
    y_ = tf.fill(tf.pack([tf.shape(x)[0], k]), 0.0)

# propose distribution over y
qy_logit, qy = qy_graph(xb, k)

# for each proposed y, infer z and reconstruct x
z, zm, zv, zm_prior, zv_prior, px_logit = [[None] * k for i in xrange(6)]
for i in xrange(k):
    with tf.name_scope('graphs/hot_at{:d}'.format(i)):
        y = tf.add(y_, Constant(np.eye(k)[i], name='hot_at_{:d}'.format(i)))
        z[i], zm[i], zv[i] = qz_graph(xb, y)
        zm_prior[i], zv_prior[i], px_logit[i] = px_graph(z[i], y)

# Aggressive name scoping for pretty graph visualization :P
with tf.name_scope('loss'):
    with tf.name_scope('neg_entropy'):
        nent = -cross_entropy_with_logits(qy_logit, qy)
    losses = [None] * k
    for i in xrange(k):
        with tf.name_scope('loss_at{:d}'.format(i)):
            losses[i] = labeled_loss(xb, px_logit[i], z[i], zm[i], zv[i], zm_prior[i], zv_prior[i])
    with tf.name_scope('final_loss'):
        loss = tf.add_n([nent] + [qy[:, i] * losses[i] for i in xrange(k)])

cluster_images = []
num_images = 10

for i in range(k):
    cluster_images.append(generate_n_images(i, k, 10))

train_step = tf.train.AdamOptimizer().minimize(loss)
sess = tf.Session()
# sess.run(tf.initialize_all_variables())
sess.run(tf.global_variables_initializer()) # Change initialization protocol depending on tensorflow version
sess_info = (sess, qy_logit, nent, loss, train_step, cluster_images)
train('logs/gmvae_k={:d}.log'.format(k), mnist, sess_info, epochs=1000)
