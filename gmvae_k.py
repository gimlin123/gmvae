import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorbayes import Constant, Placeholder, Dense, GaussianSample, log_bernoulli_with_logits, log_normal, cross_entropy_with_logits, show_graph, progbar
import numpy as np
import sys
from shared_subgraphs import qy_graph, qz_graph, labeled_loss, labeled_loss_real, labeled_loss_custom, triplet_loss
from custom_utils import train_custom
from utils import train
import configparser
import pickle
from sklearn.preprocessing import StandardScaler

config = configparser.ConfigParser()
config.read('gmvae.ini')
config = config['gmvae_k']

formatted_triplets = None
if config.getboolean('triplet_loss'):
    formatted_triplets = np.load(open(config['triplet_path'], 'r'))
    # print(formatted_triplets.shape)

if config['data'] == 'mnist':
    data = input_data.read_data_sets("MNIST_data/", one_hot=True)
else:
    data = pickle.load(open(config['data'], "rb" ))
    # print(data['train']['data'].shape)
    if config.getboolean('normalize_data'):
        scaler = StandardScaler()
        data['train']['data'] = scaler.fit_transform(data['train']['data'])
        data['test']['data'] = scaler.transform(data['test']['data'])
        if config.getboolean('triplet_loss'):
            ft_shape = formatted_triplets.shape
            flatten_shape = (ft_shape[0] * ft_shape[1], ft_shape[2])
            formatted_triplets = scaler.transform(formatted_triplets.reshape(flatten_shape)).reshape(ft_shape)

try:
    k = int(sys.argv[1])
except IndexError:
    raise Exception('Pass an argument specifying the number of mixture components\n'
                    'e.g. python gmvae_k.py 10')

def generate_n_images(clus, k, n):
    y = tf.fill(tf.stack([1, k]), 0.0)
    y = tf.add(y, Constant(np.eye(k)[clus]))

    z_dummy = tf.fill(tf.stack([1, 64]), 0.0)
    zm, zv, _, __ = px_graph(z_dummy, y)

    z = GaussianSample(zm, zv, 'z')
    for i in range(n-1):
        z = tf.concat([z, GaussianSample(zm, zv, 'z' + str(i))], 0)

    _, __, x_reconstruct, ___ = px_graph(z, y)
    x_reconstruct_b = tf.cast(tf.greater(x_reconstruct, tf.zeros(tf.shape(x_reconstruct))), tf.uint8) * 255
    return x_reconstruct_b

def generate_mean_image(clus, k):
    y = tf.fill(tf.stack([1, k]), 0.0)
    y = tf.add(y, Constant(np.eye(k)[clus]))

    z_dummy = tf.fill(tf.stack([1, 64]), 0.0)
    zm, zv, _, __ = px_graph(z_dummy, y)

    _, __, x_reconstruct, ___ = px_graph(zm, y)
    x_reconstruct_b = tf.sigmoid(x_reconstruct) * 255
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
        xv = Dense(h2, 784, 'xv', tf.nn.softplus, reuse=reuse)
    return zm, zv, px_logit, xv

tf.reset_default_graph()
# x = Placeholder((None, 784), name='x')
# changing for custom stuff
x = Placeholder((None, 784), name='x')
l = Placeholder((None, None), name='l')


# binarize data and create a y "placeholder"
with tf.name_scope('x_binarized'):
    xb = tf.cast(tf.greater(x, tf.random_uniform(tf.shape(x), 0, 1)), tf.float32)
with tf.name_scope('y_'):
    y_ = tf.fill(tf.stack([tf.shape(x)[0], k]), 0.0)
with tf.name_scope('x_used'):
    if config['data_type'] == 'binary':
        xu = xb
    elif config['data_type'] == 'real':
        xu = x / float(config['x_downscale'])
# propose distribution over y
qy_logit, qy = qy_graph(xu, k)

# for each proposed y, infer z and reconstruct x
z, zm, zv, zm_prior, zv_prior, px_logit, xv = [[None] * k for i in range(7)]
for i in range(k):
    with tf.name_scope('graphs/hot_at{:d}'.format(i)):
        y = tf.add(y_, Constant(np.eye(k)[i], name='hot_at_{:d}'.format(i)))
        z[i], zm[i], zv[i] = qz_graph(xu, y)
        zm_prior[i], zv_prior[i], px_logit[i], xv[i] = px_graph(z[i], y)

# Aggressive name scoping for pretty graph visualization :P
with tf.name_scope('loss'):
    with tf.name_scope('neg_entropy'):
        nent = -cross_entropy_with_logits(qy_logit, qy)
    losses = [None] * k
    for i in range(k):
        with tf.name_scope('loss_at{:d}'.format(i)):
            if config['data'] == 'mnist':
                if config['data_type'] == 'binary':
                    losses[i] = labeled_loss(xu, px_logit[i], z[i], zm[i], zv[i], zm_prior[i], zv_prior[i])
                elif config['data_type'] == 'real':
                    losses[i] = labeled_loss_real(xu, px_logit[i], xv[i], z[i], zm[i], zv[i], zm_prior[i], zv_prior[i])
            else:
                d = float(config['x_downscale'])
                losses[i] = labeled_loss_custom(xu, px_logit[i], xv[i], z[i], zm[i], zv[i], zm_prior[i], zv_prior[i])
    with tf.name_scope('triplet_loss'):
        trip_loss = triplet_loss(z, qy, k)
    with tf.name_scope('final_loss'):
        loss = tf.add_n([nent] + [qy[:, i] * losses[i] for i in range(k)])
    with tf.name_scope('final_triplet_loss'):
        final_trip_loss = loss[::3] + loss[1::3] + loss[2::3] + trip_loss * float(config['tl_lambda'])


train_step = tf.train.AdamOptimizer().minimize(loss)
triplet_step = tf.train.AdamOptimizer().minimize(final_trip_loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # Change initialization protocol depending on tensorflow version
sess_info = (sess, qy_logit, nent, loss, train_step, trip_loss, triplet_step, generate_n_images, generate_mean_image, xu)

if config['data'] == 'mnist':
    if config.getboolean('triplet_loss'):
        dirname = '/'.join(['logs', 'mnist', config['data_type'], 'triplet', 'kl%s_r%s_tm%s_ti%s_tl%s' % (config['kl_loss_lambda'],
            config['reconstruct_loss_lambda'], config['tl_margin'], config['tl_interleave_epoch'], config['tl_lambda'])])
    else:
        dirname = '/'.join(['logs', 'mnist', config['data_type'], 'no-triplet', 'kl%s_r%s.log' % (config['kl_loss_lambda'], config['reconstruct_loss_lambda'])])

else:
    datafile = config['data'].split('.')[0]
    if config.getboolean('triplet_loss'):
        dirname = '/'.join(['logs', 'other', 'triplet', '%s_kl%s_r%s_tm%s_ti%s_tl%s.log' % (datafile, config['kl_loss_lambda'],
            config['reconstruct_loss_lambda'], config['tl_margin'], config['tl_interleave_epoch'], config['tl_lambda'])])
    else:
        dirname = '/'.join(['logs', 'other', 'no-triplet', '%s_kl%s_r%s.log' % (datafile, config['kl_loss_lambda'], config['reconstruct_loss_lambda'])])

train(dirname, data, sess_info, 10)
