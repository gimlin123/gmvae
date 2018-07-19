import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorbayes import Constant, Placeholder, Dense, GaussianSample, log_bernoulli_with_logits, log_normal, cross_entropy_with_logits, show_graph, progbar, deconv2d
import numpy as np
import sys
from shared_subgraphs import qy_graph, conv_qy_graph, qz_graph, conv_qz_graph, labeled_loss, labeled_loss_real, labeled_loss_custom, triplet_loss
from custom_utils import train_custom
from utils import train, train_amazon
import configparser
import pickle
from sklearn.preprocessing import StandardScaler
import os

os.environ['cuda_visible_devices'] = '1'

config = configparser.ConfigParser()
config.read('gmvae.ini')
config = config['gmvae_k']

formatted_triplets = None
if config.getboolean('triplet_loss'):
    formatted_triplets = np.load(open(config['triplet_path'], 'r'))
    # print(formatted_triplets.shape)

if config['data'] == 'mnist':
    data = input_data.read_data_sets("MNIST_data/", one_hot=True)
elif config['data'] == 'amazon':
    train_asins = np.load(config['asin_path'], encoding='latin1')
    test_asins = np.load(config['test_asin_path'], encoding='latin1')
    test_features = np.load(config['feature_path'])
    test_features = np.load(config['test_feature_path'])
    
    if config.getboolean('shuffle'):
        train_len = len(train_asins)
        
        combined_asins = np.concatenate([train_asins, test_asins], 0)
        combined_features = np.concatenate([train_features, test_features], 0)
        
        shuffle = np.random.permutation(len(combined_asins))
        combined_asins = combined_asins[shuffle]
        combined_features = combined_features[shuffle]
        
        train_asins = combined_asins[:train_len]
        test_asins = combined_asins[train_len:]
        
        train_features = combined_features[:train_len]
        test_features = combined_features[train_len:]
        
    if config.getboolean('normalize_data'):
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        test_features = scaler.transform(test_features)
elif config['data'] == 'amazon_fashion':
    asins = np.load(config['amazon_fashion_asin'])
    features = np.load(config['amazon_fashion_feature'])
    labels = np.load(config['amazon_fashion_label'])
    
    bound_asins = int(len(asins) * 4 / 5)
    bound_features = int(len(features) * 4 / 5)
    
    train_asins = asins[:bound_asins]
    train_features = features[:bound_features]
    test_asins = asins[bound_asins:]
    test_features = features[bound_features:]
    
    if config.getboolean('shuffle'):
        train_len = len(train_asins)
        
        combined_asins = np.concatenate([train_asins, test_asins], 0)
        combined_features = np.concatenate([train_features, test_features], 0)
        
        shuffle = np.random.permutation(len(combined_asins))
        combined_asins = combined_asins[shuffle]
        combined_features = combined_features[shuffle]
        
        train_asins = combined_asins[:train_len]
        test_asins = combined_asins[train_len:]
        
        train_features = combined_features[:train_len]
        test_features = combined_features[train_len:]
    
    if config.getboolean('normalize_data'):
        train_features = (train_features / 127.5) - 1
        test_features = (test_features / 127.5) - 1
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
    layer_size = int(config['r_layer_size'])
    embedding_size = int(config['embedding_size'])
    reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='px')) > 0
    # -- p(z)
    with tf.variable_scope('pz'):
        zm = Dense(y, embedding_size, 'zm', reuse=reuse)
        zv = Dense(y, embedding_size, 'zv', tf.nn.softplus, reuse=reuse)
    # -- p(x)
    with tf.variable_scope('px'):
        if config['data'] == 'amazon_fashion':
            batch_size = tf.shape(z)[0]
            log_h1 = tf.reshape(Dense(z, 32768, 'log_layer1', tf.nn.relu, reuse=reuse), [-1, 4, 4, 1024])
            log_h2 = deconv2d(log_h1, [batch_size, 8, 8, 512], 'log_layer2', tf.nn.relu, reuse=reuse)
            log_h3 = deconv2d(log_h2, [batch_size, 16, 16, 256], 'log_layer3', tf.nn.relu, reuse=reuse)
            log_h4 = deconv2d(log_h3, [batch_size, 32, 32, 128], 'log_layer4', tf.nn.relu, reuse=reuse)
            log_h5 = deconv2d(log_h4, [batch_size, 64, 64, 64], 'log_layer5', tf.nn.relu, reuse=reuse)
            px_logit = tf.tanh(tf.reshape(deconv2d(log_h5, [batch_size, 128, 128, 3], 'logit', reuse=reuse), [-1, 128*128*3]))
            
            var_h1 = tf.reshape(Dense(z, 32768, 'var_layer1', tf.nn.relu, reuse=reuse), [-1, 4, 4, 1024])
            var_h2 = deconv2d(var_h1, [batch_size, 8, 8, 512], 'var_layer2', tf.nn.relu, reuse=reuse)
            var_h3 = deconv2d(var_h2, [batch_size, 16, 16, 256], 'var_layer3', tf.nn.relu, reuse=reuse)
            var_h4 = deconv2d(var_h3, [batch_size, 32, 32, 128], 'var_layer4', tf.nn.relu, reuse=reuse)
            var_h5 = deconv2d(var_h4, [batch_size, 64, 64, 64], 'var_layer5', tf.nn.relu, reuse=reuse)
            xv = tf.reshape(deconv2d(var_h5, [batch_size, 128, 128, 3], 'xv', tf.nn.softplus, reuse=reuse), [-1, 128*128*3])
        else:
            h1 = Dense(z, layer_size, 'layer1', tf.nn.relu, reuse=reuse)
            h2 = Dense(h1, layer_size, 'layer2', tf.nn.relu, reuse=reuse)
            # px_logit = Dense(h2, int(config['data_x']), 'logit', reuse=reuse)
            px_logit = Dense(h2, int(config['data_x']), 'logit', reuse=reuse)
            xv = Dense(h2, int(config['data_x']), 'xv', tf.nn.softplus, reuse=reuse)
            
    return zm, zv, px_logit, xv

tf.reset_default_graph()
# x = Placeholder((None, int(config['data_x'])), name='x')
# changing for custom stuff
x = Placeholder((None, int(config['data_x'])), name='x')
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
qy_logit, qy = conv_qy_graph(xu, k) if config['data'] == 'amazon_fashion' else qy_graph(xu, k)

# for each proposed y, infer z and reconstruct x
z, zm, zv, zm_prior, zv_prior, px_logit, xv = [[None] * k for i in range(7)]
for i in range(k):
    with tf.name_scope('graphs/hot_at{:d}'.format(i)):
        y = tf.add(y_, Constant(np.eye(k)[i], name='hot_at_{:d}'.format(i)))
        z[i], zm[i], zv[i] = conv_qz_graph(xu, y) if config['data'] == 'amazon_fashion' else qz_graph(xu, y)
        zm_prior[i], zv_prior[i], px_logit[i], xv[i] = px_graph(z[i], y)

# Aggressive name scoping for pretty graph visualization :P
with tf.name_scope('loss'):
    with tf.name_scope('neg_entropy'):
        nent = -cross_entropy_with_logits(qy_logit, qy) * float(config['entropy_lambda'])
    losses = [None] * k
    reconstruct_losses = [None] * k
    kl_losses = [None] * k
    for i in range(k):
        with tf.name_scope('loss_at{:d}'.format(i)):
            if config['data'] == 'mnist':
                if config['data_type'] == 'binary':
                    reconstruct_losses[i], kl_losses[i] = labeled_loss(xu, px_logit[i], z[i], zm[i], zv[i], zm_prior[i], zv_prior[i])
                    losses[i] = reconstruct_losses[i] + kl_losses[i] - np.log(0.1)
                elif config['data_type'] == 'real':
                    reconstruct_losses[i], kl_losses[i] = labeled_loss_real(xu, px_logit[i], xv[i], z[i], zm[i], zv[i], zm_prior[i], zv_prior[i])
                    losses[i] = reconstruct_losses[i] + kl_losses[i] - np.log(0.1)
            else:
                x_min = float(config['x_min']) if config['x_min'] else tf.reduce_min(px_logit[i])
                x_max = float(config['x_max']) if config['x_max'] else tf.reduce_max(px_logit[i])
                px_u = tf.clip_by_value(px_logit[i], x_min, x_max)
                reconstruct_losses[i], kl_losses[i] = labeled_loss_custom(xu, px_u, xv[i], z[i], zm[i], zv[i], zm_prior[i], zv_prior[i])
                losses[i] = reconstruct_losses[i] + kl_losses[i] - np.log(0.1)
    with tf.name_scope('triplet_loss'):
        trip_loss = triplet_loss(z, qy, k)
    with tf.name_scope('reconstruct_loss'):
        reconstruct_loss = tf.add_n([qy[:, i] * reconstruct_losses[i] for i in range(k)])
    with tf.name_scope('kl_loss'):
        kl_loss = tf.add_n([qy[:, i] * kl_losses[i] for i in range(k)])
    with tf.name_scope('final_loss'):
        loss = tf.add_n([nent] + [qy[:, i] * losses[i] for i in range(k)])
    with tf.name_scope('final_triplet_loss'):
        final_trip_loss = loss[::3] + loss[1::3] + loss[2::3] + trip_loss * float(config['tl_lambda'])


train_step = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)
triplet_step = tf.train.AdamOptimizer().minimize(final_trip_loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # Change initialization protocol depending on tensorflow version
sess_info = (sess, qy_logit, nent, loss, reconstruct_loss, kl_loss, train_step, trip_loss, triplet_step, generate_n_images, generate_mean_image, xu)

if config['data'] == 'mnist':
    if config.getboolean('triplet_loss'):
        dirname = '/'.join(['logs', 'mnist', config['data_type'], 'triplet', 'kl%s_r%s_tm%s_ti%s_tl%s_clus_%d' % (config['kl_loss_lambda'],
            config['reconstruct_loss_lambda'], config['tl_margin'], config['tl_interleave_epoch'], config['tl_lambda'], k)])
    else:
        dirname = '/'.join(['logs', 'mnist', config['data_type'], 'no-triplet', 'kl%s_r%s_clus_%d' % (config['kl_loss_lambda'], config['reconstruct_loss_lambda', k])])

elif config['data'] == 'amazon':
    if config.getboolean('triplet_loss'):
        dirname = '/'.join(['logs', 'amazon', 'triplet', 'kl%s_r%s_tm%s_ti%s_tl%s_clus_%d' % (config['kl_loss_lambda'],
            config['reconstruct_loss_lambda'], config['tl_margin'], config['tl_interleave_epoch'], config['tl_lambda'], k)])
    else:
        dirname = '/'.join(['logs', 'amazon', 'no-triplet', 'kl%s_r%s_clus_%d' % (config['kl_loss_lambda'], config['reconstruct_loss_lambda'], k)])

elif config['data'] == 'amazon_fashion':
    if config.getboolean('triplet_loss'):
        dirname = '/'.join(['logs', 'amazon_fashion', 'triplet', 'kl%s_r%s_tm%s_ti%s_tl%s_clus_%d' % (config['kl_loss_lambda'],
            config['reconstruct_loss_lambda'], config['tl_margin'], config['tl_interleave_epoch'], config['tl_lambda'], k)])
    else:
        dirname = '/'.join(['logs', 'amazon_fashion', 'no-triplet', 'kl%s_r%s_clus_%d' % (config['kl_loss_lambda'], config['reconstruct_loss_lambda'], k)])
        
else:
    datafile = config['data'].split('/')[-1].split('.')[0]
    if config.getboolean('triplet_loss'):
        dirname = '/'.join(['logs', 'other', 'triplet', '%s_kl%s_r%s_tm%s_ti%s_tl%s_clus_%d' % (datafile, config['kl_loss_lambda'],
            config['reconstruct_loss_lambda'], config['tl_margin'], config['tl_interleave_epoch'], config['tl_lambda'], k)])
    else:
        dirname = '/'.join(['logs', 'other', 'no-triplet', '%s_kl%s_r%s_clus_%d' % (datafile, config['kl_loss_lambda'], config['reconstruct_loss_lambda'], k)])

if config['data'] == 'mnist':
    train(dirname, data, sess_info, 10)
elif config['data'] == 'amazon':
    train_amazon(dirname, train_asins, test_asins, train_features, test_features, sess_info, 20)
elif config['data'] == 'amazon_fashion':
    train_amazon(dirname, train_asins, test_asins, train_features, test_features, sess_info, 20, labels, k)
else:
    train_custom(dirname, data, sess_info, 10)
