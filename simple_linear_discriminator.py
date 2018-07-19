import tensorflow as tf
from tensorbayes import Constant, Placeholder, Dense, GaussianSample, log_bernoulli_with_logits, log_normal, cross_entropy_with_logits, show_graph, progbar
import numpy as np
import configparser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

config = configparser.ConfigParser()
config.read('gmvae.ini')
config = config['linear_discriminator']

positive_features = np.load(config['positive_features'])
negative_features = np.load(config['negative_features'])

def stream_print(f, string, pipe_to_file=True):
    print(string)
    if pipe_to_file and f is not None:
        f.write(string + '\n')
        f.flush()
        
def linear_discriminator(x):
    reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='logits')) > 0
    # -- q(y)
    with tf.variable_scope('logits'):
        h1 = Dense(x, 512, 'layer1', tf.nn.relu, reuse=reuse)
        h2 = Dense(h1, 512, 'layer2', tf.nn.relu, reuse=reuse)
        logits = Dense(h2, 2, 'logit', reuse=reuse)
    return logits

def labeled_loss(logits, labels):
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits=logits))
    return loss

def open_file(dirname):
    if dirname is None:
        return None
    else:
        i = 0
        while os.path.isdir('{:s}.{:d}'.format(dirname, i)):
            i += 1

        dirnum = i
        os.mkdir('{:s}.{:d}'.format(dirname, i))
        return open('{:s}.{:d}/results.log'.format(dirname, i), 'w'), dirnum

def next_batch(data, num_points, index):
    index = index % len(data)
    if index + num_points <= len(data):
        return data[index : index + num_points]

    return np.concatenate((data[index : ], data[ : index + num_points - len(data)]))

def test_acc(logits, test_features, test_labels, should_print=False):
    test_logits = sess.run(logits, feed_dict={'x:0': test_features, 'labels:0': test_labels})
    margin = int(config['min_margin'])
    test_logits[:0] += margin
    pred = test_logits.argmax(1)
    if should_print:
        print(test_logits)
    return np.mean(pred == test_labels)

def get_delete_features(sess_info):
    (sess, logits, loss, train_step) = sess_info
    asin_path = config['asin_path']
    feature_path = config['feature_path']
    
    delete_asins = np.array([])
    
    for f in os.listdir(asin_path):
        if not(f.split('.')[-1] == 'npy'):
            continue
        asins = np.load(os.path.join(asin_path, f), encoding = 'latin1')

        feature_f = f.split('_')
        feature_f[-2] = 'features'
        feature_f = '_'.join(feature_f)

        features = np.load(os.path.join(feature_path, feature_f))
        feature_logits = sess.run(logits, feed_dict={'x:0': features, 'labels:0': np.zeros(len(features))})
        
        margin = int(config['min_margin'])
        feature_logits[:0] += margin
        pred = feature_logits.argmax(1)
        delete_asins = np.concatenate((delete_asins, asins[pred == 0]), 0)
    
#     print('=======================================================')
#     print(delete_asins.shape)
    np.save(config['save_path'], delete_asins)
    return delete_asins
        

def train(dirname, pos_features, neg_features, test_pos_features, test_neg_features, sess_info, epochs):
    (sess, logits, loss, train_step) = sess_info
    
    train_features = np.concatenate((pos_features, neg_features))
    train_labels = np.concatenate((np.ones(len(pos_features)), np.zeros(len(neg_features))))
    train_labels_one_hot = np.array([[0, 1]] * len(pos_features) + [[1, 0]] * len(neg_features))
    test_features = np.concatenate((test_pos_features, test_neg_features))
    test_labels = np.concatenate((np.ones(len(test_pos_features)), np.zeros(len(test_neg_features))))
    test_labels_one_hot = np.array([[0, 1]] * len(test_pos_features) + [[1, 0]] * len(test_neg_features))
                                    
    
    shuffle = np.random.permutation(len(train_features))
    train_features = train_features[shuffle]
    train_labels = train_labels[shuffle]
    train_labels_one_hot = train_labels_one_hot[shuffle]
    
    shuffle = np.random.permutation(len(test_features))
    test_features = test_features[shuffle]
    test_labels = test_labels[shuffle]
    test_labels_one_hot = test_labels_one_hot[shuffle]

    f, dirnum = open_file(dirname)
    iterep = 100
    start_tri = 1
    batch_size = 10
    index = 0
    if config.getboolean('plot_data'):
        graph_data = []
    for i in range(iterep * epochs):
        progbar(i, iterep)
        sess.run(train_step, feed_dict={'x:0': next_batch(train_features, batch_size, index), 'labels:0': next_batch(train_labels, batch_size, index)})
        index += batch_size
        
        if i % iterep == 0:
            a = sess.run([loss], feed_dict={'x:0': train_features, 'labels:0': train_labels})
            b = sess.run([loss], feed_dict={'x:0': test_features, 'labels:0': test_labels})
            a, b = a[0],b[0]
            c = test_acc(logits, train_features, train_labels)
            d = test_acc(logits, test_features, test_labels)
            if config.getboolean('plot_data'):
                graph_data.append(c)
            string = ('{:>10s},{:>10s},{:>10s}, {:>10s},{:>10s}'
                      .format('tr_loss', 't_loss', 'tr_acc', 't_acc', 'epoch'))
            stream_print(f, string, i < iterep)
            string = ('{:10.2e},{:10.2e}, {:10.2e}, {:10.2e},{:10d}'
                      .format(a, b, c, d, int( i / iterep)))
            stream_print(f, string)
        

    if config.getboolean('plot_data'):
        plt.figure()
        plt.plot(graph_data)
        plt.title('test accuracy vs epochs')
        plt.savefig('%s.%d/results.png' % (dirname, dirnum))
    if f is not None: f.close()

x = Placeholder((None, int(config['data_x'])), name='x')
        
labels = Placeholder((None), name='labels')
labels = tf.cast(labels, tf.int32)

logits = linear_discriminator(x)
loss = labeled_loss(logits, labels)
train_step = tf.train.AdamOptimizer().minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # Change initialization protocol depending on tensorflow version
sess_info = (sess, logits, loss, train_step)

pos_boundary = int(len(positive_features) * 4 / 5)
neg_boundary = int(len(negative_features) * 4 / 5)

train('logs/discriminator/lin_discrim', positive_features[:pos_boundary], negative_features[:neg_boundary], positive_features[pos_boundary:], negative_features[neg_boundary:], sess_info, 20)

get_delete_features(sess_info)