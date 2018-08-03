from sklearn.manifold import MDS
import configparser
from sklearn.preprocessing import StandardScaler
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read('gmvae.ini')
config = config['MDS']

if config['data'] == 'mnist':
    data = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_features = data.train.images
    test_features = data.test.images
    test_labels = data.test.labels
elif config['data'] == 'amazon_fashion':
    asins = np.load(config['amazon_fashion_asin'])
    features = np.load(config['amazon_fashion_feature'])
    labels = np.load(config['amazon_fashion_label'])
    
    bound = int(len(asins) * 4 / 5)
    
    train_asins = asins[:bound]
    test_asins = asins[bound:]
    train_features = features[:bound]
    test_features = features[bound:]
    train_labels = labels[:bound]
    test_labels = labels[bound:]
    
    if config.getboolean('shuffle'):
        shuffle = np.random.permutation(len(combined_asins))
        combined_asins = np.concatenate([train_asins, test_asins], 0)[shuffle]
        combined_features = np.concatenate([train_features, test_features], 0)[shuffle]
        combined_labels = np.concatenate([train_labels, test_labels], 0)[shuffle]
        
        train_asins = combined_asins[:bound]
        test_asins = combined_asins[bound:]
        train_features = combined_features[:bound]
        test_features = combined_features[bound:]
        train_labels = combined_labels[:bound]
        test_labels = combined_labels[bound:]
    
    if config.getboolean('normalize_data'):
        train_features = (train_features / 255.)
        test_features = (test_features / 255.)
else:
    data = pickle.load(open(config['data'], "rb" ))
    # print(data['train']['data'].shape)
    if config.getboolean('normalize_data'):
        scaler = StandardScaler()
        data['train']['data'] = scaler.fit_transform(data['train']['data'])
        data['test']['data'] = scaler.transform(data['test']['data'])
    train_features = data['train']['data']
    test_features = data['test']['data']
    test_labels = data['test']['clusters']
        
def stream_print(f, string, pipe_to_file=True):
    print(string)
    if pipe_to_file and f is not None:
        f.write(string + '\n')
        f.flush()

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
    
def mds(test_data, test_labels, custom=False):
    name = 'other' if (config['data'] != 'mnist' and config['data'] != 'amazon_fashion') else config['data']
    dirname = 'logs/%s/mds/ninit%s_maxiter%s' % (name, config['n_init'], config['max_iter'])
#     colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'lightpink', 'lightblue', 'darkred']
    label_dict = {}
    mds = MDS(n_components=int(config['n_components']), n_init=int(config['n_init']), max_iter=int(config['max_iter']))
    test_reduced = mds.fit_transform(test_data)
    for i in range(len(test_labels)):
        label = test_labels[i] if (config['data'] != 'mnist' and config['data'] != 'amazon_fashion') else test_labels[i].argmax()
        if label in label_dict:
            label_dict[label].append(test_reduced[i])
        else:
            label_dict[label] = [test_reduced[i]]
    f, dirnum = open_file(dirname)
    if config.getboolean('plot_data'):
        graph_data = []
    
    if config.getboolean('plot_data'):
        plt.figure()
        for lab in label_dict:
            plt.plot([x[0] for x in label_dict[lab]], [y[1] for y in label_dict[lab]], 'o', label='clus' + str(lab))
        plt.title('2D MDS')
        plt.savefig('%s.%d/mds.png' % (dirname, dirnum))
    if f is not None: f.close()

random = np.random.choice(len(test_features), 1000)
mds(test_features[random], test_labels[random])