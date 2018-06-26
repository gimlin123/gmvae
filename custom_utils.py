from tensorbayes import progbar
from scipy.stats import mode
import numpy as np
import os.path
import configparser
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read('gmvae.ini')
config = config['gmvae_k']

def stream_print(f, string, pipe_to_file=True):
    print(string)
    if pipe_to_file and f is not None:
        f.write(string + '\n')
        f.flush()

def test_acc(custom_data, sess, qy_logit):
    logits = sess.run(qy_logit, feed_dict={'x:0': custom_data['test']['data'], 'l:0': custom_data['test']['labels']})
    cat_pred = logits.argmax(1)
    real_pred = np.zeros_like(cat_pred)
    for cat in range(logits.shape[1]):
        idx = cat_pred == cat
        test_num = sum(custom_data['test']['clusters'] == cat)
        print('for cluster %d, there are %d guesses while there are %d actual points' % (cat, sum(idx), test_num))
        lab = custom_data['test']['clusters'][idx]
        if len(lab) == 0:
            continue
        real_pred[cat_pred == cat] = mode(lab).mode[0]
    return np.mean(real_pred == custom_data['test']['clusters'])

def open_file(dirname):
    if dirname is None:
        return None
    else:
        i = 0
        while os.path.isdir('{:s}.{:d}'.format(dirname, i)):
            i += 1

        os.mkdir('{:s}.{:d}'.format(dirname, i))
        dirnum = i
        return open('{:s}.{:d}/results.log'.format(dirname, i), 'w'), dirnum

def next_batch(data, num_points, index):
    index = index % len(data)
    if index + num_points <= len(data):
        return data[index : index + num_points]

    return np.concatenate((data[index : ], data[ : index + num_points - len(data)]))

def train_custom(dirname, custom_data, sess_info, epoch):
    if config.getboolean('triplet_loss'):
        formatted_triplets = np.load(config['triplet_path'])
    if config.getboolean('scale_mnist'):
        (train_images_cats, test_images_cats) = np.load(config['mnist_categories'])
        mnist.train.images[train_images_cats] *= float(config['mnist_scaling_factor'])
        mnist.test.images[test_images_cats] *= float(config['mnist_scaling_factor'])

    (sess, qy_logit, nent, loss, train_step, trip_loss, triplet_step, generate_n_images, generate_mean_image, xdata) = sess_info
    # print(sess.run(generate_mean_image(0, 10)))
    f, dirnum = open_file(dirname)
    iterep = 500
    tripep = int(config['tl_interleave_epoch'])
    start_tri = 1
    tripepshow = 20
    index = 0
    if config.getboolean('plot_data'):
        graph_data = []
    for i in range(iterep * epochs):
        if config.getboolean('triplet_loss'):
            if (i / iterep) > start_tri and (i + 1) % tripep == 0:
                for iter in range(len(formatted_triplets)):
                    _, a = sess.run([triplet_step, trip_loss], feed_dict={'x:0': formatted_triplets[iter]})
                    # if (iter + 1) % tripepshow == 0:
                    #     print(a.mean())

        if i %  iterep == 0:
            random_choices = np.random.choice(50000, 10000)
            a, b = sess.run([nent, loss], feed_dict={'x:0': custom_data['train']['data'][random_choices], 'l:0': custom_data['train']['labels'][random_choices]})
            c, d, x_dat = sess.run([nent, loss, xdata], feed_dict={'x:0': custom_data['test']['data'], 'l:0': custom_data['test']['labels']})
            # print(x_dat)
            a, b, c, d = -a.mean(), b.mean(), -c.mean(), d.mean()
            e = test_acc(custom_data, sess, qy_logit)
            if config.getboolean('plot_data'):
                graph_data.append(e)
            string = ('{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s}'
                      .format('tr_ent', 'tr_loss', 't_ent', 't_loss', 't_acc', 'epoch'))
            stream_print(f, string, i <= iterep)
            string = ('{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10d}'
                      .format(a, b, c, d, e, int(i / iterep)))
            stream_print(f, string)
            progbar(i, iterep)
            if (i / iterep) < start_tri or config.getboolean('triplet_interleave'):
                sess.run(train_step, feed_dict={'x:0': next_batch(custom_data['train']['data'], 100, index),  'l:0': next_batch(custom_data['train']['labels'], 100, index)})
            index += 100
    if config.getboolean('plot_data'):
        plt.figure()
        plt.plot(graph_data)
        plt.title('test accuracy vs epochs')
        plt.savefig('%s.%d/results.png' % (dirname, dirnum))
    if f is not None: f.close()
