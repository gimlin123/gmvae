from tensorbayes import progbar
from scipy.stats import mode
import numpy as np
import os.path
from PIL import Image
import configparser
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read('gmvae.ini')
config = config['gmvae_k']

def stream_print(f, string, pipe_to_file=True):
    print(string)
    if pipe_to_file and f is not None:
        f.write(string + '\n')
        f.flush()

def test_acc(mnist, sess, qy_logit):
    # print('===============')
    # print(mnist.test.labels.shape)
    # print(mnist.test.labels.argmax(1).shape)
    logits = sess.run(qy_logit, feed_dict={'x:0': mnist.test.images})
    cat_pred = logits.argmax(1)
    real_pred = np.zeros_like(cat_pred)
    for cat in range(logits.shape[1]):
        idx = cat_pred == cat
        lab = mnist.test.labels.argmax(1)[idx]
        if len(lab) == 0:
            continue
        real_pred[cat_pred == cat] = mode(lab).mode[0]
    return np.mean(real_pred == mnist.test.labels.argmax(1))

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

def create_image_m(images, m, epoch, mnist, sess, qy_logit):
    logits = sess.run(qy_logit, feed_dict={'x:0': mnist.test.images})
    cat_pred = logits.argmax(1)
    for cat in range(logits.shape[1]):
        idx = cat_pred == cat
        lab = mnist.test.labels.argmax(1)[idx]
        if len(lab) == 0:
            continue
        real_cat = mode(lab).mode[0]

        image = np.mean(images[cat][0:m], 0).astype(np.uint8)
        im = Image.fromarray(np.reshape(image, (28, 28)), 'L')
        im.save('generated_images/%d_images_e%d.png' % (real_cat, epoch))

def create_mean_image(mean_image, epoch, mnist, sess, qy_logit):
    logits = sess.run(qy_logit, feed_dict={'x:0': mnist.test.images})
    cat_pred = logits.argmax(1)
    for cat in range(logits.shape[1]):
        idx = cat_pred == cat
        lab = mnist.test.labels.argmax(1)[idx]
        if len(lab) == 0:
            continue
        real_cat = mode(lab).mode[0]

        image = np.array(mean_image[cat]).astype(np.uint8)
        im = Image.fromarray(np.reshape(image, (28, 28)), 'L')
        im.save('generated_images/mean_%d_images_e%d.png' % (real_cat, epoch))

def train(dirname, mnist, sess_info, epochs):
    if config.getboolean('triplet_loss'):
        formatted_triplets = np.load(config['triplet_path'])
    if config.getboolean('scale_mnist'):
        (train_images_cats, test_images_cats) = np.load(config['mnist_categories'])
        mnist.train.images[train_images_cats] *= float(config['mnist_scaling_factor'])
        mnist.test.images[test_images_cats] *= float(config['mnist_scaling_factor'])

    (sess, qy_logit, nent, loss, train_step, trip_loss, triplet_step, generate_n_images, generate_mean_image, xdata) = sess_info
    # print(sess.run(generate_mean_image(0, 10)))
    f, dirnum = open_file(dirname)
    print '======================'
    print dirnum
    iterep = 500
    tripep = int(config['tl_interleave_epoch'])
    start_tri = 1
    tripepshow = 20
    if config.getboolean('plot_data'):
        graph_data = []
    for i in range(iterep * epochs):
        if config.getboolean('triplet_loss'):
            if (i / iterep) > start_tri and (i + 1) % tripep == 0:
                for iter in range(len(formatted_triplets)):
                    _, a = sess.run([triplet_step, trip_loss], feed_dict={'x:0': formatted_triplets[iter]})
                    if (iter + 1) % tripepshow == 0:
                        print(a.mean())
        if i % iterep == 0:
            a, b = sess.run([nent, loss], feed_dict={'x:0': mnist.train.images[np.random.choice(50000, 10000)]})
            c, d = sess.run([nent, loss], feed_dict={'x:0': mnist.test.images})
            # create_image_m(clus, 8, (i+1)/iterep, mnist, sess, qy_logit)
            # create_mean_image(mean, (i+1)/iterep, mnist, sess, qy_logit)
            a, b, c, d = -a.mean(), b.mean(), -c.mean(), d.mean()
            e = test_acc(mnist, sess, qy_logit)
            if config.getboolean('plot_data'):
                graph_data.append(e)
            string = ('{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s}'
                      .format('tr_ent', 'tr_loss', 't_ent', 't_loss', 't_acc', 'epoch'))
            stream_print(f, string, i < iterep)
            string = ('{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10d}'
                      .format(a, b, c, d, e, int( i / iterep)))
            stream_print(f, string)
        progbar(i, iterep)
        if (i / iterep) < start_tri or config.getboolean('triplet_interleave'):
            sess.run(train_step, feed_dict={'x:0': mnist.train.next_batch(100)[0]})

    if config.getboolean('plot_data'):
        plt.figure()
        plt.plot(graph_data)
        plt.title('test accuracy vs epochs')
        plt.savefig('%s.%d/results.png' % (dirname, dirnum))
    if f is not None: f.close()
