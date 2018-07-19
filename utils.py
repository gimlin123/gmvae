from tensorbayes import progbar
from scipy.stats import mode
import numpy as np
import os.path
from PIL import Image
import configparser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shutil import copyfile
import os

config = configparser.ConfigParser()
config.read('gmvae.ini')
config = config['gmvae_k']

def stream_print(f, string, pipe_to_file=True):
    print(string)
    if pipe_to_file and f is not None:
        f.write(string + '\n')
        f.flush()

def test_acc(mnist, sess, qy_logit):
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

def test_acc_amazon_fashion(test_features, labels, sess, qy_logit, eval_samp_size, k=6):
    random = np.random.choice(len(test_features), eval_samp_size)
    test_features_rand = test_features[random]
    labels_rand = labels[random]
    logits = sess.run(qy_logit, feed_dict={'x:0': test_features_rand})
    cat_pred = logits.argmax(1)
    real_pred = np.zeros_like(cat_pred)
    for cat in range(logits.shape[1]):
        idx = cat_pred == cat
        lab = labels_rand[idx]
        if len(lab) == 0:
            continue
        real_pred[cat_pred == cat] = mode(lab).mode[0]
    return np.mean(real_pred == labels_rand)

def get_clusters(asins, features, sess, qy_logit, eval_samp_size, eval_samp_num, labels = [], k=6):
    pred_dict = {}
    for i in range(eval_samp_num):
        random =  np.random.choice(len(features), eval_samp_size)
        asins_rand = asins[random]
        features_rand = features[random]
        test_labels = labels[int(4 * len(labels) / 5) : ]
        test_labels_rand = test_labels[random]
        logits = sess.run(qy_logit, feed_dict={'x:0': features_rand})
        cat_pred = logits.argmax(1)

        for cat in range(logits.shape[1]):
            idx = cat_pred == cat
            cat_asins = asins_rand[idx]
            if len(cat_asins) == 0:
                continue
            if not(len(labels)):
                pred_dict[cat] = (cat_asins,)
            else:
                pred_dict[cat] = (cat_asins, test_labels_rand[idx])
    return pred_dict

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

def next_batch(data, num_points, index):
    index = index % len(data)
    if index + num_points <= len(data):
        return data[index : index + num_points]

    return np.concatenate((data[index : ], data[ : index + num_points - len(data)]))

def train_amazon(dirname, asins, test_asins, train_features, test_features, sess_info, epochs, labels=None, k=6):
    (sess, qy_logit, nent, loss, reconstruct_loss, kl_loss, train_step, trip_loss, triplet_step, generate_n_images, generate_mean_image, xdata) = sess_info
    f, dirnum = open_file(dirname)
    iterep = 1000
    index = 0
    batch_size = 32
    eval_samp_size = 32
    eval_samp_num = 100
    image_base = dirname + ''
    for i in range(iterep * epochs):
        progbar(i, iterep)
        sess.run(train_step, feed_dict={'x:0': next_batch(train_features, batch_size, index)})
        index += batch_size
        if (i+1) % iterep == 0:
            a, b, g, h, c, d, j, l, e = [[]] * 9
            if not os.path.exists('%s.%d/epoch_%d' % (dirname, dirnum, int((i+1) / iterep))):
                os.makedirs('%s.%d/epoch_%d' % (dirname, dirnum, int((i+1) / iterep)))
            for eval_iter in range(eval_samp_num):
                train_vals = sess.run([nent, loss, reconstruct_loss, kl_loss], feed_dict={'x:0': train_features[np.random.choice(len(train_features), eval_samp_size)]})
                a.append(train_vals[0])
                b.append(train_vals[1])
                g.append(train_vals[2])
                h.append(train_vals[3])
                test_vals = sess.run([nent, loss, reconstruct_loss, kl_loss], feed_dict={'x:0': test_features[np.random.choice(len(test_features), eval_samp_size)]})
                c.append(test_vals[0])
                d.append(test_vals[1])
                j.append(test_vals[2])
                l.append(test_vals[3])
                if len(labels):
                    e.append(test_acc_amazon_fashion(test_features, labels, sess, qy_logit, eval_samp_size, k))
            if not len(labels):
                clusters = get_clusters(test_asins, test_features, sess, qy_logit, eval_samp_size, eval_samp_num)
            else:
                clusters = get_clusters(test_asins, test_features, sess, qy_logit, eval_samp_size, labels, eval_samp_num, k)
            for clus in clusters:
                if not os.path.exists('%s.%d/epoch_%d/clus_%d' % (dirname, dirnum, int((i+1) / iterep), clus)):
                    os.makedirs('%s.%d/epoch_%d/clus_%d' % (dirname, dirnum, int((i+1) / iterep), clus))
                random_indices = np.random.choice(len(clusters[clus][0]), min(int(config['image_sample_cluster']), len(clusters[clus][0])))
                retrieve_asins = clusters[clus][0][random_indices]
                if len(labels):
                    retrieve_cats = clusters[clus][1][random_indices]
                for ind in range(len(retrieve_asins)):
                    asin = retrieve_asins[ind]
#                     first_4 = asin[:4]
#                     first_6 = asin[:6]
#                     print('%s%s/%s/%s.jpg' % (config['image_path'], first_4, first_6, asin))
#                     print('%s.%d/epoch_%d/clus_%d/%s.jpg' % (dirname, dirnum, int((i+1) / iterep), clus, asin))
#                     if os.path.exists('%s%s/%s/%s.jpg' % (config['image_path'], first_4, first_6, asin)):
#                         copyfile('%s%s/%s/%s.jpg' % (config['image_path'], first_4, first_6, asin), '%s.%d/epoch_%d/clus_%d/%s.jpg' % (dirname, dirnum, int((i+1) / iterep), clus, asin))
                    image_path = config['image_path'] if config['data'] == 'amazon' else config['amazon_fashion_image']
                    if config['data'] == 'amazon':
                        if os.path.exists('%s%s.jpg' % (image_path, asin)):
                            copyfile('%s%s.jpg' % (config['image_path'], asin), '%s.%d/epoch_%d/clus_%d/%s.jpg' % (dirname, dirnum, int((i+1) / iterep), clus, asin))
                    else:
                        cat = np.array(['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6'])[int(retrieve_cats[ind])]
                        print('%s%s/%s.jpg' % (config['image_path'], cat, asin))
                        if os.path.exists('%s%s/%s.jpg' % (config['image_path'], cat, asin)):
                            copyfile('%s%s/%s.jpg' % (config['image_path'], cat, asin), '%s.%d/epoch_%d/clus_%d/%s.jpg' % (dirname, dirnum, int((i+1) / iterep), clus, asin))
            a, b, c, d, g, h, j, l = np.mean(a, 0),np.mean(b, 0),np.mean(c, 0),np.mean(d, 0),np.mean(g, 0),np.mean(h, 0),np.mean(j, 0),np.mean(l, 0)
            a, b, c, d, g, h, j, l= -a.mean(), b.mean(), -c.mean(), d.mean(), g.mean(), h.mean(), j.mean(), l.mean()
            if not(len(labels)):
                string = ('{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s}'
                      .format('tr_ent', 'tr_loss', 'tr_rloss', 'tr_klloss', 't_ent', 't_loss', 't_rloss', 't_klloss', 'epoch'))
            else:
                string = ('{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s}'
                      .format('tr_ent', 'tr_loss', 'tr_rloss', 'tr_klloss', 't_ent', 't_loss', 't_rloss', 't_klloss', 't_acc', 'epoch'))
            stream_print(f, string, i < iterep)
            if not(len(labels)):
                string = ('{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10d}'
                      .format(a, b, g, h, c, d, j, l, int( (i+1) / iterep)))
            else:
                string = ('{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10d}'
                      .format(a, b, g, h, c, d, j, l, e, int( (i+1) / iterep)))
            stream_print(f, string)
            
def train(dirname, mnist, sess_info, epochs):
    if config.getboolean('triplet_loss'):
        formatted_triplets = np.load(config['triplet_path'])
    if config.getboolean('scale_mnist'):
        (train_images_cats, test_images_cats) = np.load(config['mnist_categories'])
        mnist.train.images[train_images_cats] *= float(config['mnist_scaling_factor'])
        mnist.test.images[test_images_cats] *= float(config['mnist_scaling_factor'])

    (sess, qy_logit, nent, loss, reconstruct_loss, kl_loss, train_step, trip_loss, triplet_step, generate_n_images, generate_mean_image, xdata) = sess_info
    # print(sess.run(generate_mean_image(0, 10)))
    f, dirnum = open_file(dirname)
    print('======================')
    print(dirnum)
    iterep = 100
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
