from tensorbayes import progbar
from scipy.stats import mode
import numpy as np
import os.path
from PIL import Image

#Used to generate data - will move
#=======================================================
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
#=======================================================

def stream_print(f, string, pipe_to_file=True):
    print string
    if pipe_to_file and f is not None:
        f.write(string + '\n')
        f.flush()

def test_acc(mnist, sess, qy_logit):
    # print '==============='
    # print mnist.test.labels.shape
    # print mnist.test.labels.argmax(1).shape
    logits = sess.run(qy_logit, feed_dict={'x:0': mnist.test.images})
    cat_pred = logits.argmax(1)
    real_pred = np.zeros_like(cat_pred)
    for cat in xrange(logits.shape[1]):
        idx = cat_pred == cat
        lab = mnist.test.labels.argmax(1)[idx]
        if len(lab) == 0:
            continue
        real_pred[cat_pred == cat] = mode(lab).mode[0]
    return np.mean(real_pred == mnist.test.labels.argmax(1))

def open_file(fname):
    if fname is None:
        return None
    else:
        i = 0
        while os.path.isfile('{:s}.{:d}'.format(fname, i)):
            i += 1
        return open('{:s}.{:d}'.format(fname, i), 'w', 0)

def create_image_m(images, m, epoch, mnist, sess, qy_logit):
    logits = sess.run(qy_logit, feed_dict={'x:0': mnist.test.images})
    cat_pred = logits.argmax(1)
    for cat in xrange(logits.shape[1]):
        idx = cat_pred == cat
        lab = mnist.test.labels.argmax(1)[idx]
        if len(lab) == 0:
            continue
        real_cat = mode(lab).mode[0]

        image = np.mean(images[cat][0:m], 0).astype(np.uint8)
        im = Image.fromarray(np.reshape(image, (28, 28)), 'L')
        im.save('generated_images/%d_images_e%d.png' % (real_cat, epoch))

def generate_random_triplets(num_triplets, mnist):
    num_samples = 5000

    sample = np.random.choice(50000, num_samples, replace=False)
    images = mnist.train.images[sample]
    labels = mnist.train.labels[sample]

    label_dict = {}
    for i in range(len(labels)):
        label = labels[i]
        if label in label_dict:
            label_dict[label].append(i)
        else:
            label_dict[label] = [i]

    triplets = []
    for i in range(num_triplets):
        anchor = np.random.randint(num_samples)
        anchor_label = labels[anchor]
        anchor_group = label_dict[anchor_label]

        positive = anchor_group[np.random.randint(len(anchor_group))]

        neg_label = anchor_label
        negative = -1
        while neg_label == anchor_label:
            negative = np.random.randint(num_samples-1)
            if negative >= anchor:
                negative += 1
            neg_label = labels[negative]
        triplets.append([anchor, positive, negative])

    return images, triplets

def format_triplets(images, triplets, batches):
    trip_arr = np.array(triplets)
    anchors = trip_arr[:, 0]
    positives = trip_arr[:, 1]
    negatives = trip_arr[:, 2]

    a_images_split = np.array(np.split(images[anchors], batches))
    p_images_split = np.array(np.split(images[positives], batches))
    n_images_split = np.array(np.split(images[negatives], batches))

    return np.concatenate((a_images_split, p_images_split, n_images_split), axis=1)

#Used to generate data - will move
#=======================================================
batches = 50
# images, triplets = generate_random_triplets(5000, mnist)
# formatted_triplets = format_triplets(images, triplets, batches)
# np.save(open("triplets.npy", "wb"), formatted_triplets)
formatted_triplets = np.load(open('triplets.npy', 'r'))
#=======================================================

def train(fname, mnist, sess_info, epochs):
    (sess, qy_logit, nent, loss, train_step, clusters, trip_loss, triplet_step) = sess_info
    f = open_file(fname)
    iterep = 500
    tripep = 250
    start_tri = 500
    tripepshow = 20
    for i in range(iterep * epochs):
        sess.run(train_step, feed_dict={'x:0': mnist.train.next_batch(100)[0]})
        progbar(i, iterep)
        if i > start_tri and (i + 1) % tripep == 0:
            for iter in range(batches):
                _, a = sess.run([triplet_step, trip_loss], feed_dict={'x:0': formatted_triplets[iter]})
                # if (iter + 1) % tripepshow == 0:
                #     print a.mean()
        if (i + 1) % iterep == 0:
            a, b = sess.run([nent, loss], feed_dict={'x:0': mnist.train.images[np.random.choice(50000, 10000)]})
            c, d, clus = sess.run([nent, loss, clusters], feed_dict={'x:0': mnist.test.images})
            # create_image_m(clus, 8, (i+1)/iterep, mnist, sess, qy_logit)
            a, b, c, d = -a.mean(), b.mean(), -c.mean(), d.mean()
            e = test_acc(mnist, sess, qy_logit)
            string = ('{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s}'
                      .format('tr_ent', 'tr_loss', 't_ent', 't_loss', 't_acc', 'epoch'))
            stream_print(f, string, i <= iterep)
            string = ('{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10d}'
                      .format(a, b, c, d, e, (i + 1) / iterep))
            stream_print(f, string)
    if f is not None: f.close()
