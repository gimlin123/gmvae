from tensorbayes import progbar
from scipy.stats import mode
import numpy as np
import os.path

def stream_print(f, string, pipe_to_file=True):
    print string
    if pipe_to_file and f is not None:
        f.write(string + '\n')
        f.flush()

def test_acc(custom_data, sess, qy_logit):
    logits = sess.run(qy_logit, feed_dict={'x:0': custom_data['test']['data'], 'l:0': custom_data['test']['labels']})
    cat_pred = logits.argmax(1)
    real_pred = np.zeros_like(cat_pred)
    for cat in xrange(logits.shape[1]):
        idx = cat_pred == cat
        test_num = sum(custom_data['test']['clusters'] == cat)
        print 'for cluster %d, there are %d guesses while there are %d actual points' % (cat, sum(idx), test_num)
        lab = custom_data['test']['clusters'][idx]
        if len(lab) == 0:
            continue
        real_pred[cat_pred == cat] = mode(lab).mode[0]
    return np.mean(real_pred == custom_data['test']['clusters'])

def open_file(fname):
    if fname is None:
        return None
    else:
        i = 0
        while os.path.isfile('{:s}.{:d}'.format(fname, i)):
            i += 1
        return open('{:s}.{:d}'.format(fname, i), 'w', 0)

def next_batch(data, num_points, index):
    index = index % len(data)
    if index + num_points <= len(data):
        return data[index : index + num_points]

    return np.concatenate((data[index : ], data[ : index + num_points - len(data)]))

def train_custom(fname, custom_data, sess_info, epochs):
    (sess, qy_logit, nent, loss, train_step, x, x_reconstruct) = sess_info
    f = open_file(fname)
    iterep = 500
    index = 0

    # print 0 epoch
    random_choices = np.random.choice(50000, 10000)
    a, b = sess.run([nent, loss], feed_dict={'x:0': custom_data['train']['data'][random_choices], 'l:0': custom_data['train']['labels'][random_choices]})
    c, d = sess.run([nent, loss], feed_dict={'x:0': custom_data['test']['data'], 'l:0': custom_data['test']['labels']})
    a, b, c, d = -a.mean(), b.mean(), -c.mean(), d.mean()
    e = test_acc(custom_data, sess, qy_logit)
    string = ('{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s}'
              .format('tr_ent', 'tr_loss', 't_ent', 't_loss', 't_acc', 'epoch'))
    print string
    string = ('{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10d}'
              .format(a, b, c, d, e, 0))
    print string

    for i in range(iterep * epochs):
        sess.run(train_step, feed_dict={'x:0': next_batch(custom_data['train']['data'], 100, index), 'l:0': next_batch(custom_data['train']['labels'], 100, index)})
        index += 100
        progbar(i, iterep)
        if (i + 1) %  iterep == 0:
            random_choices = np.random.choice(50000, 10000)
            a, b = sess.run([nent, loss], feed_dict={'x:0': custom_data['train']['data'][random_choices], 'l:0': custom_data['train']['labels'][random_choices]})
            c, d = sess.run([nent, loss, x], feed_dict={'x:0': custom_data['test']['data'], 'l:0': custom_data['test']['labels']})
            a, b, c, d = -a.mean(), b.mean(), -c.mean(), d.mean()
            e = test_acc(custom_data, sess, qy_logit)
            string = ('{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s}'
                      .format('tr_ent', 'tr_loss', 't_ent', 't_loss', 't_acc', 'epoch'))
            stream_print(f, string, i <= iterep)
            string = ('{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10d}'
                      .format(a, b, c, d, e, (i + 1) / iterep))
            stream_print(f, string)
    if f is not None: f.close()
