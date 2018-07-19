import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.layers import xavier_initializer
import numpy as np
from IPython.display import clear_output, Image, display, HTML

import os
import sys
import time
import configparser

config = configparser.ConfigParser()
config.read('gmvae.ini')
config = config['gmvae_k']

def Constant(value, dtype='float32', name=None):
    return tf.constant(value, dtype, name=name)

def Placeholder(shape, dtype='float32', name=None):
    return tf.placeholder(dtype, shape, name=name)

@add_arg_scope
def Dense(x,
          num_outputs,
          scope=None,
          activation=None,
          reuse=None,
          bn=False,
          post_bn=False,
          phase=None):

    with tf.variable_scope(scope, 'dense', reuse=reuse):
        # convert x to 2-D tensor
        dim = np.prod(x._shape_as_list()[1:])
        x = tf.reshape(x, [-1, dim])
        weights_shape = (x.get_shape().dims[-1], num_outputs)

        # dense layer
        weights = tf.get_variable('weights', weights_shape,
                                  initializer=xavier_initializer()) / float(config['var_downscale'])
        biases = tf.get_variable('biases', [num_outputs],
                                 initializer=tf.zeros_initializer)
        output = tf.matmul(x, weights) + biases
        if bn: output = batch_norm(output, phase, scope='bn')
        if activation: output = activation(output)
        if post_bn: output = batch_norm(output, phase, scope='post_bn')

    return output

def conv2d(input_,
           output_dim,
           scope=None,
           activation=None,
           reuse=None,
           k_h=5,
           k_w=5,
           d_h=2,
           d_w=2):
    
    with tf.variable_scope(scope, "conv2d", reuse=reuse):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                  initializer=xavier_initializer()) / float(config['conv_var_downscale'])
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv_shape = conv.get_shape().as_list()
        conv_shape[0] = -1
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv_shape)
        if activation: conv = activation(conv)
    return conv

def deconv2d(input_,
             output_shape,
             scope=None,
             activation=None,
             reuse=None,
             k_h=5,
             k_w=5,
             d_h=2, 
             d_w=2):
    with tf.variable_scope(scope, "conv2d", reuse=reuse):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                  initializer=xavier_initializer()) / float(config['conv_var_downscale'])
        
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv_shape = deconv.get_shape().as_list()
        deconv_shape[0] = -1
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv_shape)
        if activation: deconv = activation(deconv)
    return deconv

def log_bernoulli_with_logits(x, logits, eps=0.0, axis=-1):
    if eps > 0.0:
        max_val = np.log(1.0 - eps) - np.log(eps)
        logits = tf.clip_by_value(logits, -max_val, max_val,
                                  name='clipped_logit')
    return -tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=x), axis)

def log_normal(x, mu, var, eps=0.0, axis=-1):
    if eps > 0.0:
        var = tf.add(var, eps, name='clipped_var')
    return -0.5 * tf.reduce_sum(
        tf.log(2 * np.pi) + tf.log(var) + tf.square(x - mu) / var, axis)


def progbar(i, iter_per_epoch, message='', bar_length=50, display=True):
    j = (i % iter_per_epoch) + 1
    end_epoch = j == iter_per_epoch
    if display:
        perc = int(100. * j / iter_per_epoch)
        prog = ''.join(['='] * int(bar_length * perc / 100))
        template = "\r[{:" + str(bar_length) + "s}] {:3d}%. {:s}"
        string = template.format(prog, perc, message)
        sys.stdout.write(string)
        sys.stdout.flush()
        if end_epoch:
            sys.stdout.write('\r{:100s}\r'.format(''))
            sys.stdout.flush()
    return end_epoch, (i + 1)/iter_per_epoch

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

def GaussianSample(mean, var, scope=None):
    with tf.variable_scope(scope, 'gaussian_sample'):
        sample = tf.random_normal(tf.shape(mean), mean, tf.sqrt(var))
        sample.set_shape(mean.get_shape())
        return sample

def cross_entropy_with_logits(logits, targets):
    log_q = tf.nn.log_softmax(logits)
    return -tf.reduce_sum(targets * log_q, 1)
