import tensorflow as tf
import numpy as np


'''
Cross Entropy loss
'''
def cross_entropy( x , labels, batch_num , name = '' ):
    with tf.variable_scope('cross_entropy_sigmoid'+name):
        x  = tf.reshape(x , [batch_num, -1])
        labels  = tf.reshape(labels , [batch_num, -1])

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=x, name='cross_entropy_per_example')
        cross_entropy_sigmoid = tf.reduce_mean(cross_entropy, name = 'cross_entropy_sigmoid')
        return cross_entropy_sigmoid
