import logging
import tensorflow as tf


def cross_entropy(x, labels, name=""):
    """
    Cross Entropy loss
    """
    with tf.compat.v1.variable_scope("cross_entropy_sigmoid" + name):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=x, name="cross_entropy_per_example"
        )
        cross_entropy_sigmoid = tf.reduce_mean(
            input_tensor=cross_entropy, name="cross_entropy_sigmoid"
        )

    tf.compat.v1.summary.scalar(name, cross_entropy_sigmoid)
    logging.info("[Loss]\tCross Entropy Loss:{}".format(cross_entropy_sigmoid))

    return cross_entropy_sigmoid
