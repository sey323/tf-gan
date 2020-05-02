import tensorflow as tf


def ReLU(x):
    """
    ReLU
    """
    return tf.nn.relu(x)


def leakyReLU(x, alpha=0.1):
    """
    LeakyReLU
    """
    return tf.maximum(x * alpha, x)


def tanh(x):
    """
    Tanh
    """
    return tf.nn.tanh(x)


def sigmoid(x):
    """
    Sigmoid
    """
    return tf.nn.sigmoid(x)
