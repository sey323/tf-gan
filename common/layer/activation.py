import tensorflow as tf


def ReLU(x):
    """
    ReLUによる処理を行う。

    Args:
        x (tensor): 入力

    Returns:
        x :
    """

    return tf.nn.relu(x)


def leakyReLU(x, alpha=0.1):
    """
    leakyReLUによる処理を行う。

    Args:
        x (tensor): 入力

    Returns:
        x :
    """
    return tf.maximum(x * alpha, x)


def tanh(x):
    """
    TanHによる処理を行う。

    Args:
        x (tensor): 入力

    Returns:
        x :
    """
    return tf.nn.tanh(x)


def sigmoid(x):
    """
    sigmoidによる処理を行う。

    Args:
        x (tensor): 入力

    Returns:
        x :
    """
    return tf.nn.sigmoid(x)
