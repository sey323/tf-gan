import logging
import tensorflow as tf
from common.layer.activation import *
from common.layer.opts import batch_norm as norm


def fc(input, output_dim, name=None, batch_norm=False, save=False):
    """
    全結合層(Fully Connection)

    Args:
        input (tensor):
        output_dim (int):
            出力する特徴次元のサイズ
        batch_norm (boolean):
            Batch Normalizationを適応するかどうか
        save (boolean):
            Tesorbardで確認可能な情報を保存するかどうか
        name:
            層の名前
    """
    d_fc_w, d_fc_b = _fc_variable([input.get_shape()[-1], output_dim], name=name)
    fc = tf.matmul(input, d_fc_w) + d_fc_b
    if save:
        tf.compat.v1.summary.histogram(name, fc)
    if batch_norm:
        fc = norm(fc)
    logging.debug("\t\t[Layer]\tFull Connection Layer:{}".format(fc))
    return fc


def defc(
    input,
    output_shape,
    output_dim,
    activation=ReLU,
    batch_norm=False,
    save=False,
    name="defc",
):
    """
    逆全結合層(DeFully Connection)

    Args:
        input (tensor):
        output_shape ([int,int]):
            出力する特徴量の縦横のサイズ
        output_dim (int):
            出力する特徴次元のサイズ
        activation (tensorflow.nn):
            活性化関数
        batch_norm (boolean):
            Batch Normalizationを適応するかどうか
        save (boolean):
            Tesorbardで確認可能な情報を保存するかどうか
        name:
            層の名前
    """
    noise_dim = input.get_shape()[1]
    d_fc_w, d_fc_b = _fc_variable(
        [noise_dim, output_shape[0] * output_shape[1] * output_dim], name=name
    )
    h_fc_r = tf.matmul(input, d_fc_w) + d_fc_b
    h_fc_a = activation(h_fc_r)
    defc = tf.reshape(h_fc_a, [-1, output_shape[0], output_shape[1], output_dim])
    if batch_norm:
        defc = norm(defc)
    if save:
        tf.compat.v1.summary.histogram(name, defc)
    logging.debug("\t\t[Layer]\tDe Full Connection Layer:{}".format(defc))
    return defc


def _fc_variable(weight_shape, name="fc"):
    with tf.compat.v1.variable_scope(name):
        # check weight_shape
        input_channels = int(weight_shape[0])
        output_channels = int(weight_shape[1])
        weight_shape = (input_channels, output_channels)
        # define variables
        weight = tf.compat.v1.get_variable(
            "w",
            weight_shape,
            initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0, mode="fan_avg", distribution="uniform"
            ),
        )
        bias = tf.compat.v1.get_variable(
            "b", [weight_shape[1]], initializer=tf.compat.v1.constant_initializer(0.0)
        )
    return weight, bias
