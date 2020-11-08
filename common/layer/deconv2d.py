import logging

import tensorflow as tf
from common.layer.opts import batch_norm as norm


def deconv2d(
    input,
    stride,
    filter_size,
    output_shape,
    output_dim=None,
    batch_norm=False,
    save=False,
    name=None,
):
    """
    逆畳み込み層(Fractionally-strided Convolution)

    Args:
        input (tensor):
        stride (int):
            処理をするストライドのサイズ
        filter_size ([int,int]):
            畳み込みフィルタのサイズ
        output_shape ([int,int]):
            出力する特徴量の縦横のサイズ
        output_dim (int):
            出力する特徴次元のサイズ
        batch_norm (boolean):
            Batch Normalizationを適応するかどうか
        save (boolean):
            Tesorbardで確認可能な情報を保存するかどうか
        name:
            層の名前
    """
    if name is not None:
        layer_name = name
    else:
        logging.warning("No Layer Name")
        exit()

    input_dim = input.get_shape()[-1]
    deconv_w, deconv_b = _deconv_variable(
        [filter_size[0], filter_size[1], input_dim, output_dim],
        name=layer_name,
    )
    output_shape = tf.stack(
        [
            tf.shape(input)[0],
            output_shape[0],
            output_shape[1],
            output_dim,
        ],
    )
    deconv = (
        tf.nn.conv2d_transpose(
            input,
            deconv_w,
            output_shape=output_shape,
            strides=[1, stride, stride, 1],
            padding="SAME",
            data_format="NHWC",
        )
        + deconv_b
    )
    if save:
        tf.compat.v1.summary.histogram(layer_name, deconv)
    if batch_norm:
        deconv = norm(deconv)
    logging.debug("\t\t[Layer]\tDe Convolution:{}".format(deconv))
    return deconv


# 逆畳み込み層の計算グラフの定義
def _deconv_variable(weight_shape, name="deconv"):
    with tf.compat.v1.variable_scope(name):
        # check weight_shape
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        output_channels = int(weight_shape[2])
        input_channels = int(weight_shape[3])
        weight_shape = (w, h, input_channels, output_channels)
        # define variables
        weight = tf.compat.v1.get_variable(
            "w",
            weight_shape,
            initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0, mode="fan_avg", distribution="uniform"
            ),
        )
        bias = tf.compat.v1.get_variable(
            "b", [input_channels], initializer=tf.compat.v1.constant_initializer(0.0)
        )
    return weight, bias
