import logging

import tensorflow as tf
from common.layer.opts import batch_norm as norm


def conv2d(
    input,
    stride,
    filter_size,
    output_dim=None,
    padding="SAME",
    batch_norm=True,
    save=False,
    name=None,
):
    """
    畳み込み(Convolution)

    Args:
        input (tensor):
        stride (int):
            処理をするストライドのサイズ
        filter_size ([int,int]):
            畳み込みフィルタのサイズ
        output_dim (int):
            出力する特徴次元のサイズ
        padding:
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

    input_dim = input.shape[-1]
    conv_w, conv_b = _conv_variable(
        [filter_size[0], filter_size[1], input_dim, output_dim],
        name=layer_name,
    )
    conv = (
        tf.nn.conv2d(
            input=input,  # 入力
            filters=conv_w,  # 畳み込みフィルタ
            strides=[1, stride, stride, 1],  # ストライド
            padding=padding,
        )
        + conv_b
    )
    if save:
        tf.compat.v1.summary.histogram(layer_name, conv)
    if batch_norm:
        conv = norm(conv)
    logging.debug("\t\t[Layer]\tConvolution:{}".format(conv))
    return conv


def _conv_variable(weight_shape, name="conv"):
    with tf.compat.v1.variable_scope(name):
        # check weight_shape
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        input_channels = int(weight_shape[2])
        output_channels = int(weight_shape[3])
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
            "b", [output_channels], initializer=tf.compat.v1.constant_initializer(0.0)
        )
    return weight, bias
