import logging
import tensorflow as tf


def batch_norm(
    x,
    decay=0.9,
    updates_collections=None,
    epsilon=1e-5,
    scale=True,
    is_training=True,
    scope=None,
):
    """
    batch_norm
    x               : input
    """
    return tf.keras.layers.BatchNormalization(
        epsilon=epsilon,
        scale=scale,
        trainable=is_training,
    )(x)


def flatten(input, batch_num=None, name=None):
    """
    flatten(平滑化層)
    入力したTensorを[バッチサイズ, width * height * channel]サイズに変換する。

    Args:
        input (tensor):
        name:
            層の名前
    """
    _, n_h, n_w, n_f = [x for x in input.get_shape()]
    output = tf.reshape(input, [-1, int(n_h) * int(n_w) * int(n_f)])
    logging.info("[Layer]\tFlatten Layer:{}".format(output))
    return output


def padding(input, pad=1, type="CONSTANT"):
    """
    指定したサイズで画像を埋める

    Args:
        input (tensor):
        pad (int):
            上下左右のパディング幅
        type (String):
            paddingの方式
    """
    y = tf.pad(
        tensor=input, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]], mode=type
    )
    return y
