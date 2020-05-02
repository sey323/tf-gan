import logging
import tensorflow as tf


def max_pool(input, pool_size=[2, 2], stride=2, name=""):
    """
    Max pooling層

    Args:
        input (tensor):
        stride (int):
            処理をするストライドのサイズ
        pool_size ([int,int]):
            プーリングを行うフィルタのサイズ
        name:
            層の名前
    """
    output = tf.nn.max_pool2d(
        input,
        ksize=[1, pool_size[0], pool_size[1], 1],
        strides=[1, stride, stride, 1],
        padding="SAME",
        name=name,
    )
    logging.info("[Layer]\tMax Pooling Layer:{}".format(output))

    return output


def average_pooling(x, pool_size=[2, 2], strides=2, padding="VALID"):
    """
    Average pooling層

    Args:
        input (tensor):
        stride (int):
            処理をするストライドのサイズ
        pool_size ([int,int]):
            プーリングを行うフィルタのサイズ
        padding (String):
            Paddingの方式
    """
    output = tf.keras.layers.AveragePooling2D(
        pool_size=pool_size, strides=strides, padding=padding
    )(x)
    logging.info("[Layer]\tMax Pooling Layer:{}".format(output))
    return output
