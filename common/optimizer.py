import tensorflow as tf


def adam(loss, learn_rate, target_name=""):
    return tf.compat.v1.train.AdamOptimizer(learn_rate, beta1=0.5).minimize(
        loss,
        var_list=[
            x for x in tf.compat.v1.trainable_variables() if target_name in x.name
        ],
    )
