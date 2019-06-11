import tensorflow as tf
import numpy as np

'''
ReLU
'''
def ReLU( x ):
    return tf.nn.relu( x )


'''
LeakyReLU
'''
def leakyReLU( x , alpha=0.1):
    return tf.maximum(x*alpha,x)


'''
Tanh
'''
def tanh( x ):
    return tf.nn.tanh( x )


'''
Sigmoid
'''
def sigmoid( x ):
    return tf.nn.sigmoid(x)
