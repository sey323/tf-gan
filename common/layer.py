import tensorflow as tf
import numpy as np
import activation


def conv2d( x , stride , filter_size , i ,padding = 'SAME',batch_norm = True , save = False ):
    '''
    畳み込み(Convolution)
    @x          :input
    @filter_size[0],[1] : Conv filter(width,height)
    @filter_size[2]     : input_shape(直前のoutputの次元数と合わせる)
    @filter_size[3]     : output_shape(出力する次元数)
    '''
    conv_w,conv_b = _conv_variable([ filter_size[0], filter_size[1], filter_size[2] , filter_size[3] ],name="conv{0}".format(i))
    conv =  tf.nn.conv2d( x,                                # 入力
                        conv_w,                             # 畳み込みフィルタ
                        strides = [1, stride, stride, 1],   # ストライド
                        padding = padding) + conv_b
    if save:
        tf.summary.histogram("conv{0}".format(i) ,conv)
    if batch_norm:
        conv = bn(conv)
    return conv


def _conv_variable( weight_shape , name="conv" ):
    with tf.variable_scope(name):
        # check weight_shape
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        input_channels  = int(weight_shape[2])
        output_channels = int(weight_shape[3])
        weight_shape = (w,h,input_channels, output_channels)
        # define variables
        weight = tf.get_variable("w", weight_shape     , initializer=tf.contrib.layers.xavier_initializer_conv2d())
        bias   = tf.get_variable("b", [output_channels], initializer=tf.constant_initializer(0.0))
    return weight, bias


def deconv2d( x, stride , filter_size , output_shape ,  i ,batch_norm=False ,save = False):
    '''
    逆畳み込み層(Fractionally-strided Convolution)
    @x          :input
    @filter_size[0],[1]  : Conv filter
    @filter_size[2]      : output_shape
    @filter_size[3]      : input_shape
    @output_shape[0]     : Batch Num
    @output_shape[1],[2] : output width , height
    @output_shape[3]     : output_shape
    '''
    deconv_w,deconv_b = _deconv_variable([ filter_size[0], filter_size[1], filter_size[2] , filter_size[3] ],name="deconv{0}".format(i))
    deconv =  tf.nn.conv2d_transpose(x,
                                    deconv_w,
                                    output_shape=[output_shape[0],output_shape[1],output_shape[2],output_shape[3]],
                                    strides=[1,stride,stride,1],
                                    padding = "SAME",
                                    data_format="NHWC") + deconv_b
    if save:
        tf.summary.histogram("deconv{0}".format(i) ,deconv)
    if batch_norm:
        deconv = bn(deconv)
    return deconv


def _deconv_variable( weight_shape  , name = "deconv"):
    with tf.variable_scope(name):
        # check weight_shape
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        output_channels = int(weight_shape[2])
        input_channels  = int(weight_shape[3])
        weight_shape = (w,h,input_channels, output_channels)
        # define variables
        weight = tf.get_variable("w", weight_shape    , initializer = tf.contrib.layers.xavier_initializer_conv2d())
        bias   = tf.get_variable("b", [input_channels], initializer = tf.constant_initializer(0.0))
    return weight, bias


def max_pool( input , filter_size = 2 , stride = 2 , name = ''):
    '''
    Maxpooling層
    @input      : input
    '''
    return tf.nn.max_pool( input, ksize=[1, filter_size, filter_size, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)


def fc( input , output , i = None,batch_norm = False,save = False):
    '''
    全結合層(Fully Connection)
    @input      : input
    @output     : output shape (example:classes num)
    @batch_num     : Batch size
    '''
    _, n_h = [int(x) for x in input.get_shape()]
    d_fc_w, d_fc_b = _fc_variable([n_h,output],name="fc{0}".format(i))
    fc = tf.matmul( input , d_fc_w) + d_fc_b
    if save:
        tf.summary.histogram("fc{0}".format(i) ,fc)
    if batch_norm:
        fc = bn(fc)
    return fc


def defc( input , output , i = None , activation = activation.ReLU,batch_norm = False,save=False):
    '''
    逆全結合層(DeFully Connection)
    @input         : input
    @output[0]     : Batch Num
    @output[1],[2] : output width , height
    @output[3]     : output_channels
    '''
    zdim = input.get_shape()[1]
    d_fc_w, d_fc_b = _fc_variable([zdim, output[1]*output[2]*output[3]],name="defc{0}".format(i))
    h_fc_r  = tf.matmul( input , d_fc_w) + d_fc_b
    h_fc_a  = activation( h_fc_r )
    defc    = tf.reshape(h_fc_a ,[output[0] , output[1] , output[2] , output[3]] )
    if batch_norm:
        defc = bn(defc)
    if save:
        tf.summary.histogram("defc{0}".format(i) ,defc)
    return defc


def _fc_variable( weight_shape , name="fc" ):
    with tf.variable_scope(name):
        # check weight_shape
        input_channels  = int(weight_shape[0])
        output_channels = int(weight_shape[1])
        weight_shape    = ( input_channels, output_channels)
        # define variables
        weight = tf.get_variable("w", weight_shape     , initializer=tf.contrib.layers.xavier_initializer())
        bias   = tf.get_variable("b", [weight_shape[1]], initializer=tf.constant_initializer(0.0))
    return weight, bias


def flatten( input , batch_num , name = None):
    '''
    flatten(平滑化層)
    @input      : input shape
    @output     : output shape
    '''
    _, n_h, n_w, n_f = [x for x in input.get_shape()]
    output = tf.reshape(input ,[batch_num,int(n_h)*int(n_w)*int(n_f)])
    return output



def bn( x , decay=0.9 , updates_collections=None, epsilon=1e-5, scale=True, is_training=True, scope=None):
    '''
    batch_norm
    x               : input
    '''
    return tf.contrib.layers.batch_norm(x,
                                        decay=decay,
                                        updates_collections=updates_collections,
                                        epsilon=epsilon,
                                        scale=scale,
                                        is_training=is_training,
                                        scope = scope )
