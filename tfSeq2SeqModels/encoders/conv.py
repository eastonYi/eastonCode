'''@file listener.py
contains the listener code'''

import tensorflow as tf
import numpy as np
from .encoder import Encoder
from tfModels.layers import residual, conv_lstm

from tfModels.tensor2tensor.common_layers import layer_norm


class CONV(Encoder):
    '''VERY DEEP CONVOLUTIONAL NETWORKS FOR END-TO-END SPEECH RECOGNITION
    '''

    def encode(self, features, len_feas):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: [batch_size x time x ...] tensor
            input_seq_length: [batch_size] vector
            is_train: whether or not the network is in training mode

        Returns:
            - [bath_size x time x ...] tensor
            - [batch_size] tensor
        '''
        num_cell_units = self.args.model.encoder.num_cell_units
        use_residual = self.args.model.encoder.use_residual
        dropout = self.args.model.encoder.dropout
        num_filters = self.args.model.encoder.num_filters
        size_feat = self.args.data.dim_input

        # x = tf.expand_dims(features, -1)
        size_batch  = tf.shape(features)[0]
        size_length = tf.shape(features)[1]
        size_feat = int(size_feat/3)
        x = tf.reshape(features, [size_batch, size_length, size_feat, 3])
        # the first cnn layer
        x = self.normal_conv(
            inputs=x,
            filter_num=num_filters,
            kernel=(3,3),
            stride=(2,2),
            padding='SAME',
            use_relu=True,
            name="conv",
            w_initializer=None,
            norm_type='layer')
        x = conv_lstm(
            inputs=x,
            kernel_size=(3,3),
            filters=num_filters)

        size_feat = int(np.ceil(size_feat/2))*num_filters
        size_length  = tf.cast(tf.ceil(tf.cast(size_length,tf.float32)/2), tf.int32)
        len_sequence = tf.cast(tf.ceil(tf.cast(len_feas,tf.float32)/2), tf.int32)
        x = tf.reshape(x, [size_batch, size_length, size_feat])


        return outputs, output_seq_lengths

    @staticmethod
    def normal_conv(inputs, filter_num, kernel, stride, padding, use_relu, name,
                    w_initializer=None, norm_type="batch"):
        with tf.variable_scope(name):
            net = tf.layers.conv2d(inputs, filter_num, kernel, stride, padding,
                               kernel_initializer=w_initializer, name="conv")
            if norm_type == "batch":
                net = tf.layers.batch_normalization(net, name="bn")
            elif norm_type == "layer":
                net = layer_norm(net)
            else:
                net = net
            output = tf.nn.relu(net) if use_relu else net

        return output
