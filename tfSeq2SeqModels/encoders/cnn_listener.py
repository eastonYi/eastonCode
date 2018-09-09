'''@file listener.py
contains the listener code'''

import tensorflow as tf
from .encoder import Encoder
import numpy as np

from nabu.neuralnetworks.components import layer
from tfModels.layers import conv_layer
from tfModels.tensor2tensor.dcommon_layers import normal_conv

class CNN_Listener(Encoder):
    '''a listener object
    transforms input features into a high level representation'''

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
        num_pblayers = self.args.model.encoder.num_pblayers
        num_blayers = self.args.model.encoder.num_blayers
        num_cell_units = self.args.model.encoder.num_cell_units
        dropout = self.args.model.encoder.dropout
        size_feat = self.args.data.dim_input

        # the first cnn layer
        size_batch  = tf.shape(features)[0]
        size_length = tf.shape(features)[1]
        x = tf.reshape(features, [size_batch, size_length, int(size_feat/3), 3])
        x = normal_conv(
            inputs=x,
            filter_num=64,
            kernel=(3,3),
            stride=(2,2),
            padding='SAME',
            use_relu=True,
            name="conv",
            w_initializer=None,
            norm_type='layer')
        # conv_output = tf.expand_dims(features, -1)
        # len_sequence = len_feas
        # conv_output, len_sequence, size_feat = conv_layer(
        #     inputs=conv_output,
        #     len_sequence=len_sequence,
        #     size_feat=size_feat,
        #     num_filter=64,
        #     kernel=(3,3),
        #     stride=(2,2),
        #     scope='en_conv_0')

        # the second pblstm layer
        size_feat = int(np.ceil(40/2))*64
        size_length  = tf.cast(tf.ceil(tf.cast(size_length,tf.float32)/2), tf.int32)
        output_seq_lengths = tf.cast(tf.ceil(tf.cast(len_feas,tf.float32)/2), tf.int32)
        outputs = tf.reshape(x, [size_batch, size_length, size_feat])

        for l in range(num_pblayers):
            outputs, output_seq_lengths = layer.pblstm(
                inputs=outputs,
                sequence_length=output_seq_lengths,
                num_units=num_cell_units,
                num_steps=2,
                layer_norm=True,
                scope='en_pblstm_%d' % l)

            if dropout > 0 and self.is_train:
                outputs = tf.nn.dropout(outputs, keep_prob=1.0-dropout)

        # the third blstm layer
        for l in range(num_blayers):
            outputs = layer.blstm(
                inputs=outputs,
                sequence_length=output_seq_lengths,
                num_units=num_cell_units,
                scope='en_blstm_%d' % (l+num_pblayers))

            if dropout > 0 and self.is_train:
                outputs = tf.nn.dropout(outputs, keep_prob=1.0-dropout)

        return outputs, output_seq_lengths
