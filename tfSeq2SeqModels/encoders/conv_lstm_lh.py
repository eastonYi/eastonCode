'''@file listener.py
contains the listener code'''

import tensorflow as tf
import numpy as np
from .encoder import Encoder

from tfModels.tensor2tensor.common_layers import conv_lstm
from tfModels.tensor2tensor.dcommon_layers import normal_conv, blstm_cell, normal_pooling


class CONV_LSTM(Encoder):
    '''a listener object
    transforms input features into a high level representation
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
        dropout = self.args.model.encoder.dropout
        size_feat = self.args.data.dim_input

        # x = tf.expand_dims(features, -1)
        size_batch  = tf.shape(features)[0]
        size_length = tf.shape(features)[1]
        size_feat = int(size_feat/3)
        x = tf.reshape(features, [size_batch, size_length, size_feat, 3])
        # the first cnn layer
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
        len_sequence = tf.cast(tf.ceil(tf.cast(len_feas,tf.float32)/2), tf.int32)
        x = conv_lstm(
            x=x,
            kernel_size=(3,3),
            filters=64)

        size_feat = int(np.ceil(size_feat/2))*64
        size_length  = tf.cast(tf.ceil(tf.cast(size_length,tf.float32)/2), tf.int32)

        # x.set_shape([None, None, size_feat, None])
        x = tf.reshape(x, [size_batch, size_length, size_feat])

        # the second blstm layer
        with tf.variable_scope('lstm_1'):
            fwd_lstm_cell, bwd_lstm_cell = blstm_cell(
                num_cell_units,
                num_projs=None,
                add_residual=False,
                dropout=dropout)
            x, _ = tf.nn.bidirectional_dynamic_rnn(
              cell_fw=fwd_lstm_cell,
              cell_bw=bwd_lstm_cell,
              inputs=x,
              dtype=tf.float32,
              time_major=False,
              sequence_length=len_sequence)
            x = tf.concat(x, 2)
            x = tf.expand_dims(x, axis=2)
            x = normal_conv(
                x,
                num_cell_units,
                (1, 1),
                (1, 1),
                'SAME',
                'True',
                name="tdnn",
                norm_type='layer')
            x = normal_pooling(x, (2, 1), (2, 1), 'SAME')
            len_sequence = tf.cast(tf.ceil(tf.cast(len_sequence, tf.float32)/2), tf.int32)
            x = tf.squeeze(x, axis=2)

        with tf.variable_scope('lstm_2'):
            fwd_lstm_cell, bwd_lstm_cell = blstm_cell(
                num_cell_units,
                num_projs=None,
                add_residual=False,
                dropout=dropout)
            x, _ = tf.nn.bidirectional_dynamic_rnn(
              cell_fw=fwd_lstm_cell,
              cell_bw=bwd_lstm_cell,
              inputs=x,
              dtype=tf.float32,
              time_major=False,
              sequence_length=len_sequence)
            x = tf.concat(x, 2)
            x = tf.expand_dims(x, axis=2)
            x = normal_conv(
                x,
                num_cell_units,
                (1, 1),
                (1, 1),
                'SAME',
                'True',
                name="tdnn",
                norm_type='layer')
            x = normal_pooling(x, (1, 1), (1, 1), 'SAME')
            x = tf.squeeze(x, axis=2)

        with tf.variable_scope('lstm_3'):
            fwd_lstm_cell, bwd_lstm_cell = blstm_cell(
                num_cell_units,
                num_projs=None,
                add_residual=False,
                dropout=dropout)
            x, _ = tf.nn.bidirectional_dynamic_rnn(
              cell_fw=fwd_lstm_cell,
              cell_bw=bwd_lstm_cell,
              inputs=x,
              dtype=tf.float32,
              time_major=False,
              sequence_length=len_sequence)
            x = tf.concat(x, 2)
            x = tf.expand_dims(x, axis=2)
            x = normal_conv(
                x,
                num_cell_units,
                (1, 1),
                (1, 1),
                'SAME',
                'True',
                name="tdnn",
                norm_type='layer')
            x = normal_pooling(x, (2, 1), (2, 1), 'SAME')
            len_sequence = tf.cast(tf.ceil(tf.cast(len_sequence, tf.float32)/2), tf.int32)
            x = tf.squeeze(x, axis=2)

        with tf.variable_scope('lstm_4'):
            fwd_lstm_cell, bwd_lstm_cell = blstm_cell(
                num_cell_units,
                num_projs=None,
                add_residual=False,
                dropout=dropout)
            x, _ = tf.nn.bidirectional_dynamic_rnn(
              cell_fw=fwd_lstm_cell,
              cell_bw=bwd_lstm_cell,
              inputs=x,
              dtype=tf.float32,
              time_major=False,
              sequence_length=len_sequence)
            x = tf.concat(x, 2)
            x = tf.expand_dims(x, axis=2)
            x = normal_conv(
                x,
                num_cell_units,
                (1, 1),
                (1, 1),
                'SAME',
                'True',
                name="tdnn",
                norm_type='layer')
            x = normal_pooling(x, (1, 1), (1, 1), 'SAME')
            x = tf.squeeze(x, axis=2)

        outputs = x
        output_seq_lengths = len_sequence

        return outputs, output_seq_lengths
