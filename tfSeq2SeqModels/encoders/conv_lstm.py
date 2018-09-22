'''@file listener.py
contains the listener code'''

import tensorflow as tf
import numpy as np
from .encoder import Encoder

from tfModels.tensor2tensor.common_layers import conv_lstm
from tfModels.tensor2tensor.dcommon_layers import normal_conv, normal_pooling


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
        num_filters = self.args.model.encoder.num_filters
        use_residual = self.args.model.encoder.use_residual
        size_feat = self.args.data.dim_input

        # x = tf.expand_dims(features, -1)
        size_batch  = tf.shape(features)[0]
        size_length = tf.shape(features)[1]
        size_feat = int(size_feat/3)
        x = tf.reshape(features, [size_batch, size_length, size_feat, 3])
        # the first cnn layer
        x = normal_conv(
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
            x=x,
            kernel_size=(3,3),
            filters=num_filters)

        size_feat = int(np.ceil(size_feat/2))*num_filters
        size_length  = tf.cast(tf.ceil(tf.cast(size_length,tf.float32)/2), tf.int32)
        len_sequence = tf.cast(tf.ceil(tf.cast(len_feas,tf.float32)/2), tf.int32)
        x = tf.reshape(x, [size_batch, size_length, size_feat])

        outputs = x
        output_seq_lengths = len_sequence

        outputs = self.blstm(
            hidden_output=outputs,
            len_feas=output_seq_lengths,
            num_cell_units=num_cell_units,
            dropout=dropout,
            use_residual=use_residual,
            name='blstm_1')
        outputs, output_seq_lengths = self.pooling(outputs, output_seq_lengths, 'HALF', 1)

        outputs = self.blstm(
            hidden_output=outputs,
            len_feas=output_seq_lengths,
            num_cell_units=num_cell_units,
            dropout=dropout,
            use_residual=use_residual,
            name='blstm_2')
        outputs, output_seq_lengths = self.pooling(outputs, output_seq_lengths, 'SAME', 2)

        outputs = self.blstm(
            hidden_output=outputs,
            len_feas=output_seq_lengths,
            num_cell_units=num_cell_units,
            dropout=dropout,
            use_residual=use_residual,
            name='blstm_3')
        outputs, output_seq_lengths = self.pooling(outputs, output_seq_lengths, 'HALF', 3)

        # outputs, _ = self.blstm(
        #     hidden_output=outputs,
        #     len_feas=output_seq_lengths,
        #     num_cell_units=num_cell_units,
        #     num_layers=1,
        #     is_train=self.is_train,
        #     cell_type=cell_type,
        #     name='en_blstm_4')
        outputs = self.blstm(
            hidden_output=outputs,
            len_feas=output_seq_lengths,
            num_cell_units=num_cell_units,
            dropout=dropout,
            use_residual=use_residual,
            name='blstm_4')
        outputs, output_seq_lengths = self.pooling(outputs, output_seq_lengths, 'SAME', 4)

        return outputs, output_seq_lengths

    def blstm(self, hidden_output, len_feas, num_cell_units, dropout, use_residual, name):
        num_cell_units /= 2

        with tf.variable_scope(name):
            fwd_lstm_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_cell_units)
            bwd_lstm_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_cell_units)
            if dropout > 0.0:
                fwd_lstm_cell = tf.contrib.rnn.DropoutWrapper(
                    fwd_lstm_cell,
                    state_keep_prob=1.0 - dropout)
                bwd_lstm_cell = tf.contrib.rnn.DropoutWrapper(
                    bwd_lstm_cell,
                    state_keep_prob=1.0 - dropout)
            if use_residual:
                fwd_lstm_cell = tf.contrib.rnn.ResidualWrapper(fwd_lstm_cell)
                bwd_lstm_cell = tf.contrib.rnn.ResidualWrapper(bwd_lstm_cell)
            x, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fwd_lstm_cell,
                cell_bw=bwd_lstm_cell,
                inputs=hidden_output,
                dtype=tf.float32,
                time_major=False,
                sequence_length=len_feas)
            x = tf.concat(x, 2)

        return x

    def pooling(self, x, len_sequence, type, name):
        num_cell_units = self.args.model.encoder.num_cell_units

        x = tf.expand_dims(x, axis=2)
        x = normal_conv(
            x,
            num_cell_units,
            (1, 1),
            (1, 1),
            'SAME',
            'True',
            name="tdnn_"+str(name),
            norm_type='layer')

        if type == 'SAME':
            x = normal_pooling(x, (1, 1), (1, 1), 'SAME')
        elif type == 'HALF':
            x = normal_pooling(x, (2, 1), (2, 1), 'SAME')
            len_sequence = tf.cast(tf.ceil(tf.cast(len_sequence, tf.float32)/2), tf.int32)

        x = tf.squeeze(x, axis=2)

        return x, len_sequence
