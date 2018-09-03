'''@file listener.py
contains the listener code'''

import tensorflow as tf
from .encoder import Encoder

from nabu.neuralnetworks.components import layer
from tfModels.layers import conv_lstm

class CONV_LSTM(Encoder):
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
        conv_output = tf.expand_dims(features, -1)
        conv_output, len_sequence = conv_lstm(conv_output, len_feas, (3,3), filters=64)
        conv_output.set_shape([None, None, size_feat, None])
        size_batch = tf.shape(conv_output)[0]
        conv_output = tf.reshape(conv_output, [size_batch, -1, size_feat*64])

        # the second pblstm layer
        outputs = conv_output
        output_seq_lengths = len_sequence
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
