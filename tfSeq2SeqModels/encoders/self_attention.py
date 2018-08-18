'''@file listener.py
contains the listener code'''

import tensorflow as tf
from .encoder import Encoder
from tfModels.tensor2tensor import common_attention
from tfModels.tensor2tensor import common_layers
from tfModels.layers import residual, ff_hidden

from nabu.neuralnetworks.components import layer


class SelfAttention(Encoder):
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
        #add input noise
        num_blocks = self.args.model.encoder.num_blocks
        num_cell_units = self.args.model.encoder.num_cell_units
        num_heads = self.args.model.encoder.num_heads
        dropout = self.args.model.encoder.dropout
        attention_dropout_rate = self.args.model.encoder.attention_dropout_rate if self.is_train else 0.0
        residual_dropout_rate = self.args.model.encoder.residual_dropout_rate if self.is_train else 0.0

        # "relu": tf.nn.relu,
        # "sigmoid": tf.sigmoid,
        # "tanh": tf.tanh,
        # "swish": lambda x: x * tf.sigmoid(x),
        # "glu": lambda x, y: x * tf.sigmoid(y)}
        _ff_activation = tf.nn.relu

        encoder_output = features

        # Mask
        encoder_padding = tf.equal(tf.reduce_sum(encoder_output, -1), 0)
        encoder_attention_bias = common_attention.attention_bias_ignore_padding(encoder_padding)

        # Add positional signal
        encoder_output = common_attention.add_timing_signal_1d(encoder_output)
        # Dropout
        if dropout > 0 and self.is_train:
            encoder_output = tf.nn.dropout(encoder_output, keep_prob=1.0-dropout)

        # Blocks
        with tf.variable_scope("block_0"):
            encoder_output = common_attention.multihead_attention(
                query_antecedent=encoder_output,
                memory_antecedent=None,
                bias=encoder_attention_bias,
                total_key_depth=num_cell_units,
                total_value_depth=num_cell_units,
                output_depth=num_cell_units,
                num_heads=num_heads,
                dropout_rate=attention_dropout_rate,
                name='encoder_self_attention',
                summaries=True)
            encoder_output = ff_hidden(
                inputs=encoder_output,
                hidden_size=4 * num_cell_units,
                output_size=num_cell_units,
                activation=_ff_activation)
        for i in range(1, num_blocks, 1):
            with tf.variable_scope("block_{}".format(i)):
                # Multihead Attention
                encoder_output = residual(
                    encoder_output,
                    common_attention.multihead_attention(
                        query_antecedent=encoder_output,
                        memory_antecedent=None,
                        bias=encoder_attention_bias,
                        total_key_depth=num_cell_units,
                        total_value_depth=num_cell_units,
                        output_depth=num_cell_units,
                        num_heads=num_heads,
                        dropout_rate=attention_dropout_rate,
                        name='encoder_self_attention',
                        summaries=True),
                    dropout_rate=residual_dropout_rate,
                    index_layer=i*2-1)

                # Feed Forward
                encoder_output = residual(
                    encoder_output,
                    ff_hidden(
                        inputs=encoder_output,
                        hidden_size=4 * num_cell_units,
                        output_size=num_cell_units,
                        activation=_ff_activation),
                    dropout_rate=residual_dropout_rate,
                    index_layer=i*2)
        # Mask padding part to zeros.
        encoder_output *= tf.expand_dims(1.0 - tf.to_float(encoder_padding), axis=-1)

        return encoder_output, len_feas
