'''@file listener.py
contains the listener code'''

import tensorflow as tf
import numpy as np
from .encoder import Encoder
from tfModels.layers import residual, conv_lstm
from tfModels.tensor2tensor import common_attention
from ..tools.utils import residual, multihead_attention, ff_hidden

from tfModels.tensor2tensor.common_layers import layer_norm


class Transformer_Encoder(Encoder):
    '''VERY DEEP CONVOLUTIONAL NETWORKS FOR END-TO-END SPEECH RECOGNITION
    '''
    def __init__(self, args, is_train, embed_table=None, name=None):
        super().__init__(args, is_train, embed_table=None, name=None)
        self.attention_dropout_rate = args.model.encoder.attention_dropout_rate if is_train else 0.0
        self.residual_dropout_rate = args.model.encoder.residual_dropout_rate if is_train else 0.0
        self.hidden_units = args.model.encoder.num_cell_units
        self.num_heads = args.model.encoder.num_heads
        self.num_blocks = args.model.encoder.num_blocks
        self._ff_activation = tf.nn.relu


    def encode(self, features, len_feas):


        # Mask
        encoder_padding = tf.equal(tf.sequence_mask(len_feas, maxlen=tf.shape(features)[1]), False) # bool tensor
        encoder_attention_bias = common_attention.attention_bias_ignore_padding(encoder_padding)

        # Add positional signal
        encoder_output = common_attention.add_timing_signal_1d(features)
        # Dropout
        encoder_output = tf.layers.dropout(encoder_output,
                                           rate=self.residual_dropout_rate,
                                           training=self.is_train)
        encoder_output = tf.layers.dense(
            inputs=encoder_output,
            units=self.hidden_units,
            activation=None,
            use_bias=False,
            name='encoder_fc')

        # Blocks
        for i in range(self.num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                # Multihead Attention
                encoder_output = residual(encoder_output,
                                          multihead_attention(
                                              query_antecedent=encoder_output,
                                              memory_antecedent=None,
                                              bias=encoder_attention_bias,
                                              total_key_depth=self.hidden_units,
                                              total_value_depth=self.hidden_units,
                                              output_depth=self.hidden_units,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.attention_dropout_rate,
                                              name='encoder_self_attention',
                                              summaries=False),
                                          dropout_rate=self.residual_dropout_rate)

                # Feed Forward
                encoder_output = residual(encoder_output,
                                          ff_hidden(
                                              inputs=encoder_output,
                                              hidden_size=4 * self.hidden_units,
                                              output_size=self.hidden_units,
                                              activation=self._ff_activation),
                                          dropout_rate=self.residual_dropout_rate)
        # Mask padding part to zeros.
        encoder_output *= tf.expand_dims(1.0 - tf.to_float(encoder_padding), axis=-1)

        return encoder_output, len_feas
