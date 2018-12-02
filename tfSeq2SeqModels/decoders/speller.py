'''@file speller.py
contains the speller functionality'''

import tensorflow as tf

from .rnn_decoder import RNNDecoder

from tfModels.layers import build_cell


class Speller(RNNDecoder):
    '''a speller decoder for the LAS architecture'''

    def create_cell(self, num_cell_units, encoded, len_encoded):
        '''create the rnn cell

        Args:
            encoded: the encoded sequences as a [batch_size x max_time x dim]
                tensor that will be queried with attention
                set to None if the rnn_cell should be created without the
                attention part (for zero_state)
            encoded_seq_length: the encoded sequence lengths as a [batch_size]
                vector

        Returns:
            an RNNCell object'''

        num_layers = self.args.model.decoder.num_layers
        # num_cell_project = self.args.model.decoder.num_cell_project
        # dropout = self.args.model.decoder.dropout
        # forget_bias = self.args.model.decoder.forget_bias
        # cell_type = self.args.model.decoder.cell_type

        # cell = build_cell(
        #     num_units=num_cell_units,
        #     num_layers=num_layers,
        #     is_train=self.is_train,
        #     dropout=dropout,
        #     forget_bias=forget_bias,
        #     cell_type=cell_type,
        #     dim_project=num_cell_project)
        cell = self.make_multi_cell(num_cell_units, num_layers)
        self.zero_state = cell.zero_state

        #create the attention mechanism
        attention_mechanism = self.create_attention_mechanism(
            self.args.model.decoder.attention_mechanism,
            num_units=num_cell_units,
            memory=encoded,
            source_sequence_length=len_encoded)

        #add attention to the rnn cell
        cell = tf.contrib.seq2seq.AttentionWrapper(
            cell=cell,
            attention_mechanism=attention_mechanism,
            attention_layer_size=num_cell_units,
            alignment_history=False,
            output_attention=True)

        #the output layer
        cell = tf.contrib.rnn.OutputProjectionWrapper(
            cell=cell,
            output_size=self.args.dim_output)

        return cell

    def make_cell(self, num_cell_units):
        dropout = self.args.model.decoder.dropout
        keep_prob = 1-dropout
        cell = tf.contrib.rnn.LSTMBlockCell(num_cell_units, forget_bias=0.0)
        if self.is_train and keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(
                cell, output_keep_prob=keep_prob)
        return cell

    def make_multi_cell(self, num_cell_units, num_layers):
        list_cells = [self.make_cell(num_cell_units) for _ in range(num_layers-1)]
        cell_proj = tf.contrib.rnn.OutputProjectionWrapper(
            cell=self.make_cell(num_cell_units),
            output_size=self.args.dim_output)
        list_cells.append(cell_proj)
        multi_cell = tf.contrib.rnn.MultiRNNCell(list_cells, state_is_tuple=True)

        return multi_cell

    def zero_state(self):

        return self.zero_state
