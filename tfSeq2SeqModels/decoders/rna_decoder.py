'''
@file speller.py
contains the speller functionality
'''

import tensorflow as tf

from .rnn_decoder import RNNDecoder
from ..tools import helpers

from tfModels.layers import build_cell


class RNADecoder(RNNDecoder):
    '''a speller decoder for the LAS architecture'''

    def create_cell(self, num_cell_units, encoded, len_encoded):

        num_layers = self.args.model.decoder.num_layers
        num_cell_project = self.args.model.decoder.num_cell_project
        dropout = self.args.model.decoder.dropout
        forget_bias = self.args.model.decoder.forget_bias
        cell_type = self.args.model.decoder.cell_type

        cell = build_cell(
            num_units=num_cell_units,
            num_layers=num_layers,
            is_train=self.is_train,
            dropout=dropout,
            forget_bias=forget_bias,
            cell_type=cell_type,
            dim_project=num_cell_project)

        cell = tf.contrib.rnn.OutputProjectionWrapper(
            cell=cell,
            output_size=self.args.dim_output)
        self.zero_state = cell.zero_state

        return cell

    def zero_state(self):

        return self.zero_state

    def create_DecoderCell_and_initState(self, num_cell_units, encoded, len_encoded, batch_size):
        #create the rnn cell
        decoder_cell = self.create_cell(num_cell_units, encoded, len_encoded)

        #create the decoder_initial_state
        decoder_init_state = decoder_cell.zero_state(
            batch_size=batch_size,
            dtype=tf.float32)

        return decoder_cell, decoder_init_state

    def create_decoder_helper(self, encoded, len_encoded, labels, len_labels, batch_size):
        """
        the training and infering has the same helper. At any timestep, the RNA
        don't has the label to guide the decode output.
        """
        helper = helpers.RNAHelper(
            encoded,
            len_encoded,
            embedding=self.embedding,
            start_tokens=tf.fill([batch_size], self.start_token),
            end_token=self.end_token,
            softmax_temperature=self.args.model.decoder.softmax_temperature,
            sampling_probability=self.sample_prob)

        return helper

    def _decode(self, encoded, len_encoded, labels, len_labels):
        num_cell_units = self.args.model.decoder.num_cell_units
        batch_size = tf.shape(len_encoded)[0]

        helper = self.create_decoder_helper(labels, len_labels, batch_size)

        decoder_cell, decoder_init_state = self.create_DecoderCell_and_initState(
            num_cell_units=num_cell_units,
            encoded=encoded,
            len_encoded=len_encoded,
            batch_size=batch_size)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cell,
            helper=helper,
            initial_state=decoder_init_state,
            output_layer=None)

        outputs, _, len_decode = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            maximum_iterations=self.max_decoder_len())

        logits = outputs.rnn_output
        sample_id = outputs.sample_id

        return logits, sample_id, len_decode
