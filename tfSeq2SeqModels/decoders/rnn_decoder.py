'''@file rnn_decoder.py
contains the general recurrent decoder class'''

import logging
from abc import ABCMeta, abstractmethod
import tensorflow as tf
from .decoder import Decoder
# from ..tools import helpers


class RNNDecoder(Decoder):
    '''a speller decoder for the LAS architecture'''

    __metaclass__ = ABCMeta

    def _decode(self, encoded, len_encoded, labels, len_labels):
        '''
        Create the variables and do the forward computation to decode an entire
        sequence

        Args:
            encoded: the encoded inputs, this is a list of
                [batch_size x ...] tensors
            encoded_seq_length: the sequence lengths of the encoded inputs
                as a list of [batch_size] vectors
            labels: the labels used as decoder inputs as a list of
                [batch_size x ...] tensors
            len_labels: the sequence lengths of the labels
                as a list of [batch_size] vectors

        Returns:
            - the output logits of the decoder as a list of
                [batch_size x ...] tensors
            - the logit sequence_lengths as a list of [batch_size] vectors
            - the final state of the decoder as a possibly nested tupple
                of [batch_size x ... ] tensors
        '''
        num_cell_units = self.args.model.decoder.num_cell_units

        #get the batch size
        batch_size = tf.shape(len_encoded)[0]

        #create the rnn cell
        decoder_cell, decoder_init_state = self.create_DecoderCell_and_initState(
            num_cell_units=num_cell_units,
            encoded=encoded,
            len_encoded=len_encoded,
            batch_size=batch_size)

        #create the decoder
        if self.helper:
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=self.helper,
                initial_state=decoder_init_state,
                output_layer=None)
        else:
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_cell,
                embedding=self.embedding,
                start_tokens=tf.fill([batch_size], self.start_token),
                end_token=self.end_token,
                initial_state=decoder_init_state,
                beam_width=self.beam_size,
                output_layer=None,
                length_penalty_weight=self.args.length_penalty_weight)

        #use the decoder
        outputs, _, len_decode = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            maximum_iterations=self.max_decoder_len())

        if 'BeamSearch' in str(type(outputs)):
            # logits = tf.reduce_sum(outputs.beam_search_decoder_output.scores, 1)[0]
            logits = tf.no_op()
            sample_id = outputs.predicted_ids
            # sample_id = tf.Print(sample_id, [logits], message='scores: ', summarize=1000)
        else:
            logits = outputs.rnn_output
            sample_id = outputs.sample_id
            # sample_id = tf.Print(sample_id, [tf.shape(sample_id)], message='sample_id: ', summarize=1000)

        return logits, sample_id, len_decode

    @abstractmethod
    def create_cell(self, num_cell_units, encoded, len_encoded):
        '''create the rnn cell

        Args:
            encoded: the encoded sequences as a [batch_size x max_time x dim]
                tensor that will be queried with attention
                set to None if the decoder_cell should be created without the
                attention part (for zero_state)
            encoded_seq_length: the encoded sequence lengths as a [batch_size]
                vector

        Returns:
            an RNNCell object'''


    def create_DecoderCell_and_initState(self, num_cell_units, encoded, len_encoded, batch_size):
        if self.beam_size <= 1:
            #create the rnn cell
            decoder_cell = self.create_cell(num_cell_units, encoded, len_encoded)

            #create the decoder_initial_state
            decoder_init_state = decoder_cell.zero_state(
                batch_size=batch_size,
                dtype=tf.float32)
        else:
            encoded = tf.contrib.seq2seq.tile_batch(encoded, self.beam_size)
            len_encoded = tf.contrib.seq2seq.tile_batch(len_encoded, self.beam_size)
            decoder_cell = self.create_cell(num_cell_units, encoded, len_encoded)

            #create the decoder_initial_state
            decoder_init_state = decoder_cell.zero_state(
                batch_size=batch_size*self.beam_size,
                dtype=tf.float32)

        return decoder_cell, decoder_init_state

    def __getstate__(self):
        '''getstate'''

        return self.__dict__

    @staticmethod
    def create_attention_mechanism(attention_option, num_units, memory,
                                   source_sequence_length):
        if attention_option == "luong":
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
              num_units, memory, memory_sequence_length=source_sequence_length)
        elif attention_option == "scaled_luong":
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
              num_units,
              memory,
              memory_sequence_length=source_sequence_length,
              scale=True)
        elif attention_option == "bahdanau":
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
              num_units, memory, memory_sequence_length=source_sequence_length)
        elif attention_option == "normed_bahdanau":
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
              num_units,
              memory,
              memory_sequence_length=source_sequence_length,
              normalize=True)
        else:
            raise ValueError("Unknown attention option %s" % attention_option)

        return attention_mechanism

    # def create_decoder_helper(self, labels, len_labels, batch_size, beam_size):
    #     if self.is_train:
    #         if self.args.model.decoder.helper_type == 'ScheduledEmbeddingTrainingHelper':
    #             helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
    #                 inputs=self.embedding(labels),
    #                 sequence_length=len_labels,
    #                 embedding=self.embedding,
    #                 sampling_probability=self.sample_prob)
    #         elif self.args.model.decoder.helper_type == 'ScheduledArgmaxEmbeddingTrainingHelper':
    #             helper = helpers.ScheduledArgmaxEmbeddingTrainingHelper(
    #                 embedding=self.embedding,
    #                 start_tokens=tf.fill([batch_size], self.start_token),
    #                 end_token=self.end_token,
    #                 softmax_temperature=self.args.model.decoder.softmax_temperature,
    #                 sampling_probability=self.sample_prob)
    #         elif self.args.model.decoder.helper_type == 'ScheduledSelectEmbeddingHelper':
    #             helper = helpers.ScheduledSelectEmbeddingHelper(
    #                 embedding=self.embedding,
    #                 start_tokens=tf.fill([batch_size], self.start_token),
    #                 end_token=self.end_token,
    #                 softmax_temperature=self.args.model.decoder.softmax_temperature,
    #                 sampling_probability=self.sample_prob)
    #         elif self.args.model.decoder.helper_type == 'TrainingHelper':
    #             helper = tf.contrib.seq2seq.TrainingHelper(
    #                 inputs=self.embedding(labels),
    #                 sequence_length=len_labels,
    #                 name='TrainingHelper')
    #         else:
    #             raise NotImplementedError
    #     else:
    #         if beam_size > 0:
    #             helper = None
    #         else:
    #             helper =tf.contrib.seq2seq.GreedyEmbeddingHelper(
    #                 embedding=self.embedding,
    #                 start_tokens=tf.fill([batch_size], self.start_token),
    #                 end_token=self.end_token)
    #
    #     return helper
