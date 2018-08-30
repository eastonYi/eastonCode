'''@file asr_decoder.py
contains the EDDecoder class'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf
from collections import namedtuple

from tfTools.tfTools import right_shift_rows


class Decoder(object):
    '''a general decoder for an encoder decoder system
    converts the high level features into output logits
    '''

    __metaclass__ = ABCMeta

    def __init__(self, args, is_train, global_step, embed_table=None, name=None):
        '''EDDecoder constructor
        Args:
            conf: the decoder configuration as a configparser
            outputs: the name of the outputs of the model
            constraint: the constraint for the variables

            self.start_token is used in the infer_graph, for auto feed the first
            <sos> tokens to the decoder, while in the train_graph, you need to
            pad the <sos> for the decoder input manually!
            Also, in the infer_graph, decoder should know when to stop, so the
            decoder need to specify the <eos> in the helper or BeamSearchDecoder.
        '''
        self.args = args
        self.name = name
        self.is_train = is_train
        self.start_token = args.token2idx['<sos>'] # tf.fill([self.batch_size], args.token2idx['<sos>'])
        self.end_token = args.token2idx['<eos>']
        self.embed_table = embed_table
        self.global_step = global_step
        self.start_warmup_steps = self.args.model.decoder.start_warmup_steps
        self.sample_prob = self.linear_increase(
            prob_init=self.args.model.decoder.sample_prob,
            global_step=self.global_step,
            start_warmup_steps=self.args.model.decoder.start_warmup_steps,
            step_increasement=self.args.model.decoder.step_increasement)

    def __call__(self, decoder_input):
        '''
        Create the variables and do the forward computation to decode an entire
        sequence

        Returns:
            - the output logits of the decoder as a dictionary of
                [batch_size x time x ...] tensors
            - the logit sequence_lengths as a dictionary of [batch_size] vectors
            - the final state of the decoder as a possibly nested tupple
                of [batch_size x ... ] tensors
        '''
        with tf.variable_scope(self.name or 'decoder'):
            logits, sample_id, len_decode = self._decode(
                encoded=decoder_input.encoded,
                len_encoded=decoder_input.len_encoded,
                labels=decoder_input.input_labels,
                len_labels=decoder_input.len_labels)

        return logits, sample_id, len_decode

    def build_input(self, id_gpu, encoded, len_encoded, tensors_input):
        """
        the decoder label input is tensors_input.labels left concat <sos>,
        the lengths correspond add 1.
        Create a tgt_input prefixed with <sos> and
        PLEASE create a tgt_output suffixed with <eos> in the ce_loss.
        """
        decoder_input = namedtuple('decoder_input',
            'encoded, label, len_encoded_splits, len_label_splits, shape_batch')

        decoder_input.encoded = encoded
        decoder_input.len_encoded = len_encoded

        if tensors_input.label_splits:
            decoder_input.output_labels = tensors_input.label_splits[id_gpu]
            decoder_input.input_labels = right_shift_rows(
                p=tensors_input.label_splits[id_gpu],
                shift=1,
                pad=self.start_token)
            decoder_input.len_labels = tensors_input.len_label_splits[id_gpu]
        else:
            decoder_input.output_labels = None
            decoder_input.input_labels = None
            decoder_input.len_labels = None

        return decoder_input

    def max_decoder_len(self, len_src=None):
        if self.args.model.decoder.len_max_decoder:
            len_max_decode = self.args.model.decoder.len_max_decoder
        else:
            assert len_src
            decoding_length_factor = 2.0
            len_max_decode = tf.to_int32(tf.round(
                tf.to_float(len_src) * decoding_length_factor))

        return len_max_decode

    def embedding(self, ids):
        if self.embed_table:
            embeded = tf.nn.embedding_lookup(self.embed_table, ids)
        else:
            embeded = tf.one_hot(ids, self.args.dim_output, dtype=tf.float32)

        return embeded

    def build_helper(self, type, batch_size=None, labels=None, len_labels=None,
                     encoded=None, len_encoded=None):
        """
        two types of helper:
            training: need labels, len_labels,
            infer: need batch_size
        """
        from ..tools import helpers

        batch_size = tf.size(len_labels) if len_labels is not None else batch_size
        assert batch_size is not None

        if type == 'ScheduledEmbeddingTrainingHelper':
            helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                inputs=self.embedding(labels),
                sequence_length=len_labels,
                embedding=self.embedding,
                sampling_probability=self.sample_prob)
            self.beam_size = 1
        elif type == 'ScheduledArgmaxEmbeddingTrainingHelper':
            helper = helpers.ScheduledArgmaxEmbeddingTrainingHelper(
                embedding=self.embedding,
                start_tokens=tf.fill([batch_size], self.start_token),
                end_token=self.end_token,
                softmax_temperature=self.args.model.decoder.softmax_temperature,
                sampling_probability=self.sample_prob)
            self.beam_size = 1
        elif type == 'ScheduledSelectEmbeddingHelper':
            helper = helpers.ScheduledSelectEmbeddingHelper(
                embedding=self.embedding,
                start_tokens=tf.fill([batch_size], self.start_token),
                end_token=self.end_token,
                softmax_temperature=self.args.model.decoder.softmax_temperature,
                sampling_probability=self.sample_prob)
            self.beam_size = 1
        elif type == 'TrainingHelper':
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=self.embedding(labels),
                sequence_length=len_labels,
                name='TrainingHelper')
            self.beam_size = 1
        # infer helper
        elif type == 'GreedyEmbeddingHelper':
            helper =tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=self.embedding,
                start_tokens=tf.fill([batch_size], self.start_token),
                end_token=self.end_token)
            self.beam_size = 1
        elif type == 'BeamSearchDecoder':
            helper = None
            self.beam_size = self.args.beam_size
        elif type == 'RNAGreedyEmbeddingHelper':
            helper = helpers.RNAGreedyEmbeddingHelper(
                encoded=encoded,
                len_encoded=len_encoded,
                embedding=self.embedding,
                start_tokens=tf.fill([batch_size], self.start_token))
            self.beam_size = self.args.beam_size
        else:
            raise NotImplementedError

        self.helper = helper

    @abstractmethod
    def _decode(self, encoded, len_encoded, labels, len_labels):
        '''
        Create the variables and do the forward computation to decode an entire
        sequence

        Args:
            encoded: the encoded inputs, this is a dictionary of
                [batch_size x time x ...] tensors
            encoded_seq_length: the sequence lengths of the encoded inputs
                as a dictionary of [batch_size] vectors
            targets: the targets used as decoder inputs as a dictionary of
                [batch_size x time x ...] tensors
            target_seq_length: the sequence lengths of the targets
                as a dictionary of [batch_size] vectors

        Returns:
            - the output logits of the decoder as a dictionary of
                [batch_size x time x ...] tensors
            - the logit sequence_lengths as a dictionary of [batch_size] vectors
            - the final state of the decoder as a possibly nested tupple
                of [batch_size x ... ] tensors
        '''

    @abstractmethod
    def zero_state(self, encoded_dim, batch_size):
        '''get the decoder zero state

        Args:
            encoded_dim: the dimension of the encoded sequence as a list of
                integers
            batch size: the batch size as a scalar Tensor

        Returns:
            the decoder zero state as a possibly nested tupple
                of [batch_size x ... ] tensors
        '''

    @property
    def variables(self):
        '''
        get a list of the models's variables
        '''
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=self.name + '/')

        if hasattr(self, 'wrapped'):
            #pylint: disable=E1101
            variables += self.wrapped.variables

        return variables

    @abstractmethod
    def get_output_dims(self):
        '''get the decoder output dimensions

        args:
            trainlabels: the number of extra labels the trainer needs

        Returns:
            a dictionary containing the output dimensions
        '''

    @staticmethod
    def linear_increase(prob_init, global_step, start_warmup_steps, step_increasement):
        if step_increasement > 0:
            global_step = tf.to_float(global_step)
            start_warmup_steps = tf.to_float(start_warmup_steps)
            step_increasement = tf.to_float(step_increasement)
            sample_prob = step_increasement * (global_step-start_warmup_steps) + prob_init

            return  tf.maximum(tf.minimum(1.0, sample_prob), 0.0)
        else:
            return tf.to_float(prob_init)
