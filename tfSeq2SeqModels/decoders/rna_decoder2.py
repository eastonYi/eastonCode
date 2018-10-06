'''
@file speller.py
contains the speller functionality
'''

import tensorflow as tf
from tensorflow.python.util import nest

from .decoder import Decoder
from tfModels.tensor2tensor import dcommon_layers

class RNADecoder(Decoder):

    def _decode(self, encoder_outputs, len_encoded):
        """Decode RNA outputs from encoder outputs and the decoder prediction of last step.

        Args:
          encoder_outputs: Encoder representation.
            [batch_size, input_length/stride, hidden_dim]
          hparams: hyper-parameters for model.

        Returns:
          decoder_output: Decoder representation.
            [batch_size, input_length/stride, 1, 1, vocab_size]
        """
        # While loop initialization
        num_cell_units = self.args.model.decoder.num_cell_units
        num_layers = self.args.model.decoder.num_layers
        size_embedding = self.args.model.decoder.size_embedding
        dim_output = self.args.dim_output
        batch_size = tf.shape(len_encoded)[0]
        decode_length = tf.shape(encoder_outputs)[1]

        initial_ids = tf.fill([batch_size], dim_output-1)
        initial_logits = tf.zeros([batch_size, 1, dim_output], dtype=tf.float32)

        # collect the initial states of lstms used in decoder.
        all_initial_states = {}

        # the initial states of lstm which model the symbol recurrence (lm).
        initial_states = []
        zero_states = tf.zeros([batch_size, num_cell_units], dtype=tf.float32)

        tf.get_variable(
            shape=(dim_output, num_cell_units+size_embedding),
            name='fully_connected',
            dtype=tf.float32)

        for i in range(num_layers):
            initial_states.append(tf.contrib.rnn.LSTMStateTuple(zero_states, zero_states))
        all_initial_states["lstm_states"] = tuple(initial_states)

        # Loop body
        def inner_loop(i, prev_ids, prev_states, logits, att_results=None):
            return symbols_to_logits_fn(
                i,
                prev_ids,
                prev_states,
                logits,
                encoder_outputs,
                self.embedding,
                self.args,
                att_results)

        # The While loop
        _, _, _, logits = tf.while_loop(
            cond=lambda i, *_: tf.less(i, decode_length),
            body=inner_loop,
            loop_vars=[tf.constant(0), initial_ids, all_initial_states, initial_logits],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None]),
                nest.map_structure(lambda t: tf.TensorShape(t.shape), all_initial_states),
                tf.TensorShape([None, None, dim_output])
                ]
            )
        # logits = tf.expand_dims(tf.expand_dims(logits[:, 1:, :], 2), 3)
        # decoded_ids = tf.argmax(tf.squeeze(logits, [2, 3]), -1)
        logits = logits[:, 1:, :]
        decoded_ids = tf.argmax(logits, -1)
        not_padding = tf.to_int32(tf.sequence_mask(len_encoded))
        decoded_ids = tf.multiply(tf.to_int32(decoded_ids), not_padding)

        return logits, decoded_ids, len_encoded


def symbols_to_logits_fn(i, prev_ids, all_lstm_states, logits, encoder_outputs, embedding, args, prev_att_results=None):
    """ Predicting current logits based on previous

    Args:
    i: loop index
    prev_ids: The ids of previous step. [batch_size]
    "lstm1_states": ([batch_size, decode_lstm_cells],
                     [batch_size, decode_lstm_cells])
    logits: The concatanated logits of previous steps. [batch_size, None, vocab_size]
    encoder_outputs: The outputs of encoder.

    Returns:
    i, next_ids, next_states, logits.

    """
    num_cell_units_en = args.model.decoder.num_cell_units
    num_cell_units_de = args.model.decoder.num_cell_units
    num_layers = args.model.decoder.num_layers
    size_embedding = args.model.decoder.size_embedding
    dropout = args.model.decoder.dropout
    dim_output = args.dim_output
    # Get the previous states of lstms.
    lstm_states = all_lstm_states["lstm_states"]

    # Get the embedding of previous prediction labels.
    # prev_emb = tf.gather(embedding, prev_ids, axis=0)
    prev_emb = embedding(prev_ids)

    # Concat the prediction embedding and the encoder_output
    eshape = tf.shape(encoder_outputs)
    initial_tensor = tf.zeros([eshape[0], eshape[2]])
    initial_tensor.set_shape([None, num_cell_units_en])
    prev_encoder_output = tf.cond(tf.equal(i, 0),
                                  lambda: initial_tensor,
                                  lambda: encoder_outputs[:, i - 1, :])
    decoder_inputs = tf.concat([prev_encoder_output, prev_emb], axis=1)
    decoder_inputs.set_shape([None, num_cell_units_en + size_embedding])


    # Lstm part
    with tf.variable_scope("decoder_lstms"):
        multi_lstm_cells = dcommon_layers.lstm_cells(
            num_layers,
            num_cell_units_de,
            initializer=None,
            dropout=dropout)
        prev_lstm_states = lstm_states
        lstm_outputs, lstm_states = tf.contrib.legacy_seq2seq.rnn_decoder(
            decoder_inputs=[decoder_inputs],
            initial_state=lstm_states,
            cell=multi_lstm_cells)

        pre_softmax_inputs = tf.concat([lstm_outputs[0], encoder_outputs[:, i, :]], axis=1)

    # lstm_outputs: a list of outputs, using the element 0
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        var_top = tf.get_variable(name='fully_connected')
    cur_logits = tf.matmul(pre_softmax_inputs, var_top, transpose_b=True)

    # Refresh the elements
    cur_ids = tf.to_int32(tf.argmax(cur_logits, -1))
    logits = tf.concat([logits, tf.expand_dims(cur_logits, 1)], 1)

    all_lstm_states["lstm_states"] = lstm_states

    return i + 1, cur_ids, all_lstm_states, logits
