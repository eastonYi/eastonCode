'''@file rnn_decoder.py
contains the general recurrent decoder class'''

import logging
from abc import ABCMeta, abstractmethod
import tensorflow as tf
from tfModels.tensor2tensor import common_attention
from .decoder import Decoder
from tensorflow.python.util import nest
from ..tools.utils import shift_right, embedding, residual, multihead_attention, ff_hidden, dense
from tfModels.tensor2tensor import dcommon_layers


class SelfAttentionLSTMDecoder(Decoder):
    '''a speller decoder for the LAS architecture'''

    def _decode(self, encoded, len_encoded):
        '''
        '''
        num_cell_units_en = self.args.model.encoder.num_cell_units
        num_cell_units_de = self.args.model.decoder.num_cell_units
        num_blocks = self.args.model.decoder.num_blocks
        size_embedding = self.args.model.decoder.size_embedding
        num_layers = self.args.model.decoder.num_layers
        dim_output = self.args.dim_output
        dropout = self.args.model.decoder.dropout

        batch_size = tf.shape(len_encoded)[0]
        blank_id = dim_output-1
        initial_ids = tf.fill([batch_size, 1], blank_id)
        initial_logits = tf.zeros([batch_size, 1, dim_output], dtype=tf.float32)
        mask_preds_init = tf.fill([batch_size, 1], False)

        # collect the initial states of lstms used in decoder.
        all_initial_states = {}
        # the initial states of lstm which model the symbol recurrence (lm).
        initial_states = []
        zero_states = tf.zeros([batch_size, num_cell_units_de], dtype=tf.float32)
        for i in range(num_layers):
            initial_states.append(tf.contrib.rnn.LSTMStateTuple(zero_states, zero_states))
        all_initial_states["lstm_states"] = tuple(initial_states)

        tf.get_variable(
            shape=(dim_output, num_cell_units_en+size_embedding),
            name='fully_connected',
            dtype=tf.float32)

        cache = tf.zeros([batch_size, 0, num_blocks, num_cell_units_de])

        def step(i, preds, mask_preds, all_lstm_states, cache, logits):

            lstm_states = all_lstm_states["lstm_states"]

            # Concat the prediction embedding and the encoder_output
            eshape = tf.shape(encoded)
            initial_tensor = tf.zeros([eshape[0], eshape[2]])
            # initial_tensor.set_shape([None, num_cell_units_de])
            prev_encoder_output = tf.cond(tf.equal(i, 0),
                                          lambda: initial_tensor,
                                          lambda: encoded[:, i-1, :])

            decoder_output, cache = self.decoder_with_caching_impl(
                decoder_input=preds,
                decoder_input_mask=mask_preds,
                decoder_cache=cache)
            decoder_inputs = tf.concat([prev_encoder_output, decoder_output[:, -1, :]], axis=1)
            decoder_inputs.set_shape([None, num_cell_units_en + num_cell_units_de])

            # Lstm part
            with tf.variable_scope("decoder_lstms"):
                multi_lstm_cells = dcommon_layers.lstm_cells(
                    num_layers,
                    num_cell_units_de,
                    initializer=None,
                    dropout=dropout)

                lstm_outputs, lstm_states = tf.contrib.legacy_seq2seq.rnn_decoder(
                    decoder_inputs=[decoder_inputs],
                    initial_state=lstm_states,
                    cell=multi_lstm_cells)

                pre_softmax_inputs = tf.concat([lstm_outputs[0], encoded[:, i, :]], axis=1)

            # lstm_outputs: a list of outputs, using the element 0
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                var_top = tf.get_variable(name='fully_connected')
            cur_logits = tf.matmul(pre_softmax_inputs, var_top, transpose_b=True)
            cur_ids = tf.to_int32(tf.argmax(cur_logits, -1))

            # Update.
            mask_preds_cur = tf.equal(cur_ids, blank_id)
            mask_preds = tf.concat([mask_preds, tf.expand_dims(mask_preds_cur, 1)], axis=1)
            # is_empty = tf.reduce_sum(tf.to_int32(mask_preds_cur))
            # mask_preds = tf.cond(is_empty,
            #                      lambda: mask_preds,
            #                      lambda: tf.concat([mask_preds, tf.expand_dims(mask_preds_cur, 1)], axis=1))

            preds = tf.concat([preds, tf.expand_dims(cur_ids, 1)], axis=1)

            # Refresh the elements
            logits = tf.concat([logits, tf.expand_dims(cur_logits, 1)], 1)

            all_lstm_states["lstm_states"] = lstm_states

            return i+1, preds, mask_preds, all_lstm_states, cache, logits

        _, preds, mask_preds, _, _, logits = tf.while_loop(
            cond=lambda i, *_: tf.less(i, tf.shape(encoded)[1]),
            body=step,
            loop_vars=[0, initial_ids, mask_preds_init, all_initial_states, cache, initial_logits],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([None, None]),
                              tf.TensorShape([None, None]),
                              nest.map_structure(lambda t: tf.TensorShape(t.shape), all_initial_states),
                              tf.TensorShape([None, None, None, None]),
                              tf.TensorShape([None, None, dim_output])],
            back_prop=True)

        logits = logits[:, 1:, :]
        preds = preds[:, 1:]
        not_padding = tf.to_int32(tf.sequence_mask(len_encoded))
        preds = tf.multiply(tf.to_int32(preds), not_padding)

        return logits, preds, len_encoded

    def decoder_with_caching_impl(self, decoder_input, decoder_input_mask, decoder_cache):

        attention_dropout_rate = self.args.model.decoder.attention_dropout_rate if self.is_train else 0.0
        residual_dropout_rate = self.args.model.decoder.residual_dropout_rate if self.is_train else 0.0
        num_blocks = self.args.model.decoder.num_blocks
        num_heads = self.args.model.decoder.num_heads
        num_cell_units_de = self.args.model.decoder.num_cell_units

        decoder_attention_bias = common_attention.attention_bias_ignore_padding(decoder_input_mask)

        decoder_output = self.embedding(decoder_input)

        # Positional Encoding
        decoder_output += common_attention.add_timing_signal_1d(decoder_output)

        # Dropout
        decoder_output = tf.layers.dropout(decoder_output,
                                           rate=residual_dropout_rate,
                                           training=self.is_train)
        new_cache = []

        # Blocks
        for i in range(num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                # Multihead Attention (self-attention)
                decoder_output = residual(decoder_output[:, -1:, :],
                                          multihead_attention(
                                              query_antecedent=decoder_output,
                                              memory_antecedent=None,
                                              bias=decoder_attention_bias,
                                              total_key_depth=num_cell_units_de,
                                              total_value_depth=num_cell_units_de,
                                              num_heads=num_heads,
                                              dropout_rate=attention_dropout_rate,
                                              num_queries=1,
                                              output_depth=num_cell_units_de,
                                              name="decoder_self_attention",
                                              summaries=True),
                                          dropout_rate=residual_dropout_rate)

                # Feed Forward
                decoder_output = residual(decoder_output,
                                          ff_hidden(
                                              decoder_output,
                                              hidden_size=4 * num_cell_units_de,
                                              output_size=num_cell_units_de,
                                              activation=tf.nn.relu),
                                          dropout_rate=residual_dropout_rate)

                decoder_output = tf.concat([decoder_cache[:, :, i, :], decoder_output], axis=1)
                new_cache.append(decoder_output[:, :, None, :])

        new_cache = tf.concat(new_cache, axis=2)  # [batch_size, n_step, num_blocks, num_hidden]

        return decoder_output, new_cache

    def test_output(self, decoder_output):
        beam_size = self.args.beam_size
        dim_output = self.args.dim_output
        """During test, we only need the last prediction at each time."""
        last_logits = dense(decoder_output[:, -1], dim_output, use_bias=False,
                            kernel=self.embed_table, name='dst_softmax')
        next_pred = tf.to_int32(tf.argmax(last_logits, axis=-1))
        z = tf.nn.log_softmax(last_logits)
        next_scores, next_preds = tf.nn.top_k(z, k=beam_size, sorted=False)
        next_preds = tf.to_int32(next_preds)

        return last_logits, next_pred, next_preds, next_scores
