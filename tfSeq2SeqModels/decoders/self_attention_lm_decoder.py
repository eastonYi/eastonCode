'''@file rnn_decoder.py
contains the general recurrent decoder class'''

import logging
import tensorflow as tf
from tfModels.tensor2tensor import common_attention
from .lm_decoder import LM_Decoder
from ..tools.utils import shift_right, embedding, residual, multihead_attention, ff_hidden, dense
from tfModels.tensor2tensor import dcommon_layers


class SelfAttentionDecoder(LM_Decoder):
    '''a speller decoder for the LAS architecture'''
    def __init__(self, args, is_train, name=None):
        self.num_cell_units = args.model.decoder.num_cell_units
        self.num_blocks = args.model.decoder.num_blocks
        self.attention_dropout_rate = args.model.decoder.attention_dropout_rate if is_train else 0.0
        self.residual_dropout_rate = args.model.decoder.residual_dropout_rate if is_train else 0.0
        self.num_heads = args.model.decoder.num_heads
        # self.beam_size = args.model.decoder.beam_size
        self.dim_output = args.dim_output
        self._ff_activation = tf.nn.relu
        self.is_train = is_train
        self.name = name
        self.args = args

    def __call__(self, inputs, len_inputs):
        with tf.variable_scope(self.name or 'decoder'):
            output = self.decoder_impl(inputs)
            logits = tf.layers.dense(
                inputs=output,
                units=self.dim_output,
                activation=None,
                use_bias=False)

        return logits

    def decoder_impl(self, decoder_input):
        # Positional Encoding
        decoder_input += common_attention.add_timing_signal_1d(decoder_input)
        # Dropout
        decoder_output = tf.layers.dropout(decoder_input,
                                           rate=self.residual_dropout_rate,
                                           training=self.is_train)
        # Bias for preventing peeping later information
        self_attention_bias = common_attention.attention_bias_lower_triangle(tf.shape(decoder_input)[1])

        # Blocks
        for i in range(self.num_blocks):
            # print('here!!!!!!!{}'.format(i))
            with tf.variable_scope("block_{}".format(i)):
                # Multihead Attention (self-attention)
                decoder_output = residual(decoder_output,
                                          multihead_attention(
                                              query_antecedent=decoder_output,
                                              memory_antecedent=None,
                                              bias=self_attention_bias,
                                              total_key_depth=self.num_cell_units,
                                              total_value_depth=self.num_cell_units,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.attention_dropout_rate,
                                              output_depth=self.num_cell_units,
                                              name="decoder_self_attention",
                                              summaries=True),
                                          dropout_rate=self.residual_dropout_rate)
                # Feed Forward
                decoder_output = residual(decoder_output,
                                          ff_hidden(
                                              decoder_output,
                                              hidden_size=4 * self.num_cell_units,
                                              output_size=self.num_cell_units,
                                              activation=self._ff_activation),
                                          dropout_rate=self.residual_dropout_rate)

        return decoder_output

    def decoder_with_caching_impl(self, decoder_input, decoder_cache):

        # decoder_input = tf.expand_dims(decoder_input, -1)
        decoder_output = self.embedding(decoder_input)

        # Positional Encoding
        decoder_output += common_attention.add_timing_signal_1d(decoder_output)

        # Dropout
        decoder_output = tf.layers.dropout(decoder_output,
                                           rate=self.residual_dropout_rate,
                                           training=self.is_train)
        new_cache = []

        # Blocks
        for i in range(self.num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                # Multihead Attention (self-attention)
                decoder_output = residual(decoder_output[:, -1:, :],
                                          multihead_attention(
                                              query_antecedent=decoder_output,
                                              memory_antecedent=None,
                                              bias=None,
                                              total_key_depth=self.num_cell_units,
                                              total_value_depth=self.num_cell_units,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.attention_dropout_rate,
                                              num_queries=1,
                                              output_depth=self.num_cell_units,
                                              name="decoder_self_attention",
                                              summaries=True),
                                          dropout_rate=self.residual_dropout_rate)

                # Feed Forward
                decoder_output = residual(decoder_output,
                                          ff_hidden(
                                              decoder_output,
                                              hidden_size=4 * self.num_cell_units,
                                              output_size=self.num_cell_units,
                                              activation=tf.nn.relu),
                                          dropout_rate=self.residual_dropout_rate)

                decoder_output = tf.concat([decoder_cache[:, :, i, :], decoder_output], axis=1)
                new_cache.append(decoder_output[:, :, None, :])

        new_cache = tf.concat(new_cache, axis=2)  # [batch_size, n_step, num_blocks, num_hidden]

        return decoder_output, new_cache

    def test_output(self, decoder_output):
        beam_size = self.beam_size
        dim_output = self.dim_output
        """During test, we only need the last prediction at each time."""
        last_logits = dense(decoder_output[:, -1], dim_output, use_bias=False,
                            kernel=self.embed_table, name='dst_softmax')
        next_pred = tf.to_int32(tf.argmax(last_logits, axis=-1))
        z = tf.nn.log_softmax(last_logits)
        next_scores, next_preds = tf.nn.top_k(z, k=beam_size, sorted=False)
        next_preds = tf.to_int32(next_preds)

        return last_logits, next_pred, next_preds, next_scores

    def sample(self, token_init=None, max_length=50):
        """sample in graph."""
        num_samples = tf.placeholder(tf.int32, [], name='num_samples')
        scores = tf.zeros([num_samples], dtype=tf.float32)
        finished = tf.zeros([num_samples], dtype=tf.bool)
        cache = tf.zeros([num_samples, 0, self._config.num_blocks, self._config.num_cell_units])

        def step(i, finished, preds, scores, cache):
            # Where are we.
            i += 1

            # Call decoder and get predictions.
            decoder_output, cache = self.decoder_with_caching(preds, cache)

            _, next_preds, next_scores = self.test_output(decoder_output)
            next_preds = next_preds[:, None, 0]
            next_scores = next_scores[:, 0]

            # Update.
            scores = scores + next_scores
            preds = tf.concat([preds, next_preds], axis=1)

            # Whether sequences finished.
            has_eos = tf.equal(next_preds[:, 0], 3)
            finished = tf.logical_or(finished, has_eos)

            return i, finished, preds, scores, cache

        def not_finished(i, finished, preds, scores, cache):
            return tf.logical_and(
                tf.reduce_any(tf.logical_not(finished)),
                tf.less_equal(
                    i,
                    max_length
                )
            )

        i, finished, preds, scores, cache = \
            tf.while_loop(cond=not_finished,
                          body=step,
                          loop_vars=[0, finished, token_init, scores, cache],
                          shape_invariants=[
                              tf.TensorShape([]),
                              tf.TensorShape([None]),
                              tf.TensorShape([None, None]),
                              tf.TensorShape([None]),
                              tf.TensorShape([None, None, None, None])],
                          back_prop=False)

        preds = preds[:, 1:]  # remove <S> flag
        return preds, num_samples
