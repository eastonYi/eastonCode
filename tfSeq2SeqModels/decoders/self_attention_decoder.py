'''@file rnn_decoder.py
contains the general recurrent decoder class'''

import logging
from abc import ABCMeta, abstractmethod
import tensorflow as tf
from tfModels.tensor2tensor import common_attention
from .decoder import Decoder
from ..tools.utils import shift_right, embedding, residual, multihead_attention, ff_hidden, dense
from tfModels.tensor2tensor import dcommon_layers


class SelfAttentionDecoder(Decoder):
    '''a speller decoder for the LAS architecture'''

    def _decode(self, encoded, len_encoded):
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
        hidden_units = self.args.model.decoder.num_cell_units
        num_blocks = self.args.model.decoder.num_blocks
        size_embedding = self.args.model.decoder.size_embedding
        dim_output = self.args.dim_output

        batch_size = tf.shape(len_encoded)[0]
        initial_ids = tf.fill([batch_size, 1], dim_output-1)
        initial_logits = tf.zeros([batch_size, 1, dim_output], dtype=tf.float32)
        scores = tf.zeros([batch_size], dtype=tf.float32)
        cache = tf.zeros([batch_size, 0, num_blocks, hidden_units])

        def step(i, preds, scores, cache, logits):
            i += 1
            # Concat the prediction embedding and the encoder_output
            eshape = tf.shape(encoded)
            initial_tensor = tf.zeros([eshape[0], eshape[2]])
            # initial_tensor.set_shape([None, hidden_units])
            prev_encoder_output = tf.cond(tf.equal(i, 0),
                                          lambda: initial_tensor,
                                          lambda: encoded[:, i-1, :])

            decoder_output, cache = self.decoder_with_caching_impl(
                decoder_input=preds,
                encoder_output=encoded,
                decoder_cache=cache)
            last_logit, _, next_preds, next_scores = self.test_output(decoder_output)
            next_preds = next_preds[:, None, 0]
            next_scores = next_scores[:, 0]

            # Update.
            scores = scores + next_scores
            preds = tf.concat([preds, next_preds], axis=1)
            logits = tf.concat([logits, tf.expand_dims(last_logit, axis=1)], axis=1)

            return i, preds, scores, cache, logits

        _, preds, scores, cache, logits = tf.while_loop(
            cond=lambda i, *_: tf.less(i, tf.shape(encoded)[1]),
            body=step,
            loop_vars=[0, initial_ids, scores, cache, initial_logits],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([None, None]),
                              tf.TensorShape([None]),
                              tf.TensorShape([None, None, None, None]),
                              tf.TensorShape([None, None, dim_output])],
            back_prop=True)

        logits = logits[:, 1:, :]
        # decoded_ids = tf.argmax(logits, -1)
        decoded_ids = preds
        not_padding = tf.to_int32(tf.sequence_mask(len_encoded))
        decoded_ids = tf.multiply(tf.to_int32(decoded_ids), not_padding)

        return logits, decoded_ids, len_encoded

    def decoder_with_caching_impl(self, decoder_input, encoder_output, decoder_cache):

        attention_dropout_rate = self.args.model.decoder.attention_dropout_rate if self.is_train else 0.0
        residual_dropout_rate = self.args.model.decoder.residual_dropout_rate if self.is_train else 0.0
        num_blocks = self.args.model.decoder.num_blocks
        num_heads = self.args.model.decoder.num_heads
        hidden_units = self.args.model.decoder.num_cell_units

        # decoder_input = tf.expand_dims(decoder_input, -1)
        decoder_output = self.embedding(decoder_input)

        # Positional Encoding
        decoder_output += common_attention.add_timing_signal_1d(decoder_output)
        decoder_output = tf.concat([encoder_output, decoder_output], axis=1)

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
                                              bias=None,
                                              total_key_depth=hidden_units,
                                              total_value_depth=hidden_units,
                                              num_heads=num_heads,
                                              dropout_rate=attention_dropout_rate,
                                              num_queries=1,
                                              output_depth=hidden_units,
                                              name="decoder_self_attention",
                                              summaries=True),
                                          dropout_rate=residual_dropout_rate)

                # Feed Forward
                decoder_output = residual(decoder_output,
                                          ff_hidden(
                                              decoder_output,
                                              hidden_size=4 * hidden_units,
                                              output_size=hidden_units,
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
