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
    def __init__(self, args, is_train, embed_table=None, name=None):
        self.num_cell_units = args.model.decoder.num_cell_units
        self.num_blocks = args.model.decoder.num_blocks
        self.attention_dropout_rate = args.model.decoder.attention_dropout_rate if is_train else 0.0
        self.residual_dropout_rate = args.model.decoder.residual_dropout_rate if is_train else 0.0
        self.num_heads = args.model.decoder.num_heads
        self.share_embedding = args.model.decoder.share_embedding
        self.embed_table = embed_table
        # self.beam_size = args.model.decoder.beam_size
        self.dim_output = args.dim_output
        self._ff_activation = tf.nn.relu
        self.is_train = is_train
        self.name = name
        self.args = args

    def __call__(self, inputs, len_inputs):
        with tf.variable_scope(self.name or 'decoder'):
            self.scope = tf.get_variable_scope()
            output = self.decoder_impl(inputs)
            if self.embed_table is None or (not self.share_embedding):
                logits = tf.layers.dense(
                    inputs=output,
                    units=self.dim_output,
                    use_bias=False)
                # preserve the fully_connected for reuse
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    self.fully_connected = tf.get_variable('dense/kernel')
            else:
                # not recomend
                kernel = self.embed_table
                logits = dense(
                    inputs=output,
                    output_size=self.dim_output,
                    kernel=kernel,
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
        # Positional Encoding
        decoder_input += common_attention.add_timing_signal_1d(decoder_input)
        # Dropout
        decoder_output = tf.layers.dropout(decoder_input,
                                           rate=self.residual_dropout_rate,
                                           training=self.is_train)
        new_cache = []

        # Blocks
        for i in range(self.num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                # Multihead Attention (self-attention)
                # the caching_impl only need to calculate decoder_output[:, -1:, :]!
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
                                              summaries=False),
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

    def sample(self, token_init=None, state_init=None, max_length=50):
        """sample in graph.
        utilize the `decoder_with_caching_impl`
        """
        num_samples = tf.placeholder(tf.int32, [], name='num_samples')
        scores = tf.zeros([num_samples], dtype=tf.float32)
        cache = tf.zeros([num_samples, 0, self.num_blocks, self.num_cell_units])
        if token_init is None:
            token_init = tf.ones([num_samples, 1], dtype=tf.int32) * self.args.sos_idx

        def step(i, preds, scores, cache):
            # Call decoder and get predictions.
            preds_emb = tf.nn.embedding_lookup(self.embed_table, preds)
            decoder_output, cache = self.decoder_with_caching_impl(preds_emb, cache)
            cur_logit = dense(
                inputs=decoder_output[:, -1, :],
                units=self.dim_output,
                kernel=tf.transpose(self.fully_connected),
                use_bias=False)

            # Update.
            next_pred = tf.distributions.Categorical(logits=cur_logit).sample()
            # next_pred = tf.to_int32(tf.argmax(cur_logit, axis=-1))
            preds = tf.concat([preds, next_pred[:, None]], -1)

            return i+1, preds, scores, cache

        i, preds, scores, cache = \
            tf.while_loop(cond=lambda i, *_: tf.less(i, max_length),
                          body=step,
                          loop_vars=[0, token_init, scores, cache],
                          shape_invariants=[
                              tf.TensorShape([]),
                              tf.TensorShape([None, None]),
                              tf.TensorShape([None]),
                              tf.TensorShape([None, None, None, None])],
                          back_prop=False)

        preds = preds[:, 1:]  # remove <S> flag
        return preds, num_samples

    def score(self, decoder_input, len_seqs):
        '''
        decoder_input : <sos> + sent + <eos>
        score batch sentences
        utilize the `decoder_impl`
        return batch_score(log scale)
        '''

        eps = 1e-10
        decoder_input = tf.to_int32(decoder_input)
        # input is `<sos> + sent`
        decoder_input_embed = tf.nn.embedding_lookup(self.embed_table, decoder_input[:, :-1])
        hidden_output = self.decoder_impl(decoder_input_embed)
        # reuse the `fully_connected`
        logits = dense(
            inputs=hidden_output,
            units=self.dim_output,
            kernel=tf.transpose(self.fully_connected),
            use_bias=False)
        distribution = tf.nn.softmax(logits, -1)
        # output is `sent + <eos>`
        scores = tf.gather_nd(distribution, self.tensor2indices(decoder_input[:, 1:]))
        mask = tf.sequence_mask(len_seqs, maxlen=tf.shape(decoder_input)[1]-1, dtype=scores.dtype)
        scores = scores * mask + eps
        scores_log = tf.log(scores)
        scores_log = tf.reduce_sum(scores_log, -1)

        return scores_log, distribution

    @staticmethod
    def tensor2indices(batch_sents):
        """
        batch_sents = tf.constant([[2,3,3,1,4],
                                   [4,3,5,1,0]], dtype=tf.int32)
        sess.run(tensor2indices(batch_sents))
        >>>
        array([[[0, 0, 2],
                [0, 1, 3],
                [0, 2, 3],
                [0, 3, 1],
                [0, 4, 4]],

               [[1, 0, 4],
                [1, 1, 3],
                [1, 2, 5],
                [1, 3, 1],
                [1, 4, 0]]], dtype=int32)
        """
        size_batch = tf.shape(batch_sents)[0]
        len_batch = tf.shape(batch_sents)[1]
        batch_i = tf.range(size_batch)
        len_i = tf.range(len_batch)

        # [0,0,0,1,1,1,2,2,2,...]
        batch_i = tf.tile(batch_i[:, None], [1, len_batch])
        # [0,1,2,0,1,2,0,1,2,...]
        len_i = tf.tile(len_i[None, :], [size_batch, 1])

        indices = tf.stack([batch_i, len_i, batch_sents], -1)

        return indices
