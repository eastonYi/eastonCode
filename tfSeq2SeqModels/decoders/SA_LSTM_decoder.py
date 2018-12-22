'''@file rnn_decoder.py
contains the general recurrent decoder class'''

import logging
import tensorflow as tf
from tfModels.tensor2tensor import common_attention
from .decoder import Decoder
from tensorflow.python.util import nest
from ..tools.utils import residual, multihead_attention, ff_hidden, dense
from tfModels.tensor2tensor import dcommon_layers


class SA_LSTM_Decoder(Decoder):
    '''lstm + self-attention decoder where lstm as a normal decoder and SA as a lm embedding'''

    def _decode(self, encoded, len_encoded):
        '''
        '''
        num_cell_units_en = self.args.model.encoder.num_cell_units
        num_cell_units_de = self.args.model.decoder.num_cell_units
        num_blocks = self.args.model.decoder.num_blocks
        dim_output = self.args.dim_output
        softmax_temperature = self.args.model.decoder.softmax_temperature

        batch_size = tf.shape(len_encoded)[0]
        blank_id = dim_output-1
        token_init = tf.fill([batch_size, 1], blank_id)
        logits_init = tf.zeros([batch_size, 1, dim_output], dtype=tf.float32)
        mask_preds_init = tf.fill([batch_size, 1], False)

        self.cell = self.create_cell()
        # collect the initial states of lstms used in decoder.
        all_initial_states = {}
        cache = tf.zeros([batch_size, 0, num_blocks, num_cell_units_de])
        all_initial_states["state_decoder"] = self.zero_state(batch_size, dtype=tf.float32)

        def step(i, preds, mask_preds, all_states, cache, logits):
            state_decoder = all_states["state_decoder"]

            decoder_output, cache = self.decoder_with_caching_impl(
                decoder_input=preds,
                decoder_input_mask=mask_preds,
                decoder_cache=cache)
            decoder_input = tf.concat([encoded[:, i, :], decoder_output[:, -1, :]], axis=1)
            decoder_input.set_shape([None, num_cell_units_en + num_cell_units_de])

            # Lstm part
            with tf.variable_scope("decoder_lstms"):
                output_decoder, state_decoder = tf.contrib.legacy_seq2seq.rnn_decoder(
                    decoder_inputs=[decoder_input],
                    initial_state=state_decoder,
                    cell=self.cell)
                all_states["state_decoder"] = state_decoder
                output_decoder = [tf.concat([output_decoder[0], encoded[:, i, :]], axis=1)]

            cur_logit = tf.layers.dense(
                inputs=output_decoder[0],
                units=dim_output,
                activation=None,
                use_bias=False,
                name='fully_connected')

            if self.is_train and self.args.model.decoder.sample_decoder:
                sample_logit = cur_logit/softmax_temperature - tf.reduce_max(cur_logit, -1, keepdims=True)
                cur_ids = tf.distributions.Categorical(logits=sample_logit).sample()
            else:
                cur_ids = tf.to_int32(tf.argmax(cur_logit, -1))
            preds = tf.concat([preds, tf.expand_dims(cur_ids, 1)], axis=1)
            logits = tf.concat([logits, tf.expand_dims(cur_logit, 1)], 1)
            # Here we coarsely remove all the blank labels in the predictions, hoping to reduce the computation of self-attention
            mask_preds_cur = tf.equal(cur_ids, blank_id)
            mask_preds = tf.concat([mask_preds, tf.expand_dims(mask_preds_cur, 1)], axis=1)
            # mask_preds = tf.cond(
            #     tf.reduce_all(mask_preds_cur),
            #     lambda: mask_preds,
            #     lambda: tf.concat([mask_preds, tf.expand_dims(mask_preds_cur, 1)], axis=1))
            # preds = tf.cond(
            #     tf.reduce_all(mask_preds_cur),
            #     lambda: preds,
            #     lambda: tf.concat([preds, tf.expand_dims(cur_ids, 1)], axis=1))

            return i+1, preds, mask_preds, all_states, cache, logits

        _, preds, mask_preds, _, _, logits = tf.while_loop(
            cond=lambda i, *_: tf.less(i, tf.shape(encoded)[1]),
            body=step,
            loop_vars=[0, token_init, mask_preds_init, all_initial_states, cache, logits_init],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([None, None]),
                              tf.TensorShape([None, None]),
                              nest.map_structure(lambda t: tf.TensorShape(t.shape), all_initial_states),
                              tf.TensorShape([None, None, None, None]),
                              tf.TensorShape([None, None, dim_output])],
            back_prop=True)

        logits = logits[:, 1:, :]
        preds = preds[:, 1:]
        not_padding = tf.to_int32(tf.sequence_mask(len_encoded, maxlen=tf.shape(encoded)[1]))
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
                                              summaries=False),
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

    def create_cell(self):
        num_layers = self.args.model.decoder.num_layers
        num_cell_units_de = self.args.model.decoder.num_cell_units
        dropout = self.args.model.decoder.dropout

        cell = dcommon_layers.lstm_cells(
            num_layers,
            num_cell_units_de,
            initializer=None,
            dropout=dropout)

        return cell

    def zero_state(self, batch_size, dtype):
        return self.cell.zero_state(batch_size, dtype)
