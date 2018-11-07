'''@file rnn_decoder.py
the while_loop implementation'''

import tensorflow as tf
from .decoder import Decoder
from tensorflow.python.util import nest
from tfModels.tensor2tensor import dcommon_layers


class RNADecoder(Decoder):

    def _decode(self, encoded, len_encoded):
        num_cell_units_en = self.args.model.encoder.num_cell_units
        num_cell_units_de = self.args.model.decoder.num_cell_units
        size_embedding = self.args.model.decoder.size_embedding
        num_layers = self.args.model.decoder.num_layers
        dim_output = self.args.dim_output
        dropout = self.args.model.decoder.dropout

        batch_size = tf.shape(len_encoded)[0]
        blank_id = dim_output-1
        token_init = tf.fill([batch_size, 1], blank_id)
        logits_init = tf.zeros([batch_size, 1, dim_output], dtype=tf.float32)

        # collect the initial states of lstms used in decoder.
        all_initial_states = {}
        initial_states = []
        zero_states = tf.zeros([batch_size, num_cell_units_de], dtype=tf.float32)
        for i in range(num_layers):
            initial_states.append(tf.contrib.rnn.LSTMStateTuple(zero_states, zero_states))
        all_initial_states["lstm_states"] = tuple(initial_states)

        tf.get_variable(
            shape=(dim_output, num_cell_units_en+num_cell_units_de),
            name='fully_connected',
            dtype=tf.float32)

        def step(i, preds, all_lstm_states, logits):

            lstm_states = all_lstm_states["lstm_states"]

            eshape = tf.shape(encoded)
            initial_tensor = tf.zeros([eshape[0], eshape[2]])
            initial_tensor.set_shape([None, num_cell_units_de])
            prev_encoder_output = tf.cond(tf.equal(i, 0),
                                          lambda: initial_tensor,
                                          lambda: encoded[:, i-1, :])

            prev_emb = self.embedding(preds[:, -1])
            decoder_inputs = tf.concat([prev_encoder_output, prev_emb], axis=1)
            decoder_inputs.set_shape([None, num_cell_units_en + size_embedding])
            if size_embedding < 2:
                decoder_inputs = decoder_inputs[:, :num_cell_units_en]

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

            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                var_top = tf.get_variable(name='fully_connected')
            cur_logits = tf.matmul(pre_softmax_inputs, var_top, transpose_b=True)
            cur_ids = tf.to_int32(tf.argmax(cur_logits, -1))
            preds = tf.concat([preds, tf.expand_dims(cur_ids, 1)], axis=1)
            logits = tf.concat([logits, tf.expand_dims(cur_logits, 1)], 1)

            all_lstm_states["lstm_states"] = lstm_states

            return i+1, preds, all_lstm_states, logits

        _, preds, _, logits = tf.while_loop(
            cond=lambda i, *_: tf.less(i, tf.shape(encoded)[1]),
            body=step,
            loop_vars=[0, token_init, all_initial_states, logits_init],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([None, None]),
                              nest.map_structure(lambda t: tf.TensorShape(t.shape), all_initial_states),
                              tf.TensorShape([None, None, dim_output])]
            )

        logits = logits[:, 1:, :]
        preds = preds[:, 1:]
        not_padding = tf.to_int32(tf.sequence_mask(len_encoded))
        preds = tf.multiply(tf.to_int32(preds), not_padding)

        return logits, preds, len_encoded
