'''@file rnn_decoder.py
the while_loop implementation'''

import tensorflow as tf
from .rna_decoder3 import RNADecoder
from tensorflow.python.util import nest
from tfModels.coldFusion import cold_fusion


class CTC_LM_Decoder(RNADecoder):
    """language model cold fusion
    """

    def __init__(self, args, is_train, global_step, embed_table=None, name=None):
        super().__init__(args, is_train, global_step, embed_table, name)
        self.num_layers = args.model.decoder2.num_layers
        self.num_cell_units_de = args.model.decoder2.num_cell_units
        self.dropout = args.model.decoder2.dropout
        self.num_cell_units_en = args.dim_output
        self.size_embedding = args.model.decoder2.size_embedding
        self.dim_output = args.dim_output
        self.softmax_temperature = args.model.decoder2.softmax_temperature

    def _decode(self, encoded, len_encoded):
        batch_size = tf.shape(len_encoded)[0]
        blank_id = self.dim_output-1
        token_init = tf.fill([batch_size, 1], blank_id)
        logits_init = tf.zeros([batch_size, 1, self.dim_output], dtype=tf.float32)

        self.cell = self.create_cell()
        # collect the initial states of lstms used in decoder.
        all_initial_states = {}
        all_initial_states["state_decoder"] = self.zero_state(batch_size, dtype=tf.float32)
        if self.args.model.decoder.cold_fusion:
            all_initial_states["state_lm"] = self.lm.zero_state(batch_size, dtype=tf.float32)
            all_initial_states["logit_lm"] = tf.zeros([batch_size, self.dim_output], dtype=tf.float32)

        def step(i, preds, all_states, logits):
            state_decoder = all_states["state_decoder"]
            prev_emb = self.embedding(preds[:, -1])
            decoder_input = tf.concat([encoded[:, i, :], prev_emb], axis=1)
            decoder_input.set_shape([None, self.size_embedding + self.num_cell_units_en])

            # Lstm part
            with tf.variable_scope("decoder_lstms"):
                output_decoder, state_decoder = tf.contrib.legacy_seq2seq.rnn_decoder(
                    decoder_inputs=[decoder_input],
                    initial_state=state_decoder,
                    cell=self.cell)
                all_states["state_decoder"] = state_decoder
                # output_decoder = [tf.concat([output_decoder[0], encoded[:, i, :]], axis=1)]

            if self.args.model.decoder.cold_fusion:
                logit_lm, state_lm = self.lm.forward(
                    input=preds[:, -1],
                    state=all_states["state_lm"],
                    stop_gradient=True)

                logit_lm, state_lm = self.update_lm(
                    preds=preds[:, -1],
                    blank_id=blank_id,
                    logit_lm=logit_lm[0],
                    state_lm=state_lm,
                    logit_lm_pre=all_initial_states["logit_lm"],
                    state_lm_pre=all_initial_states["state_lm"],
                    num_cell_units_lm=self.args.model.lm.model.decoder.num_cell_units)

                all_initial_states["logit_lm"] = logit_lm
                all_states["state_lm"] = state_lm

                cur_logit = cold_fusion(
                    logit_lm=logit_lm,
                    state_decoder=state_decoder,
                    num_cell_units=self.num_cell_units_lm,
                    dim_output=self.dim_output)
            else:
                cur_logit = tf.layers.dense(
                    inputs=output_decoder[0],
                    units=self.dim_output,
                    activation=None,
                    use_bias=False,
                    name='fully_connected'
                    )

            if self.is_train and self.args.model.decoder.sample_decoder:
                cur_ids = tf.distributions.Categorical(logits=cur_logit/self.softmax_temperature).sample()
            else:
                cur_ids = tf.to_int32(tf.argmax(cur_logit, -1))
            preds = tf.concat([preds, tf.expand_dims(cur_ids, 1)], axis=1)
            logits = tf.concat([logits, tf.expand_dims(cur_logit, 1)], 1)

            return i+1, preds, all_states, logits

        _, preds, _, logits = tf.while_loop(
            cond=lambda i, *_: tf.less(i, tf.shape(encoded)[1]),
            body=step,
            loop_vars=[0, token_init, all_initial_states, logits_init],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([None, None]),
                              nest.map_structure(lambda t: tf.TensorShape(t.shape), all_initial_states),
                              tf.TensorShape([None, None, self.dim_output])]
            )

        logits = logits[:, 1:, :]
        preds = preds[:, 1:]
        not_padding = tf.to_int32(tf.sequence_mask(len_encoded, maxlen=tf.shape(encoded)[1]))
        preds = tf.multiply(tf.to_int32(preds), not_padding)

        return logits, preds, len_encoded
