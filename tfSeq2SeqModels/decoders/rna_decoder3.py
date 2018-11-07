'''@file rnn_decoder.py
the while_loop implementation'''

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from .decoder import Decoder
from tensorflow.python.util import nest
from tfModels.tensor2tensor import dcommon_layers
from tfModels.coldFusion import cold_fusion

class RNADecoder(Decoder):
    """language model cold fusion
    """
    def __init__(self, args, is_train, global_step, embed_table=None, name=None):
        if args.model.decoder.cold_fusion:
            from tfSeq2SeqModels.languageModel import LanguageModel
            from tfSeq2SeqModels.decoders.speller import Speller as decoder

            args.model.lm.dim_output = args.dim_output
            args.model.lm.list_gpus = args.list_gpus
            self.lm = LanguageModel(
                tensor_global_step=global_step,
                encoder=None,
                decoder=decoder,
                is_train=False,
                args=args.model.lm
            )
            self.num_cell_units_lm = args.model.decoder.num_cell_units_lm
        super().__init__(args, is_train, global_step, embed_table, name)

    def _decode(self, encoded, len_encoded):
        num_cell_units_en = self.args.model.encoder.num_cell_units
        size_embedding = self.args.model.decoder.size_embedding
        dim_output = self.args.dim_output

        batch_size = tf.shape(len_encoded)[0]
        blank_id = dim_output-1
        token_init = tf.fill([batch_size, 1], blank_id)
        logits_init = tf.zeros([batch_size, 1, dim_output], dtype=tf.float32)

        self.cell = self.create_cell()
        # collect the initial states of lstms used in decoder.
        all_initial_states = {}
        all_initial_states["state_decoder"] = self.zero_state(batch_size, dtype=tf.float32)
        if self.args.model.decoder.cold_fusion:
            all_initial_states["state_lm"] = self.lm.zero_state(batch_size, dtype=tf.float32)

        def step(i, preds, all_states, logits):
            state_decoder = all_states["state_decoder"]
            prev_emb = self.embedding(preds[:, -1])
            decoder_input = tf.concat([encoded[:, i, :], prev_emb], axis=1)
            decoder_input.set_shape([None, size_embedding + num_cell_units_en])

            # Lstm part
            with tf.variable_scope("decoder_forward"):
                output_decoder, state_decoder = tf.contrib.legacy_seq2seq.rnn_decoder(
                    decoder_inputs=[decoder_input],
                    initial_state=state_decoder,
                    cell=self.cell)
                all_states["state_decoder"] = state_decoder

                if self.args.model.decoder.cold_fusion:
                    logit_lm, state_lm = self.lm.forward(
                        input=prev_emb,
                        state=all_states["state_lm"])
                    logit_lm = tf.stop_gradient(logit_lm)
                    cur_logit = cold_fusion(
                        logit_lm=logit_lm[0],
                        state_decoder=state_decoder,
                        num_cell_units=self.num_cell_units_lm,
                        dim_output=dim_output)
                    all_states["state_lm"] = state_lm
                else:
                    cur_logit = fully_connected(
                        inputs=output_decoder[0],
                        num_outputs=dim_output)

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
                              tf.TensorShape([None, None, dim_output])]
            )

        logits = logits[:, 1:, :]
        preds = preds[:, 1:]
        not_padding = tf.to_int32(tf.sequence_mask(len_encoded))
        preds = tf.multiply(tf.to_int32(preds), not_padding)

        return logits, preds, len_encoded

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
        # num_cell_units_de = self.args.model.decoder.num_cell_units
        # num_layers = self.args.model.decoder.num_layers
        #
        # initial_states = []
        # zero_states = tf.zeros([batch_size, num_cell_units_de], dtype=dtype)
        # for i in range(num_layers):
        #     initial_states.append(tf.contrib.rnn.LSTMStateTuple(zero_states, zero_states))
        #
        # return tuple(initial_states)
