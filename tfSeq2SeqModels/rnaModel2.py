import tensorflow as tf
import logging
from tensorflow.contrib.layers import fully_connected
from tfModels.tools import choose_device
from tfTools.tfTools import dense_sequence_to_sparse
from tfModels.math_tf import sum_log

from tfSeq2SeqModels.seq2seqModel import Seq2SeqModel


class RNAModel(Seq2SeqModel):
    num_Instances = 0

    def __init__(self, tensor_global_step, encoder, decoder, is_train, args,
                 batch=None, embed_table_encoder=None, embed_table_decoder=None,
                 name='RNA_Model'):
        super().__init__(tensor_global_step, encoder, decoder, is_train, args,
                     batch, None, embed_table_decoder, name)

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        tf.get_variable_scope().set_initializer(tf.variance_scaling_initializer(
            1.0, mode="fan_avg", distribution="uniform"))
        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            encoder = self.gen_encoder(
                is_train=self.is_train,
                args=self.args)
            decoder = self.gen_decoder(
                is_train=self.is_train,
                embed_table=self.embed_table_decoder,
                global_step=self.global_step,
                args=self.args)

            encoded, len_encoded = encoder(
                features=tensors_input.feature_splits[id_gpu],
                len_feas=tensors_input.len_fea_splits[id_gpu])

            self.decode_cell = decoder.create_cell()

            if self.is_train:
                loss = self.rna_loss(
                    labels=tensors_input.label_splits[id_gpu],
                    len_labels=tensors_input.len_label_splits[id_gpu],
                    encoded=encoded,
                    len_encoded=len_encoded)

                with tf.name_scope("gradients"):
                    gradients = self.optimizer.compute_gradients(loss)
            else:
                logits, sample_id, _ = decoder(encoded, len_encoded)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        if self.is_train:
            return loss, gradients
        else:
            return logits, len_decoded, sample_id

    def build_infer_graph(self):
        tensors_input = self.build_infer_input()

        with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model)):
            logits, len_logits, sample_id = self.build_single_graph(
                id_gpu=0,
                name_gpu=self.list_gpu_devices[0],
                tensors_input=tensors_input)

            decoded_sparse = self.rna_decode(logits, len_logits)
            decoded = tf.sparse_to_dense(
                sparse_indices=decoded_sparse.indices,
                output_shape=decoded_sparse.dense_shape,
                sparse_values=decoded_sparse.values,
                default_value=0,
                validate_indices=True)
            distribution = tf.nn.softmax(logits)

        return decoded, tensors_input.shape_batch, distribution

    def rna_decode(self, logits=None, len_logits=None, beam_reserve=False):
        beam_size = self.args.beam_size
        logits_timeMajor = tf.transpose(logits, [1, 0, 2])

        if beam_size == 1:
            decoded_sparse = tf.to_int32(tf.nn.ctc_greedy_decoder(
                logits_timeMajor,
                len_logits,
                merge_repeated=False)[0][0])
        else:
            if beam_reserve:
                decoded_sparse = tf.nn.ctc_beam_search_decoder(
                    logits_timeMajor,
                    len_logits,
                    beam_width=beam_size,
                    merge_repeated=False)[0]
            else:
                decoded_sparse = tf.to_int32(tf.nn.ctc_beam_search_decoder(
                    logits_timeMajor,
                    len_logits,
                    beam_width=beam_size,
                    merge_repeated=False)[0][0])

        return decoded_sparse

    def rna_loss(self, encoded, len_encoded, labels, len_labels):
        """
        using dynamic programming, which generate a lattice form seq_input to seq_label_lens
        each node is a state of the decoder. Feeding an input and label to push a movement
        on the lattic.
        """
        num_cell_units_en = self.args.model.encoder.num_cell_units
        num_cell_units_de = self.args.model.decoder.num_cell_units
        size_embedding = self.args.model.decoder.size_embedding
        dim_output = self.args.dim_output

        LOG_ZERO = tf.convert_to_tensor(-1e29) # can not be too small in tf
        LOG_ONE = tf.convert_to_tensor(0.0)
        batch_size = tf.shape(len_encoded)[0]
        len_seq = tf.shape(labels)[1]
        blank_id = dim_output-1
        blanks = tf.fill([batch_size, 1], blank_id)

        tail = tf.ones((batch_size, len_seq-1), dtype=tf.float32) * LOG_ZERO
        head = tf.ones((batch_size, 1), dtype=tf.float32) * LOG_ONE
        forward_vars_init = tf.concat([head, tail], -1)

        encoded_timeMajor = tf.transpose(encoded, ((1, 0, 2)))

        def step(i, forward_vars, multi_states):
            eshape = tf.shape(encoded)
            initial_tensor = tf.zeros([eshape[0], eshape[2]])
            initial_tensor.set_shape([None, num_cell_units_de])
            prev_encoder_output = tf.cond(tf.equal(i, 0),
                                          lambda: initial_tensor,
                                          lambda: encoded[:, i-1, :])
            embedding_label = self.embedding(labels[:, i])
            embedding_blank = self.embedding(blanks)
            step_input_1 = tf.concat([prev_encoder_output, embedding_blank], 2)
            step_input_2 = tf.concat([prev_encoder_output, embedding_label], 2)
            step_input_1.set_shape([None, num_cell_units_en + size_embedding])
            step_input_2.set_shape([None, num_cell_units_en + size_embedding])

            multi_logits_1, multi_states_1 = self.multi_states_froward(multi_states, step_input_1)
            multi_logits_2, multi_states_2 = self.multi_states_froward(multi_states, step_input_2)

            multi_logits_2 = self.right_shift_rows(multi_logits_2, 1, LOG_ZERO)
            multi_states_2 = self.right_shift_rows(multi_states_2, 1, LOG_ZERO)
            forward_vars_2 = self.right_shift_rows(forward_vars, 1, LOG_ZERO)

            multi_states = self.choose_state(multi_states_1, multi_states_2)
            prob_1 = self.logit2prob(multi_logits_1)
            prob_2 = self.logit2prob(multi_logits_2)

            forward_vars = sum_log(forward_vars + prob_1, forward_vars_2 + prob_2)

            return i+1, forward_vars, multi_states

        _, forward_vars, _,  = tf.while_loop(
            cond=lambda i, *_: tf.less(i, tf.shape(encoded)[1]),
            body=step,
            loop_vars=[0, forward_vars_init, multi_states_init],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([None, None]),
                              nest.map_structure(lambda t: tf.TensorShape(t.shape), all_initial_states)])
        indices_batch = tf.range(batch_size)
        indices_time = len_encoded-1
        res = tf.gather_nd(forward_vars, tf.stack([indices_time, indices_batch, len_labels], -1))

        return -res

    def multi_states_froward(self, multi_states, step_input):
        # TODO
        multi_outputs, multi_states = tf.contrib.legacy_seq2seq.rnn_decoder(
            decoder_inputs=[step_input],
            initial_state=multi_states,
            cell=self.decode_cell)

        multi_logits = fully_connected(
            inputs=multi_outputs,
            num_outputs=self.args.dim_output,
            scope='fc_layer')

        return multi_logits, multi_states

    @staticmethod
    def choose_state(multi_state_1, multi_state_2, prob, prob_pre, forward_vars, forward_vars_pre):
        # sum_log(prob, forward_vars), sum_log(prob_pre, forward_vars_pre)
        # TODO
        return multi_state_2

    @staticmethod
    def right_shift_rows(p, shift, pad):
        assert type(shift) is int
        return tf.concat([tf.ones((tf.shape(p)[0], shift), dtype=tf.float32)*pad,
                          p[:, :-shift]], axis=1)

    def logit2prob(logit, labels):
        """
        Args:
            logit: which is the distribution over vocab
        Return:
            prob: which is the log-prob of target sequence
        """
        logit_log = tf.log(logit)
        size_batch, len_time = tf.shape(logit)[0], tf.shape(logit)[1]
        len_seq = tf.shape(labels)[1]
        m = tf.tile(tf.expand_dims(tf.range(len_time, dtype=tf.int32), 1), [1, len_seq])
        index_time = tf.tile(tf.expand_dims(m, 0), [size_batch, 1, 1])
        index_labels = tf.tile(tf.expand_dims(labels, 1), [1, len_time, 1])
        n = tf.expand_dims(tf.expand_dims(tf.range(size_batch, dtype=tf.int32), 1), 1)
        index_batch = tf.tile(n, [1, len_time, len_seq])
        probs = tf.gather_nd(logit_log, tf.stack((index_batch, index_time, index_labels), -1))

        probs_log = tf.log(probs)

        return probs_log
