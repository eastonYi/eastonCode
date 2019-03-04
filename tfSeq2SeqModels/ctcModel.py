import tensorflow as tf
import logging
import sys

from tfModels.tools import choose_device
from tfSeq2SeqModels.seq2seqModel import Seq2SeqModel
from tfTools.tfTools import dense_sequence_to_sparse
from tfModels.regularization import confidence_penalty

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class CTCModel(Seq2SeqModel):
    def __init__(self, tensor_global_step, encoder, decoder, is_train, args,
                 batch=None, embed_table_encoder=None, embed_table_decoder=None,
                 name='tf_CTC_Model'):
        self.sample_prob = tf.convert_to_tensor(0.0)
        self.ctc_merge_repeated = args.model.decoder.ctc_merge_repeated
        super().__init__(tensor_global_step, encoder, decoder, is_train, args,
                     batch, None, None, name)

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        tf.get_variable_scope().set_initializer(tf.variance_scaling_initializer(
            1.0, mode="fan_avg", distribution="uniform"))
        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            # create encoder obj
            encoder = self.gen_encoder(
                is_train=self.is_train,
                args=self.args)
            decoder = self.gen_decoder(
                is_train=self.is_train,
                embed_table=None,
                global_step=self.global_step,
                args=self.args)
            # using encoder to encode the inout sequence
            hidden_output, len_hidden_output = encoder(
                features=tensors_input.feature_splits[id_gpu],
                len_feas=tensors_input.len_fea_splits[id_gpu])
            logits, align, len_logits = decoder(hidden_output, len_hidden_output)

            from tfModels.CTCShrink import acoustic_hidden_shrink_tf
            hidden_shrunk, len_no_blank = acoustic_hidden_shrink_tf(
                distribution_acoustic=logits,
                hidden=hidden_output,
                len_acoustic=len_logits,
                blank_id=self.args.dim_output-1,
                frame_expand=1)

            if self.is_train:
                loss = self.ctc_loss(
                    logits=logits,
                    len_logits=len_logits,
                    labels=tensors_input.label_splits[id_gpu],
                    len_labels=tensors_input.len_label_splits[id_gpu])

                if self.args.model.constrain_repeated:
                    from tfModels.CTCShrink import repeated_constrain_loss

                    loss_constrain = repeated_constrain_loss(
                        distribution_acoustic=logits,
                        hidden=hidden_output,
                        len_acoustic=len_hidden_output,
                        blank_id=self.args.dim_output-1)
                    loss += self.args.model.constrain_repeated * loss_constrain

                with tf.name_scope("gradients"):
                    gradients = self.optimizer.compute_gradients(loss)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        if self.is_train:
            return loss, gradients, [align, tensors_input.label_splits[id_gpu]]
        else:
            return logits, len_logits

    def ctc_loss(self, logits, len_logits, labels, len_labels):
        """
        No valid path found: It is possible that no valid path is found if the
        activations for the targets are zero.
        """
        with tf.name_scope("ctc_loss"):
            if self.args.model.use_wrapctc:
                import warpctc_tensorflow
                from tfTools.tfTools import get_indices

                indices = get_indices(len_labels)
                flat_labels = tf.gather_nd(labels, indices)
                ctc_loss_batch = warpctc_tensorflow.ctc(
                    activations=tf.transpose(logits, [1, 0, 2]),
                    flat_labels=flat_labels,
                    label_lengths=len_labels,
                    input_lengths=len_logits,
                    blank_label=self.args.dim_output)
            else:
                # with tf.get_default_graph()._kernel_label_map({"CTCLoss": "WarpCTC"}):
                labels_sparse = dense_sequence_to_sparse(
                    labels,
                    len_labels)
                ctc_loss_batch = tf.nn.ctc_loss(
                    labels_sparse,
                    logits,
                    sequence_length=len_logits,
                    ctc_merge_repeated=self.ctc_merge_repeated,
                    ignore_longer_outputs_than_inputs=True,
                    time_major=False)
            loss = tf.reduce_mean(ctc_loss_batch) # utter-level ctc loss

        if self.args.model.confidence_penalty:
            ls_loss = self.args.model.confidence_penalty * confidence_penalty(logits, len_logits)
            ls_loss = tf.reduce_mean(ls_loss)
            loss += ls_loss

        if self.args.model.policy_learning:
            from tfModels.regularization import policy_learning

            softmax_temperature = self.model.decoder.softmax_temperature
            dim_output = self.dim_output
            decoded_sparse = self.ctc_decode(logits, len_logits)
            rl_loss = policy_learning(logits, len_logits, decoded_sparse, labels, len_labels, softmax_temperature, dim_output, self.args)
            loss += self.args.model.policy_learning * rl_loss

        return loss


    def build_infer_graph(self):
        # cerate input tensors in the cpu
        tensors_input = self.build_infer_input()

        with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model)):
            logits, len_logits = self.build_single_graph(
                id_gpu=0,
                name_gpu=self.list_gpu_devices[0],
                tensors_input=tensors_input)

            decoded_sparse = self.ctc_decode(logits, len_logits)
            decoded = tf.sparse_to_dense(
                sparse_indices=decoded_sparse.indices,
                output_shape=decoded_sparse.dense_shape,
                sparse_values=decoded_sparse.values,
                default_value=0,
                validate_indices=True)
            distribution = tf.nn.softmax(logits)

        return decoded, tensors_input.shape_batch, distribution

    def ctc_decode(self, logits, len_logits):
        beam_size = self.args.beam_size
        logits_timeMajor = tf.transpose(logits, [1, 0, 2])

        if beam_size == 1:
            decoded_sparse = tf.to_int32(tf.nn.ctc_greedy_decoder(
                logits_timeMajor,
                len_logits,
                merge_repeated=True)[0][0])
        else:
            decoded_sparse = tf.to_int32(tf.nn.ctc_beam_search_decoder(
                logits_timeMajor,
                len_logits,
                beam_width=beam_size,
                merge_repeated=True)[0][0])

        return decoded_sparse
