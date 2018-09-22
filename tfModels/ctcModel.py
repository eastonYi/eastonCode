import tensorflow as tf
import logging
import sys

from tensorflow.contrib.layers import fully_connected

from tfModels.tools import choose_device
from tfModels.lstmModel import LSTM_Model
from tfTools.tfTools import dense_sequence_to_sparse

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class CTCModel(LSTM_Model):

    def __init__(self, tensor_global_step, is_train, args, batch=None, name='tf_CTC_Model',
                 encoder=None, decoder=None, embed_table_encoder=None, embed_table_decoder=None):
        self.sample_prob = tf.convert_to_tensor(0.0)
        self.merge_repeated = args.model.ctc_merge_repeated
        super().__init__(tensor_global_step, is_train, args, batch, name)

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        num_class = self.args.dim_output + 1
        Encoder = self.args.model.encoder.type
        tf.get_variable_scope().set_initializer(tf.variance_scaling_initializer(
            1.0, mode="fan_avg", distribution="uniform"))
        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            # create encoder obj
            encoder = Encoder(
                is_train=self.is_train,
                args=self.args)
            # using encoder to encode the inout sequence
            hidden_output, len_logits = encoder(
                features=tensors_input.feature_splits[id_gpu],
                len_feas=tensors_input.len_fea_splits[id_gpu])
            logits = fully_connected(
                inputs=hidden_output,
                num_outputs=num_class)

            if self.is_train:
                loss = self.ctc_loss(
                    logits=logits,
                    len_logits=len_logits,
                    labels=tensors_input.label_splits[id_gpu],
                    len_labels=tensors_input.len_label_splits[id_gpu])

                with tf.name_scope("gradients"):
                    gradients = self.optimizer.compute_gradients(loss)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        if self.is_train:
            return loss, gradients
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
                    ctc_merge_repeated=self.merge_repeated,
                    ignore_longer_outputs_than_inputs=True,
                    time_major=False)
            loss = tf.reduce_mean(ctc_loss_batch) # utter-level ctc loss

        if self.args.model.confidence_penalty:
            print('using confidence penalty')
            with tf.name_scope("confidence_penalty"):
                real_probs = tf.nn.softmax(logits)
                prevent_nan_constant = tf.constant(1e-10)
                real_probs += prevent_nan_constant

                neg_entropy = tf.reduce_sum(real_probs * tf.log(real_probs), axis=-1)
                ls_loss = self.args.model.confidence_penalty * tf.reduce_sum(neg_entropy, axis=-1)
            loss += ls_loss

        if self.args.model.policy_learning:
            from tfModels.CTCLoss import ctc_sample, ctc_reduce_map
            from tfTools.tfTools import sparse_shrink, pad_to_same
            num_class = self.args.dim_output + 1
            print('using policy learning')
            with tf.name_scope("policy_learning"):
                label_sparse = dense_sequence_to_sparse(labels, len_labels)
                decoded_sparse = self.ctc_decode(logits, len_logits)
                wer_bias = tf.edit_distance(decoded_sparse, label_sparse, normalize=True)
                wer_bias = tf.stop_gradient(wer_bias)

                sampled_align = ctc_sample(logits, self.args.model.softmax_temperature)
                sample_sparse = ctc_reduce_map(sampled_align, id_blank=num_class-1)
                wer = tf.edit_distance(sample_sparse, label_sparse, normalize=True)
                seq_sample, len_sample, _ = sparse_shrink(sample_sparse)

                # ==0 is not success!!
                seq_sample, labels = pad_to_same([seq_sample, labels])
                seq_sample = tf.where(len_sample<1, labels, seq_sample)
                len_sample = tf.where(len_sample<1, len_labels, len_sample)

                reward = wer_bias - wer

                rl_loss, _ = self.policy_ctc_loss(
                    logits=logits,
                    len_logits=len_logits,
                    flabels=seq_sample,
                    len_flabels=len_sample,
                    batch_reward=reward,
                    args=self.args)
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
                merge_repeated=self.merge_repeated)[0][0])
        else:
            decoded_sparse = tf.to_int32(tf.nn.ctc_beam_search_decoder(
                logits_timeMajor,
                len_logits,
                beam_width=beam_size,
                merge_repeated=self.merge_repeated)[0][0])

        return decoded_sparse

    @staticmethod
    def policy_ctc_loss(logits, len_logits, flabels, len_flabels, batch_reward, args):
        """
        flabels: not the ground-truth
        if len_flabels=None, means the `flabels` is sparse
        """
        from tfTools.math_tf import non_linear

        with tf.name_scope("policy_ctc_loss"):
            if len_flabels is not None:
                flabels_sparse = dense_sequence_to_sparse(
                    flabels,
                    len_flabels)
            else:
                flabels_sparse = flabels

            ctc_loss_batch = tf.nn.ctc_loss(
                flabels_sparse,
                logits,
                sequence_length=len_logits,
                ignore_longer_outputs_than_inputs=True,
                time_major=False)
            ctc_loss_batch *= batch_reward
            ctc_loss_batch = non_linear(
                ctc_loss_batch,
                args.model.non_linear,
                args.model.min_reward)
            loss = tf.reduce_mean(ctc_loss_batch) # utter-level ctc loss

        return loss, ctc_loss_batch
