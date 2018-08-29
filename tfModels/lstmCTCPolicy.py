import tensorflow as tf
import logging
import sys

from tensorflow.contrib.layers import fully_connected

from tfModels.tools import choose_device
from tfModels.lstmCTCModel import LSTM_CTC_Model
from tfTools.tfTools import dense_sequence_to_sparse
from tfTools.gradientTools import average_gradients, handle_gradients

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class LSTM_CTC_PolicyModel(LSTM_CTC_Model):

    def build_graph(self):
        # cerate input tensors in the cpu
        tensors_input = self.build_input()
        # create optimizer
        self.optimizer = self.build_optimizer()

        loss_step = []
        tower_grads = []
        list_sample_id = []
        list_wer = []
        list_loss = []

        for id_gpu, name_gpu in enumerate(self.list_gpu_devices):
            with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model)):
                loss, gradients, sample_id, wer, batch_loss = \
                    self.build_single_graph(id_gpu, name_gpu, tensors_input)
                loss_step.append(loss)
                tower_grads.append(gradients)
                list_sample_id.append(sample_id)
                list_wer.append(wer)
                list_loss.append(batch_loss)

        loss = tf.reduce_mean(loss_step)
        with tf.device(self.center_device):
            averaged_grads = average_gradients(tower_grads)
            handled_grads = handle_gradients(averaged_grads, self.args)
            op_optimize = self.optimizer.apply_gradients(handled_grads, self.global_step)

        self.__class__.num_Instances += 1
        logging.info("built {} {} instance(s).".format(self.__class__.num_Instances, self.__class__.__name__))

        return loss, tensors_input.shape_batch, list_sample_id, tensors_input.label_splits, list_wer, list_loss, op_optimize

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        num_class = self.args.dim_output + 1
        Encoder = self.args.model.encoder.type

        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            # create encoder obj
            encoder = Encoder(
                is_train=self.is_train,
                args=self.args)
            encoder_input = encoder.build_input(
                id_gpu=id_gpu,
                tensors_input=tensors_input)

            # using encoder to encode the inout sequence
            hidden_output, len_logits = encoder(encoder_input)
            logits = fully_connected(
                inputs=hidden_output,
                num_outputs=num_class)
            decoded_sparse = self.ctc_decode(logits, len_logits)

            if self.is_train:
                label_sparse = dense_sequence_to_sparse(
                    sequences=tensors_input.output_labels,
                    sequence_lengths=tensors_input.len_labels)

                wer_bias = tf.edit_distance(decoded_sparse, label_sparse, normalize=True)
                temped_logits = logits
                sampled_aligns = tf.distributions.Categorical(logits=temped_logits).sample()

                max_wer = tf.convert_to_tensor(self.args.model.max_wer)
                min_reward = tf.convert_to_tensor(self.args.model.min_reward)
                reward = wer_bias - wer

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
            return loss, gradients, [selected, selected], [wer_bias, wer], batch_loss
        else:
            return logits, len_logits

    def policy_ctc_loss(self, logits, len_logits, labels, len_labels):
        with tf.name_scope("ctc_loss"):
            labels_sparse = dense_sequence_to_sparse(
                labels,
                len_labels)
            ctc_loss_batch = tf.nn.ctc_loss(
                labels_sparse,
                logits,
                sequence_length=len_logits,
                ignore_longer_outputs_than_inputs=True,
                time_major=False)
            loss = tf.reduce_mean(ctc_loss_batch) # utter-level ctc loss

        return loss
