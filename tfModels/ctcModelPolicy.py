import tensorflow as tf
import logging
import sys

from tensorflow.contrib.layers import fully_connected

from tfModels.tools import choose_device
from tfModels.ctcModel import CTCModel
from tfModels.CTCLoss import ctc_sample, ctc_reduce_map
from tfTools.tfTools import dense_sequence_to_sparse, sparse_shrink
from tfTools.gradientTools import average_gradients, handle_gradients

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class CTC_PolicyModel(CTCModel):

    def build_graph(self):
        # cerate input tensors in the cpu
        tensors_input = self.build_input()
        # create optimizer
        self.optimizer = self.build_optimizer()

        loss_step = []
        tower_grads = []
        list_argmax = []
        list_sample = []
        list_wer_bias = []
        list_wer = []
        list_loss = []

        for id_gpu, name_gpu in enumerate(self.list_gpu_devices):
            with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model)):
                loss, gradients, sample_id, wer, batch_loss = \
                    self.build_single_graph(id_gpu, name_gpu, tensors_input)
                loss_step.append(loss)
                tower_grads.append(gradients)
                list_argmax.append(sample_id[0])
                list_sample.append(sample_id[1])
                list_wer_bias.append(wer[0])
                list_wer.append(wer[1])
                list_loss.append(batch_loss)

        loss = tf.reduce_mean(loss_step)
        with tf.device(self.center_device):
            averaged_grads = average_gradients(tower_grads)
            handled_grads = handle_gradients(averaged_grads, self.args)
            op_optimize = self.optimizer.apply_gradients(handled_grads, self.global_step)

        batch_argmax = tf.concat(list_argmax, axis=0)
        batch_sample =  tf.concat(list_sample, axis=0)
        batch_label = tf.concat(tensors_input.label_splits, axis=0)
        batch_wer_bias = tf.concat(list_wer_bias, axis=0)
        batch_wer = tf.concat(list_wer, axis=0)
        batch_loss = tf.concat(list_loss, axis=0)

        self.__class__.num_Instances += 1
        logging.info("built {} {} instance(s).".format(self.__class__.num_Instances, self.__class__.__name__))

        return loss, tensors_input.shape_batch, [batch_argmax, batch_sample, batch_label],\
            [batch_wer, batch_wer_bias], batch_loss, op_optimize

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        num_class = self.args.dim_output + 1
        Encoder = self.args.model.encoder.type

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
                label_sparse = dense_sequence_to_sparse(
                    seq=tensors_input.label_splits[id_gpu],
                    len_seq=tensors_input.len_label_splits[id_gpu])

                decoded_sparse = self.ctc_decode(logits, len_logits)
                decoded_id = tf.sparse_tensor_to_dense(decoded_sparse, default_value=0)
                wer_bias = tf.edit_distance(decoded_sparse, label_sparse, normalize=True)
                wer_bias = tf.stop_gradient(wer_bias)

                sampled = ctc_sample(logits, self.args.model.softmax_temperature)
                sample_sparse = ctc_reduce_map(sampled, id_blank=num_class-1)
                wer = tf.edit_distance(sample_sparse, label_sparse, normalize=True)
                seq_sample, len_sample, _ = sparse_shrink(sample_sparse)
                # logits = tf.Print(logits, [tf.shape(logits), tf.shape(seq_sample)], message='logits, seq_sample: ', summarize=1000)

                max_wer = tf.convert_to_tensor(self.args.model.max_wer)
                min_reward = tf.convert_to_tensor(self.args.model.min_reward)
                reward = wer_bias - wer
                max_wer = tf.convert_to_tensor(self.args.model.max_wer)
                min_reward = tf.convert_to_tensor(self.args.model.min_reward)
                reward = tf.where(wer < max_wer, reward, tf.zeros_like(reward))
                reward = tf.where(reward > min_reward, reward, tf.zeros_like(reward))

                loss, batch_loss = self.policy_ctc_loss(
                    logits=logits,
                    len_logits=len_logits,
                    labels=seq_sample,
                    len_labels=len_sample,
                    batch_reward=reward)

                with tf.name_scope("gradients"):
                    gradients = self.optimizer.compute_gradients(loss)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        if self.is_train:
            return loss, gradients, [decoded_id, seq_sample], [wer_bias, wer], batch_loss
        else:
            return logits, len_logits

    def policy_ctc_loss(self, logits, len_logits, labels, len_labels, batch_reward):
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
            ctc_loss_batch *= batch_reward
            loss = tf.reduce_mean(ctc_loss_batch) # utter-level ctc loss

        return loss, ctc_loss_batch
