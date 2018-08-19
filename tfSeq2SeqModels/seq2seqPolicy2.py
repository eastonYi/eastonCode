'''@file model.py
contains de Model class'''

import tensorflow as tf
import logging
from collections import namedtuple

from tfSeq2SeqModels.seq2seqModel import Seq2SeqModel
from tfModels.tools import choose_device, smoothing_cross_entropy
from tfTools.gradientTools import average_gradients, handle_gradients
from tfTools.tfTools import dense_sequence_to_sparse


class Seq2SeqPolicyModel(Seq2SeqModel):
    '''a general class for an encoder decoder system'''

    # if you need to resore the model form seq2seqModel, the name must be 'seq2seqModel'
    def __init__(self, tensor_global_step, encoder, decoder, is_train, args,
                 batch=None, embed_table_encoder=None, embed_table_decoder=None,
                 name='seq2seqModel'):
        super().__init__(tensor_global_step, encoder, decoder, is_train, args,
                     batch, embed_table_encoder, embed_table_decoder, name)

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
                loss, gradients, sample_id, wer, batch_loss = self.build_single_graph(id_gpu, name_gpu, tensors_input)
                loss_step.append(loss)
                tower_grads.append(gradients)
                list_sample_id.append(sample_id)
                list_wer.append(wer)
                list_loss.append(batch_loss)

        # mean the loss
        loss = tf.reduce_mean(loss_step)
        # merge gradients, update current model
        with tf.device(self.center_device):
            # computation relevant to gradient
            averaged_grads = average_gradients(tower_grads)
            handled_grads = handle_gradients(averaged_grads, self.args)
            # with tf.variable_scope('adam', reuse=False):
            op_optimize = self.optimizer.apply_gradients(handled_grads, self.global_step)

        self.__class__.num_Instances += 1
        logging.info("built {} {} instance(s).".format(self.__class__.num_Instances, self.__class__.__name__))

        return loss, tensors_input.shape_batch, list_sample_id, tensors_input.label_splits, list_wer, list_loss, op_optimize

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):

        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            encoder = self.gen_encoder(
                is_train=self.is_train,
                embed_table=self.embed_table_encoder,
                args=self.args)
            decoder = self.gen_decoder(
                is_train=self.is_train,
                embed_table=self.embed_table_decoder,
                global_step=self.global_step,
                args=self.args)
            self.sample_prob = decoder.sample_prob

            encoder_input = encoder.build_input(
                id_gpu=id_gpu,
                tensors_input=tensors_input)
            encoded, len_encoded = encoder(encoder_input)

            decoder_input = decoder.build_input(
                id_gpu=id_gpu,
                encoded=encoded,
                len_encoded=len_encoded,
                tensors_input=tensors_input)
            logits, sample_id, len_decode = decoder(decoder_input)

            if self.is_train:
                # bias decoder
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    decoder_bias = self.gen_decoder(
                        is_train=False,
                        embed_table=self.embed_table_decoder,
                        global_step=self.global_step,
                        args=self.args)
                    _, sample_id_bias, len_decode_bias = decoder_bias(decoder_input)

                # sample_id: selected, argmax_ids, sample_ids
                argmax = sample_id_bias[:,:,0]
                argmax_sparse = dense_sequence_to_sparse(
                    sequences=argmax,
                    sequence_lengths=len_decode_bias[:,0])
                selected = sample_id[0]
                selected_sparse = dense_sequence_to_sparse(
                    sequences=selected,
                    sequence_lengths=len_decode)
                label_sparse = dense_sequence_to_sparse(
                    sequences=decoder_input.output_labels,
                    sequence_lengths=decoder_input.len_labels)

                wer = tf.edit_distance(selected_sparse, label_sparse, normalize=True)
                # wer_bias = tf.edit_distance(selected_sparse, label_sparse, normalize=True)
                wer_bias = tf.edit_distance(argmax_sparse, label_sparse, normalize=True)
                wer_bias = tf.stop_gradient(wer_bias)
                reward = wer_bias - wer
                max_wer = tf.convert_to_tensor(self.args.model.max_wer)
                min_reward = tf.convert_to_tensor(self.args.model.min_reward)
                reward = tf.where(wer<max_wer, reward, tf.zeros_like(reward))
                reward = tf.where(reward>min_reward, reward, tf.zeros_like(reward))

                loss, batch_loss = self.policy_ce_loss(
                    logits=logits,
                    labels=sample_id[0],
                    len_labels=len_decode,
                    batch_reward=reward)
                # loss = tf.Print(loss, [loss, batch_loss], message='loss: ', summarize=1000)

                with tf.name_scope("gradients"):
                    gradients = self.optimizer.compute_gradients(loss)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        # return (loss, gradients, sample_id, [wer_bias, wer], sample_id_bias) if self.is_train else sample_id
        return (loss, gradients, [argmax, selected], [wer_bias, wer], batch_loss) if self.is_train else sample_id

    def policy_ce_loss(self, logits, labels, len_labels, batch_reward):
        """
        Compute optimization loss.
        batch major
        """
        with tf.name_scope('CE_loss'):
            crossent = smoothing_cross_entropy(
                logits=logits,
                labels=labels,
                vocab_size=self.args.dim_output,
                confidence=self.args.label_smoothing_confidence
            )
            mask = tf.sequence_mask(
                len_labels,
                dtype=logits.dtype)
            batch_loss = batch_reward * tf.reduce_sum(crossent * mask, -1)
            loss = tf.reduce_sum(batch_loss)/tf.reduce_sum(mask)

        return loss, batch_loss
