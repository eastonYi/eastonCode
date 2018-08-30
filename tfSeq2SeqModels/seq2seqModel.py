'''@file model.py
contains de Model class
During the training , using greadysearch decoder, there is loss.
During the dev/infer, using beamsearch decoder, there is no logits, therefor loss, only sample_idself.
because we cannot access label during the dev set and need to depend on last time decision.

so, there is only infer and training
'''

import tensorflow as tf
import logging
from collections import namedtuple

from tfModels.lstmModel import LSTM_Model
from tfModels.tools import choose_device, smoothing_cross_entropy

class Seq2SeqModel(LSTM_Model):
    '''a general class for an encoder decoder system
    '''

    def __init__(self, tensor_global_step, encoder, decoder, is_train, args,
                 batch=None, embed_table_encoder=None, embed_table_decoder=None,
                 name='seq2seqModel'):
        '''Model constructor

        Args:
        '''
        self.gen_encoder = encoder # encoder class
        self.gen_decoder = decoder # decoder class
        self.embed_table_encoder = embed_table_encoder
        self.embed_table_decoder = embed_table_decoder
        if embed_table_encoder:
            self.build_pl_input = self.build_idx_input
        self.helper_type = args.model.decoder.trainHelper if is_train \
            else args.model.decoder.inferHelper

        super().__init__(tensor_global_step, is_train, args, batch=batch, name=name)

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
            # if in the infer, the decoder_input.input_labels and len_labels are None
            decoder.build_helper(
                type=self.helper_type,
                labels=decoder_input.input_labels,
                len_labels=decoder_input.len_labels,
                batch_size=tf.size(len_encoded))

            logits, sample_id, _ = decoder(decoder_input)
            if self.is_train:
                loss = self.ce_loss(
                    logits=logits,
                    labels=decoder_input.output_labels[:, :tf.shape(logits)[1]],
                    len_labels=decoder_input.len_labels)

                with tf.name_scope("gradients"):
                    gradients = self.optimizer.compute_gradients(loss)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        return (loss, gradients) if self.is_train else sample_id

    def build_infer_graph(self):
        tensors_input = self.build_infer_input()

        with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model)):
            sample_id = self.build_single_graph(
                id_gpu=0,
                name_gpu=self.list_gpu_devices[0],
                tensors_input=tensors_input)

        if sample_id.get_shape().ndims == 3:
            sample_id = sample_id[:,:,0]
        return sample_id, tensors_input.shape_batch, tf.no_op()

    def ce_loss(self, logits, labels, len_labels):
        """
        Compute optimization loss.
        batch major
        """
        with tf.name_scope('CE_loss'):
            crossent = smoothing_cross_entropy(
                logits=logits,
                labels=labels,
                vocab_size=self.args.dim_output,
                confidence=self.args.label_smoothing_confidence)
            # crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            #     labels=labels,
            #     logits=logits)
            mask = tf.sequence_mask(
                len_labels,
                dtype=logits.dtype)
            # there nust be reduce_sum not reduce_mean, for the valid token number is less
            loss = tf.reduce_sum(crossent * mask)/tf.reduce_sum(mask)

        return loss

    def build_idx_input(self):
        """
        used for token-input tasks such as nmt when the `self.embed_table_encoder` is given
        for the token inputs are easy to fentch form disk, there is no need to
        use tfdata.
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, label_splits, len_fea_splits, len_label_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):

                batch_src = tf.placeholder(tf.int32, [None, None], name='input_src')
                batch_ref = tf.placeholder(tf.int32, [None, None], name='input_ref')
                batch_src_lens = tf.placeholder(tf.int32, [None], name='input_src_lens')
                batch_ref_lens = tf.placeholder(tf.int32, [None], name='input_ref_lens')
                self.list_pl = [batch_src, batch_ref, batch_src_lens, batch_ref_lens]
                # split input data alone batch axis to gpus
                batch_features = tf.nn.embedding_lookup(self.embed_table_encoder, batch_src)
                tensors_input.feature_splits = tf.split(batch_features, self.num_gpus, name="feature_splits")
                tensors_input.label_splits = tf.split(batch_ref, self.num_gpus, name="label_splits")
                tensors_input.len_fea_splits = tf.split(batch_src_lens, self.num_gpus, name="len_fea_splits")
                tensors_input.len_label_splits = tf.split(batch_ref_lens, self.num_gpus, name="len_label_splits")
        tensors_input.shape_batch = tf.shape(batch_features)

        return tensors_input
