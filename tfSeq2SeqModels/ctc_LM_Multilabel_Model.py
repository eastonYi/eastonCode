import tensorflow as tf
import logging
import sys
from collections import namedtuple

from tfModels.tools import choose_device, smoothing_cross_entropy
from tfTools.tfTools import pad_to, dense_sequence_to_sparse
from tfSeq2SeqModels.ctc_LM_Model import CTCLMModel
from tfModels.regularization import confidence_penalty


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class CTCLMMultilabelModel(CTCLMModel):

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        tf.get_variable_scope().set_initializer(tf.variance_scaling_initializer(
            1.0, mode="fan_avg", distribution="uniform"))
        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            self.encoder = self.gen_encoder(
                is_train=self.is_train,
                args=self.args)
            self.fc_decoder = self.gen_decoder(
                is_train=self.is_train,
                embed_table=None,
                global_step=self.global_step,
                args=self.args,
                name='decoder')
            self.decoder = decoder = self.gen_decoder2(
                is_train=self.is_train,
                embed_table=self.embedding_tabel,
                global_step=self.global_step,
                args=self.args,
                name='decoder2')
            self.schedule = decoder.schedule

            hidden_output, hidden_bottleneck, len_hidden_output = self.encoder(
                features=tensors_input.feature_splits[id_gpu],
                len_feas=tensors_input.len_fea_splits[id_gpu])
            if self.args.model.true_end2end:
                acoustic, alignment, len_acoustic = self.fc_decoder(hidden_output, len_hidden_output)
            else:
                acoustic, alignment, len_acoustic = self.fc_decoder(hidden_bottleneck, len_hidden_output)

            if not self.args.model.train_encoder:
                acoustic = tf.stop_gradient(acoustic)
                len_acoustic = tf.stop_gradient(len_acoustic)
            # used to guide the shrinking of the hidden_output
            distribution_acoustic = tf.nn.softmax(acoustic)

            # whether to shrink the hidden or the acoutic distribution
            if self.args.model.shrink_hidden == 'distribution':
                hidden_output = distribution_acoustic
            elif self.args.model.shrink_hidden == 'self-define':
                hidden_output = hidden_bottleneck

            blank_id = self.args.dim_ctc_output-1 if self.args.dim_ctc_output else self.args.dim_output-1
            if self.args.model.avg_repeated:
                if self.args.model.true_end2end:
                    from tfModels.CTCShrink import acoustic_hidden_shrink_v3
                    hidden_shrunk, len_no_blank = acoustic_hidden_shrink_v3(
                        distribution_acoustic,
                        hidden_output,
                        len_acoustic,
                        blank_id,
                        self.args.model.frame_expand)
                else:
                    from tfModels.CTCShrink import acoustic_hidden_shrink_tf
                    hidden_shrunk, len_no_blank = acoustic_hidden_shrink_tf(
                        distribution_acoustic=distribution_acoustic,
                        hidden=hidden_output,
                        len_acoustic=len_acoustic,
                        blank_id=blank_id,
                        frame_expand=self.args.model.frame_expand)

                if self.is_train and self.args.model.dropout > 0.0:
                    hidden_shrunk = tf.nn.dropout(hidden_shrunk, keep_prob=1-self.args.model.dropout)

            if (not self.is_train) and (self.args.beam_size>1):
                # infer phrase
                if self.args.dirs.lm_checkpoint:
                    logging.info('beam search with language model ...')
                    with tf.variable_scope(decoder.name or 'decoder'):
                        if self.args.model.rerank:
                            logits, decoded, len_decoded = decoder.beam_decode_rerank(
                                hidden_shrunk,
                                len_no_blank)
                        else:
                            logits, decoded, len_decoded = decoder.beam_decode_lm(
                                hidden_shrunk,
                                len_no_blank)
                else:
                    logging.info('beam search ...')
                    with tf.variable_scope(decoder.name or 'decoder'):
                        logits, decoded, len_decoded = decoder.beam_decode(
                            hidden_shrunk,
                            len_no_blank)
            else:
                # train phrase
                print('greedy search ...')
                logits, decoded, len_decoded = decoder(hidden_shrunk, len_no_blank)

            if self.is_train:
                if self.args.model.use_ce_loss:
                    ocd_loss = self.ce_loss(
                        logits=logits,
                        labels=tensors_input.label_splits[id_gpu],
                        len_logits=len_acoustic,
                        len_labels=tensors_input.len_label_splits[id_gpu])
                else:
                    ocd_loss = self.ocd_loss(
                        logits=logits,
                        len_logits=len_decoded,
                        labels=tensors_input.label_splits[id_gpu],
                        decoded=decoded,
                        len_decoded=len_decoded)

                if self.args.model.train_encoder:
                    ctc_loss = self.ctc_loss(
                        logits=acoustic,
                        len_logits=len_acoustic,
                        labels=tensors_input.phone_splits[id_gpu],
                        len_labels=tensors_input.len_phone_splits[id_gpu])
                else:
                    ctc_loss = tf.constant(0.0)
                loss = self.schedule * ocd_loss + (1-self.schedule) * ctc_loss

                with tf.name_scope("gradients"):
                    loss = tf.reduce_mean(loss)
                    gradients = self.optimizer.compute_gradients(loss)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        if self.is_train:
            return loss, gradients, \
            [decoded, tensors_input.label_splits[id_gpu], distribution_acoustic, len_acoustic, len_no_blank, hidden_shrunk, ctc_loss, ocd_loss]
            # return loss, gradients, tf.no_op()
        else:

            return logits, len_decoded, decoded, (alignment, len_acoustic)

    def build_infer_graph(self):
        tensors_input = self.build_infer_input()

        with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model)):
            logits, len_logits, sample_id, (alignment, len_acoustic) = self.build_single_graph(
                id_gpu=0,
                name_gpu=self.list_gpu_devices[0],
                tensors_input=tensors_input)

        return sample_id, tensors_input.shape_batch, (alignment, len_acoustic)

    def build_tf_input(self):
        """
        stand training input
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, label_splits, phone_splits, len_fea_splits, len_label_splits, len_phone_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                # split input data alone batch axis to gpus
                tensors_input.feature_splits = tf.split(self.batch[0], self.num_gpus, name="feature_splits")
                tensors_input.label_splits = tf.split(self.batch[1], self.num_gpus, name="label_splits")
                tensors_input.phone_splits = tf.split(self.batch[2], self.num_gpus, name="phone_splits")
                tensors_input.len_fea_splits = tf.split(self.batch[3], self.num_gpus, name="len_fea_splits")
                tensors_input.len_label_splits = tf.split(self.batch[4], self.num_gpus, name="len_label_splits")
                tensors_input.len_phone_splits = tf.split(self.batch[5], self.num_gpus, name="len_phone_splits")
        tensors_input.shape_batch = tf.shape(self.batch[0])

        return tensors_input
