'''@file model.py
contains de Model class
During the training , using greadysearch decoder, there is loss.
During the dev/infer, using beamsearch decoder, there is no logits, therefor loss, only predsself.
because we cannot access label during the dev set and need to depend on last time decision.

so, there is only infer and training
'''

import tensorflow as tf
import logging

from tfSeq2SeqModels.seq2seqModel import Seq2SeqModel
from tfModels.tools import choose_device


class Transformer(Seq2SeqModel):
    '''a general class for an encoder decoder system
    '''

    def __init__(self, tensor_global_step, encoder, decoder, is_train, args,
                 batch=None, name='transformer'):
        '''Model constructor

        Args:
        '''
        self.name = name
        self.size_embedding = args.model.decoder.size_embedding
        self.embedding_tabel = self.get_embedding(
            embed_table=None,
            size_input=args.dim_output,
            size_embedding=self.size_embedding)
        super().__init__(tensor_global_step, encoder, decoder, is_train, args,
                         batch, name=name)

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):

        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            encoder = self.gen_encoder(
                is_train=self.is_train,
                embed_table=None,
                args=self.args)
            decoder = self.gen_decoder(
                is_train=self.is_train,
                embed_table=self.embedding_tabel,
                global_step=self.global_step,
                args=self.args)

            with tf.variable_scope(encoder.name or 'encoder'):
                encoded, len_encoded = encoder(
                    features=tensors_input.feature_splits[id_gpu],
                    len_feas=tensors_input.len_fea_splits[id_gpu])

            with tf.variable_scope(decoder.name or 'decoder'):
                decoder_input = decoder.build_input(
                    id_gpu=id_gpu,
                    tensors_input=tensors_input)

                if (not self.is_train) or (self.args.model.training_type == 'self-learning'):
                    '''
                    training_type:
                        - self-learning: get logits fully depend on self
                        - teacher-forcing: get logits depend on labels during training
                    '''
                    # infer phrases
                    if self.args.beam_size>1:
                        logging.info('beam search with language model ...')
                        results, preds, len_decoded = decoder.beam_decode_rerank(
                            encoded,
                            len_encoded)
                    else:
                        logging.info('gready search ...')
                        results, preds, len_decoded = decoder.decoder_with_caching(
                            encoded,
                            len_encoded)
                else:
                    logging.info('teacher-forcing training ...')
                    decoder_input_labels = decoder_input.input_labels * tf.sequence_mask(
                        decoder_input.len_labels,
                        maxlen=tf.shape(decoder_input.input_labels)[1],
                        dtype=tf.int32)
                    logits, preds = decoder.decode(
                        encoded=encoded,
                        len_encoded=len_encoded,
                        decoder_input=decoder_input_labels)

            if self.is_train:
                if self.args.model.loss_type == 'OCD':
                    """
                    constrain the max decode length for ocd training since model
                    will decode to that long at beginning. Recommend 30.
                    """
                    logits = results
                    loss, _ = self.ocd_loss(
                        logits=logits,
                        len_logits=len_decoded,
                        labels=tensors_input.label_splits[id_gpu],
                        preds=preds)
                    # loss = self.ce_loss(
                    #     logits=logits,
                    #     labels=preds,
                    #     len_labels=len_decoded)
                elif self.args.model.loss_type == 'beam_OCD':
                    logits, preds, len_decoded, _, _ = results
                    batch= tf.shape(logits)[0]
                    beam_size = self.args.beam_size
                    batch_x_beam = batch*beam_size
                    logits = tf.reshape(logits, [batch_x_beam, -1, self.args.dim_output])
                    len_decoded = tf.reshape(len_decoded, [-1])
                    preds = tf.reshape(preds, [batch_x_beam, -1])
                    labels = tf.reshape(tf.tile(tensors_input.label_splits[id_gpu][:, None, :], [1, beam_size, 1]),
                                        [batch_x_beam, -1])
                    # logits = tf.Print(logits, [batch_x_beam, tf.shape(logits), tf.shape(preds), tf.shape(labels), tf.shape(len_decoded)], message='batch_x_beam, logits, preds, labels, len_decoded: ', summarize=1000)
                    loss, _ = self.ocd_loss(
                        logits=logits,
                        len_logits=len_decoded,
                        labels=labels,
                        preds=preds)
                elif self.args.model.loss_type == 'CE':
                    loss = self.ce_loss(
                        logits=logits,
                        labels=decoder_input.output_labels,
                        len_labels=decoder_input.len_labels)

                elif self.args.model.loss_type == 'Premium_CE':
                    table_targets_distributions = tf.nn.softmax(tf.constant(self.args.table_targets))
                    loss = self.premium_ce_loss(
                        logits=logits,
                        labels=tensors_input.label_splits[id_gpu],
                        table_targets_distributions=table_targets_distributions,
                        len_labels=tensors_input.len_label_splits[id_gpu])
                else:
                    raise NotImplementedError('NOT found loss type: {}'.format(self.args.model.loss_type))

                with tf.name_scope("gradients"):
                    assert loss.get_shape().ndims == 1
                    loss = tf.reduce_mean(loss)
                    gradients = self.optimizer.compute_gradients(loss)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        if self.is_train:
            # no_op is preserved for debug info to pass
            return loss, gradients, [preds, tensors_input.label_splits[id_gpu]]
        else:
            return results, len_decoded, preds
