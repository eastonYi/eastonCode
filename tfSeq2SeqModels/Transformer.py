'''@file model.py
contains de Model class
During the training , using greadysearch decoder, there is loss.
During the dev/infer, using beamsearch decoder, there is no logits, therefor loss, only predsself.
because we cannot access label during the dev set and need to depend on last time decision.

so, there is only infer and training
'''

import tensorflow as tf
import logging
from collections import namedtuple

from tfSeq2SeqModels.seq2seqModel import Seq2SeqModel
from tfModels.tools import choose_device, smoothing_cross_entropy


class Transformer(Seq2SeqModel):
    '''a general class for an encoder decoder system
    '''

    def __init__(self, tensor_global_step, encoder, decoder, is_train, args,
                 batch=None, embed_table_encoder=None, embed_table_decoder=None,
                 name='transformer'):
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
                         batch,
                         embed_table_encoder=None,
                         embed_table_decoder=None,
                         name=name)

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

            encoded, (len_encoded, _) = encoder(
                features=tensors_input.feature_splits[id_gpu],
                len_feas=tensors_input.len_fea_splits[id_gpu])

            decoder_input = decoder.build_input(
                id_gpu=id_gpu,
                tensors_input=tensors_input)

            with tf.variable_scope(self.name or 'decoder'):
                if self.is_train:
                    logits, preds, len_decoded = decoder.decode(
                        encoded,
                        len_encoded,
                        decoder_input.input_labels)
                else:
                    logits, preds, len_decoded = decoder.decoder_with_caching(
                        encoded,
                        len_encoded,
                        decoder_input.input_labels)
            # logits = tf.Print(logits, [tf.shape(logits)], message='logits:', summarize=1000)

            if self.is_train:
                if self.args.OCD_train:
                    loss, (optimal_targets, optimal_distributions) = self.ocd_loss(
                        logits=logits,
                        len_logits=len_decoded,
                        labels=tensors_input.label_splits[id_gpu],
                        preds=preds)
                else:
                    loss = self.ce_loss(
                        logits=logits,
                        # labels=decoder_input.output_labels[:, :tf.shape(logits)[1]],
                        labels=decoder_input.output_labels,
                        len_labels=decoder_input.len_labels)

                with tf.name_scope("gradients"):
                    gradients = self.optimizer.compute_gradients(loss)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        if self.is_train:
            # no_op is preserved for debug info to pass
            # return loss, gradients, tf.no_op()
            return loss, gradients, [len_decoded, preds, tensors_input.label_splits[id_gpu]]
        else:
            return logits, len_decoded, preds
