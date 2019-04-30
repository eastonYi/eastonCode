import tensorflow as tf
import logging
import sys

from tfModels.tools import choose_device, smoothing_cross_entropy
from tfTools.tfTools import dense_sequence_to_sparse
from tfSeq2SeqModels.seq2seqModel import Seq2SeqModel
from tfModels.regularization import confidence_penalty


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class CTCLMModel(Seq2SeqModel):
    '''
    '''
    def __init__(self, tensor_global_step, ctc_model, encoder, decoder, is_train,
                 args, batch=None, name='CTC-LM_Model'):
        self.name = name
        self.ctc_model = ctc_model
        self.size_embedding = args.model.decoder.size_embedding
        self.embedding_tabel = self.get_embedding(
            embed_table=None,
            size_input=args.dim_output,
            size_embedding=self.size_embedding)
        super().__init__(tensor_global_step, encoder, decoder, is_train, args,
                         batch, name=name)

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        tf.get_variable_scope().set_initializer(tf.variance_scaling_initializer(
            1.0, mode="fan_avg", distribution="uniform"))
        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            # build ctc model
            decoded_ctc, _, distribution_ctc = self.ctc_model.list_run

            from tfModels.CTCShrink import feature_shrink_tf
            blank_id = self.args.dim_output-1
            feature_shrunk, len_shrunk = feature_shrink_tf(
                distribution=distribution_ctc,
                feature=tensors_input.feature_splits[id_gpu],
                len_feature=tensors_input.len_fea_splits[id_gpu],
                blank_id=blank_id,
                frame_expand=self.args.model.frame_expand)

            feature_shrunk = tf.stop_gradient(feature_shrunk)
            len_shrunk = tf.stop_gradient(len_shrunk)

            # build sequence labellings model
            self.encoder = self.gen_encoder(
                is_train=self.is_train,
                args=self.args)
            self.decoder = self.gen_decoder(
                is_train=self.is_train,
                embed_table=self.embedding_tabel,
                global_step=self.global_step,
                args=self.args,
                name='decoder')
            self.schedule = self.decoder.schedule

            hidden_output, len_hidden_output = self.encoder(
                features=feature_shrunk,
                len_feas=len_shrunk)

            if (not self.is_train) and (self.args.beam_size>1):
                # infer phrase
                with tf.variable_scope(self.decoder.name or 'decoder'):
                    if self.args.dirs.lm_checkpoint:
                        logging.info('beam search with language model ...')
                        logits, decoded, len_decoded = self.decoder.beam_decode_rerank(
                            hidden_output,
                            len_hidden_output)
                    else:
                        logging.info('beam search ...')
                        logits, decoded, len_decoded = self.decoder.beam_decode(
                            hidden_output,
                            len_hidden_output)
            else:
                # train phrase
                print('greedy search ...')
                logits, decoded, len_decoded = self.decoder(hidden_output, len_hidden_output)

            if self.is_train:
                if self.args.model.decoder_loss == 'CE':
                    loss = self.ce_loss(
                        logits=logits,
                        labels=tensors_input.label_splits[id_gpu],
                        len_logits=len_decoded,
                        len_labels=tensors_input.len_label_splits[id_gpu])
                elif self.args.model.decoder_loss == 'OCD':
                    loss = self.ocd_loss(
                        logits=logits,
                        len_logits=len_decoded,
                        labels=tensors_input.label_splits[id_gpu],
                        decoded=decoded,
                        len_decoded=len_decoded)
                else:
                    logging.info('not found loss type for decoder!')

                with tf.name_scope("gradients"):
                    assert loss.get_shape().ndims == 1
                    loss = tf.reduce_mean(loss)
                    gradients = self.optimizer.compute_gradients(loss)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        if self.is_train:
            return loss, gradients, \
            [decoded, tensors_input.label_splits[id_gpu], loss]
            # return loss, gradients, tf.no_op()
        else:
            return logits, len_decoded, decoded

    def build_infer_graph(self):
        tensors_input = self.build_infer_input()

        with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model)):
            logits, len_logits, sample_id = self.build_single_graph(
                id_gpu=0,
                name_gpu=self.list_gpu_devices[0],
                tensors_input=tensors_input)

            distribution = logits

        return sample_id, tensors_input.shape_batch, distribution

    def ocd_loss(self, logits, len_logits, labels, decoded, len_decoded):
        """
        the logits length is the sample_id length
        return batch shape loss
        if `len_logits` is all zero. then outputs the 0
        """
        from tfModels.OptimalDistill import OCD

        optimal_distributions, optimal_targets = OCD(
            hyp=decoded,
            ref=labels,
            vocab_size=self.args.dim_output)

        crossent = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=optimal_distributions,
            logits=logits)

        pad_mask = tf.sequence_mask(
            len_logits,
            maxlen=tf.shape(logits)[1],
            dtype=logits.dtype)

        loss = tf.reduce_sum(crossent * pad_mask, -1) # utt-level

        if self.args.model.decoder2.confidence_penalty > 0: # utt-level
            cp_loss = self.args.model.decoder2.confidence_penalty * \
                        confidence_penalty(logits, len_decoded)
            loss += cp_loss

        if self.args.model.token_level_ocd: # token-level
            loss /= tf.reduce_sum(pad_mask, -1)

        return loss

    def ce_loss(self, logits, labels, len_logits, len_labels):
        """
        Compute optimization loss.
        batch major
        """
        l = tf.reduce_min([tf.shape(logits)[1], tf.shape(labels)[1]])
        with tf.name_scope('CE_loss'):
            crossent = smoothing_cross_entropy(
                logits=logits[:, :l, :],
                labels=labels[:, :l],
                vocab_size=self.args.dim_output,
                confidence=1.0)

            mask = tf.sequence_mask(
                len_labels,
                maxlen=l,
                dtype=logits.dtype)
            mask2 = tf.sequence_mask(
                len_logits,
                maxlen=l,
                dtype=logits.dtype)
            mask *= mask2
            # there must be reduce_sum not reduce_mean, for the valid token number is less
            loss = tf.reduce_sum(crossent * mask, -1)

            if self.args.model.decoder2.confidence_penalty > 0: # utt-level
                cp_loss = self.args.model.decoder2.confidence_penalty * \
                            confidence_penalty(logits, len_logits)
                loss += cp_loss

            if self.args.model.token_level_ocd: # token-level
                loss /= tf.reduce_sum(mask, -1)

        return loss
