import tensorflow as tf
import logging
import sys

from tfModels.tools import choose_device, smoothing_cross_entropy
from tfTools.tfTools import pad_to, dense_sequence_to_sparse
from tfSeq2SeqModels.seq2seqModel import Seq2SeqModel
from tfModels.regularization import confidence_penalty


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class CTCLMModel(Seq2SeqModel):
    '''
    decoder is the fc layer for ctc model;
    decoder2 is the extral model for langaige modelling
    '''
    def __init__(self, tensor_global_step, encoder, decoder, decoder2, is_train, args,
                 batch=None, embed_table_encoder=None, embed_table_decoder=None,
                 name='RNA_Model'):
        self.name = name
        self.gen_decoder2 = decoder2
        self.size_embedding = args.model.decoder2.size_embedding
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

            hidden_output, len_hidden_output = self.encoder(
                features=tensors_input.feature_splits[id_gpu],
                len_feas=tensors_input.len_fea_splits[id_gpu])

            acoustic, alignment, len_acoustic = self.fc_decoder(hidden_output, len_hidden_output)

            if not self.args.model.train_encoder:
                acoustic = tf.stop_gradient(acoustic)
                len_acoustic = tf.stop_gradient(len_acoustic)
            # used to guide the shrinking of the hidden_output
            distribution_acoustic = tf.nn.softmax(acoustic)

            blank_id = self.args.dim_output-1
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

            if (not self.is_train) and (self.args.beam_size>1):
                # infer phrase
                with tf.variable_scope(decoder.name or 'decoder'):
                    if self.args.dirs.lm_checkpoint:
                        logging.info('beam search with language model ...')
                        logits, decoded, len_decoded = decoder.beam_decode_rerank(
                            hidden_shrunk,
                            len_no_blank)
                    else:
                        logging.info('beam search ...')
                        logits, decoded, len_decoded = decoder.beam_decode(
                            hidden_shrunk,
                            len_no_blank)
            else:
                # train phrase
                print('greedy search ...')
                logits, decoded, len_decoded = decoder(hidden_shrunk, len_no_blank)

            if self.is_train:
                if self.args.model.decoder_loss == 'CE':
                    ocd_loss = self.ce_loss(
                        logits=logits,
                        labels=tensors_input.label_splits[id_gpu],
                        len_logits=len_acoustic,
                        len_labels=tensors_input.len_label_splits[id_gpu])
                elif self.args.model.decoder_loss == 'OCD':
                    ocd_loss = self.ocd_loss(
                        logits=logits,
                        len_logits=len_decoded,
                        labels=tensors_input.label_splits[id_gpu],
                        decoded=decoded,
                        len_decoded=len_decoded)
                elif self.args.model.decoder_loss == 'Premium_CE':

                    table_targets_distributions = tf.nn.softmax(tf.constant(self.args.table_targets))

                    ocd_loss = self.premium_ce_loss(
                        logits=logits,
                        labels=tensors_input.label_splits[id_gpu],
                        table_targets_distributions=table_targets_distributions,
                        len_logits=len_decoded,
                        len_labels=tensors_input.len_label_splits[id_gpu])
                elif self.args.model.decoder_loss == 'LM_CE':
                    ocd_loss = self.lm_ce_loss(
                        logits=logits,
                        len_logits=len_decoded,
                        labels=tensors_input.label_splits[id_gpu],
                        decoded=decoded,
                        len_decoded=len_decoded)
                else:
                    logging.info('not found loss type for decoder!')

                if self.args.model.train_encoder:
                    ctc_loss = self.ctc_loss(
                        logits=acoustic,
                        len_logits=len_acoustic,
                        labels=tensors_input.label_splits[id_gpu],
                        len_labels=tensors_input.len_label_splits[id_gpu])
                else:
                    ctc_loss = tf.constant(0.0)
                loss = self.schedule * ocd_loss + (1-self.schedule) * ctc_loss

                with tf.name_scope("gradients"):
                    assert loss.get_shape().ndims == 1
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

    def ctc_loss(self, logits, len_logits, labels, len_labels):
        """
        No valid path found: It is possible that no valid path is found if the
        activations for the targets are zero.
        return batch shape loss
        """
        with tf.name_scope("ctc_loss"):
            labels_sparse = dense_sequence_to_sparse(
                labels,
                len_labels)
            loss = tf.nn.ctc_loss(
                labels_sparse,
                logits,
                sequence_length=len_logits,
                ctc_merge_repeated=self.args.model.avg_repeated,
                ignore_longer_outputs_than_inputs=True,
                time_major=False)

        if self.args.model.decoder.confidence_penalty:
            ls_loss = self.args.model.decoder.confidence_penalty * \
                        confidence_penalty(logits, len_logits)
            loss += ls_loss

        return loss

    def premium_ce_loss(self, logits, labels, table_targets_distributions, len_logits, len_labels):
        """
        Compute optimization loss.
        batch major
        """
        l = tf.reduce_min([tf.shape(logits)[1], tf.shape(labels)[1]])
        logits = logits[:, :l, :]
        labels = labels[:, :l]

        target_distributions = tf.nn.embedding_lookup(table_targets_distributions, labels)

        with tf.name_scope('premium_ce_loss'):
            try:
                crossent = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=target_distributions,
                    logits=logits)
            except:
                crossent = tf.nn.softmax_cross_entropy_with_logits(
                    labels=target_distributions,
                    logits=logits)

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

            if self.args.model.token_level_ocd: # token-level
                loss /= tf.reduce_sum(mask)

        return loss

    def lm_ce_loss(self, logits, len_logits, labels, decoded, len_decoded):
        """
        Compute optimization loss.
        batch major
        """
        from tfModels.OptimalDistill import OCD

        ac_distributions, _ = OCD(
            hyp=decoded,
            ref=labels,
            vocab_size=self.args.dim_output)

        with tf.variable_scope(self.args.top_scope, reuse=True):
            with tf.variable_scope(self.args.lm_scope):
                pad_sos = tf.ones([tf.shape(decoded)[0], 1], dtype=tf.int32) * self.args.sos_idx
                decoded = tf.concat([pad_sos, decoded], 1)
                _, lm_distributions = self.args.lm_obj.decoder.score(decoded, len_decoded)
                lm_distributions = tf.stop_gradient(lm_distributions)

        # ac_lm_targets = 0.5*ac_distributions + 0.5*lm_distributions
        ac_lm_targets = 0.7*ac_distributions + 0.3*lm_distributions
        # ac_lm_targets = ac_distributions
        crossent = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=ac_lm_targets,
            logits=logits)

        pad_mask = tf.sequence_mask(
            len_logits,
            maxlen=tf.shape(logits)[1],
            dtype=logits.dtype)

        loss = tf.reduce_sum(crossent * pad_mask, -1) # utt-level

        if self.args.model.token_level_ocd: # token-level
            loss /= tf.reduce_sum(pad_mask, -1)

        return loss
