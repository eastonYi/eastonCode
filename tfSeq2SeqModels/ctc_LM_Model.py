import tensorflow as tf
import logging
import sys

from tfModels.tools import choose_device, smoothing_cross_entropy
from tfTools.tfTools import pad_to, dense_sequence_to_sparse
from tfSeq2SeqModels.seq2seqModel import Seq2SeqModel
from tfModels.regularization import confidence_penalty


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class CTCLMModel(Seq2SeqModel):

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
                    # from tfModels.CTCShrink import acoustic_feature_shrink
                    # hidden_shrunk, len_no_blank = acoustic_feature_shrink(
                    #     distribution_acoustic,
                    #     x,
                    #     len_acoustic,
                    #     blank_id,
                    #     self.args.model.frame_expand)
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
                # from tfModels.CTCShrink import acoustic_hidden_shrink_v2
                # hidden_shrunk, len_no_blank = acoustic_hidden_shrink_v2(
                #     distribution_acoustic=distribution_acoustic,
                #     hidden=hidden_output,
                #     len_acoustic=len_acoustic,
                #     blank_id=blank_id,
                #     frame_expand=self.args.model.frame_expand)
                # hidden_shrunk = tf.stop_gradient(hidden_shrunk)
                # len_no_blank = tf.stop_gradient(len_no_blank)
            # else:
            #     from tfTools.tfTools import acoustic_hidden_shrink
            #     hidden_shrunk, len_no_blank = acoustic_hidden_shrink(
            #         distribution_acoustic=distribution_acoustic,
            #         hidden=hidden_output,
            #         len_acoustic=len_acoustic,
            #         blank_id=blank_id,
            #         hidden_size=self.args.model.encoder.num_cell_units,
            #         num_avg=self.args.model.num_avg)

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

                if self.args.model.teacher_forcing and self.is_train:
                    decoder_input = decoder.build_input(
                        id_gpu=id_gpu,
                        tensors_input=tensors_input)
                    logits_lm = decoder.teacher_forcing(
                        hidden_shrunk,
                        len_no_blank,
                        decoder_input,
                        tf.reduce_max(tensors_input.len_label_splits))
                    crossent_lm = smoothing_cross_entropy(
                        logits=logits_lm,
                        labels=tensors_input.label_splits[id_gpu],
                        vocab_size=self.args.dim_output,
                        confidence=0.9)
                    mask_lm = tf.sequence_mask(
                        tensors_input.len_label_splits[id_gpu],
                        maxlen=tf.shape(logits_lm)[1],
                        dtype=logits_lm.dtype)
                    lm_loss = tf.reduce_sum(crossent_lm * mask_lm)/tf.reduce_sum(mask_lm)
                    loss += lm_loss

                if self.args.musk_update:
                    self.idx_update = self.deserve_idx(
                        decoded,
                        len_decoded,
                        tensors_input.label_splits[id_gpu],
                        tensors_input.len_label_splits[id_gpu])
                    loss = tf.reshape(tf.gather(loss, self.idx_update), [-1])

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
        from tfModels.OCDLoss import OCD_loss

        optimal_distributions, optimal_targets = OCD_loss(
            hyp=decoded,
            ref=labels,
            vocab_size=self.args.dim_output)
        try:
            crossent = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=optimal_distributions,
                logits=logits)
        except:
            crossent = tf.nn.softmax_cross_entropy_with_logits(
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
                loss /= tf.reduce_sum(mask)

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

    def deserve_idx(self, decoded, len_decoded, labels, len_labels):
        """
        if one sent is correct during training, then not to train on it
        """
        decoded_sparse = dense_sequence_to_sparse(
            seq=decoded,
            len_seq=len_decoded)
        label_sparse = dense_sequence_to_sparse(
            seq=labels,
            len_seq=len_labels)

        distance = tf.edit_distance(decoded_sparse, label_sparse, normalize=False)
        indices = tf.where(distance>1)

        return indices

    def get_embedding(self, embed_table, size_input, size_embedding):
        if size_embedding and (type(embed_table) is not tf.Variable):
            with tf.device("/cpu:0"):
                with tf.variable_scope(self.name, reuse=(self.__class__.num_Model > 0)):
                    embed_table = tf.get_variable(
                        "embedding", [size_input, size_embedding], dtype=tf.float32)

        return embed_table
