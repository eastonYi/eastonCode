import tensorflow as tf
import logging
import sys

from tfModels.tools import choose_device, smoothing_cross_entropy
from tfTools.tfTools import pad_to, dense_sequence_to_sparse
from tfSeq2SeqModels.seq2seqModel import Seq2SeqModel
from tfModels.regularization import confidence_penalty
from tfModels.CTCShrink import acoustic_hidden_shrink_tf

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class AC_LM_Classifier(Seq2SeqModel):
    '''
    this model used to fusion acoutic model and lamguage model
    the acoustic model need to be respect to acoutic and remove context dependent info
    so conv net and is the ideal model. 
    '''

    def __init__(self, tensor_global_step, encoder, decoder, decoder2, is_train, args,
                 batch=None, embed_table_encoder=None, embed_table_decoder=None,
                 name='RNA_Model'):
        self.name = name
        self.gen_decoder2 = decoder2
        self.size_embedding = args.model.classifier.size_embedding
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

            hidden_output, len_hidden_output = self.encoder(
                features=tensors_input.feature_splits[id_gpu],
                len_feas=tensors_input.len_fea_splits[id_gpu])
            logits_acoustic, alignment, len_acoustic = self.fc_decoder(hidden_output, len_hidden_output)
            logits_acoustic = tf.stop_gradient(logits_acoustic)
            len_acoustic = tf.stop_gradient(len_acoustic)

            distribution_acoustic = tf.nn.softmax(logits_acoustic)

            # whether to shrink the hidden or the acoutic distribution
            if not self.args.model.shrink_hidden:
                hidden_output = distribution_acoustic

            blank_id = self.args.dim_ctc_output-1 if self.args.dim_ctc_output else self.args.dim_output-1

            hidden_shrunk, len_no_blank = acoustic_hidden_shrink_tf(
                distribution_acoustic=distribution_acoustic,
                hidden=hidden_output,
                len_acoustic=len_acoustic,
                blank_id=blank_id,
                num_post=self.args.model.num_post,
                frame_expand=self.args.model.frame_expand)

            if (not self.is_train) and (self.args.beam_size>1):
                # infer phrase
                with tf.variable_scope(decoder.name or 'decoder'):
                    logits, decoded, len_decoded = decoder.beam_decode_rerank(
                        hidden_shrunk,
                        len_no_blank)
            else:
                # train phrase
                logits, decoded, len_decoded = decoder(hidden_shrunk, len_no_blank)

            if self.is_train:
                if self.args.model.use_ce_loss:
                    loss = self.ce_loss(
                        logits=logits,
                        labels=tensors_input.label_splits[id_gpu],
                        len_logits=len_acoustic,
                        len_labels=tensors_input.len_label_splits[id_gpu])
                else:
                    loss = self.ocd_loss(
                        logits=logits,
                        len_logits=len_decoded,
                        labels=tensors_input.label_splits[id_gpu],
                        decoded=decoded)

                if self.args.model.confidence_penalty > 0: # utt-level
                    cp_loss = self.args.model.confidence_penalty * \
                        confidence_penalty(logits, len_decoded)/len_decoded

                    loss += cp_loss

                if self.args.model.musk_update:
                    self.idx_update = self.deserve_idx(
                        decoded,
                        len_decoded,
                        tensors_input.label_splits[id_gpu],
                        tensors_input.len_label_splits[id_gpu])
                    loss = tf.reshape(tf.gather(loss, self.idx_update), [-1])
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.decoder.params])
                with tf.name_scope("gradients"):
                    loss = tf.reduce_mean(loss)
                    gradients = self.optimizer.compute_gradients(loss)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        if self.is_train:
            return loss, gradients, \
            [decoded, tensors_input.label_splits[id_gpu], l2_loss]
            # return loss, gradients, tf.no_op()
        else:
            return logits, len_decoded, decoded

    def build_infer_graph(self):
        tensors_input = self.build_infer_input()

        with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model)):
            logits, len_logits, decoded = self.build_single_graph(
                id_gpu=0,
                name_gpu=self.list_gpu_devices[0],
                tensors_input=tensors_input)

            distribution = logits

        return decoded, tensors_input.shape_batch, distribution

    def ocd_loss(self, logits, len_logits, labels, decoded):
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

        mask = tf.sequence_mask(
            len_logits,
            maxlen=tf.shape(logits)[1],
            dtype=logits.dtype)

        loss = tf.reduce_sum(crossent * mask, -1)/tf.reduce_sum(mask, -1)

        return loss

    def ce_loss(self, logits, labels, len_logits, len_labels):
        """
        Compute optimization loss.
        batch shape
        """
        l = tf.reduce_min([tf.shape(logits)[1], tf.shape(labels)[1]])
        # l = tf.Print(l , [l, tf.shape(logits), tf.shape(labels)], message='l, tf.shape(logits), tf.shape(labels): ', summarize=1000)
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
