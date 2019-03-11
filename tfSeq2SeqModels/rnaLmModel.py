import tensorflow as tf
import logging
from tfModels.tools import choose_device
from tfTools.tfTools import dense_sequence_to_sparse
from tfSeq2SeqModels.rnaModel import RNAModel

class RNALMModel(RNAModel):
    num_Instances = 0

    def __init__(self, tensor_global_step, encoder, decoder, is_train, args,
                 batch=None, embed_table_encoder=None, embed_table_decoder=None,
                 name='RNA_LM_Model'):
        self.name = name
        super().__init__(tensor_global_step, encoder, decoder, is_train, args,
                         batch,
                         embed_table_encoder=None,
                         embed_table_decoder=embed_table_decoder,
                         name=name)

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        tf.get_variable_scope().set_initializer(tf.variance_scaling_initializer(
            1.0, mode="fan_avg", distribution="uniform"))
        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            encoder = self.gen_encoder(
                is_train=self.is_train,
                args=self.args)

            self.decoder = decoder = self.gen_decoder(
                is_train=self.is_train,
                embed_table=self.embed_table_decoder,
                global_step=self.global_step,
                args=self.args)

            self.lm_decoder = self.gen_decoder(
                is_train=self.is_train,
                embed_table=self.embed_table_decoder,
                global_step=self.global_step,
                args=self.args)

            self.schedule = decoder.schedule

            encoded, len_encoded = encoder(
                features=tensors_input.feature_splits[id_gpu],
                len_feas=tensors_input.len_fea_splits[id_gpu])

            if self.helper_type:
                decoder.build_helper(
                    type=self.helper_type,
                    encoded=encoded,
                    len_encoded=len_encoded)

            logits, decoded, len_decoded = decoder(encoded, len_encoded)

            if self.is_train:
                loss = 0
                if self.args.rna_train:
                    rna_loss = self.rna_loss(
                        logits=logits,
                        len_logits=len_encoded,
                        labels=tensors_input.label_splits[id_gpu],
                        len_labels=tensors_input.len_label_splits[id_gpu],
                        encoded=encoded,
                        len_encoded=len_encoded)
                    loss += rna_loss
                if self.args.OCD_train > 0:
                    ocd_loss = self.args.OCD_train * self.ocd_loss(
                        logits=logits,
                        len_logits=len_decoded,
                        labels=tensors_input.label_splits[id_gpu],
                        decoded=decoded)
                    loss += ocd_loss

                with tf.name_scope("gradients"):
                    gradients = self.optimizer.compute_gradients(loss)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        if self.is_train:
            return loss, gradients, [decoded, tensors_input.label_splits[id_gpu], ocd_loss]
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
            # sample_id
            # if sample_id.get_shape().ndims == 3:
            #     sample_id = sample_id[:,:,0]

            # ctc decode
            # why not simply use the sample_id: https://distill.pub/2017/ctc/#inference
            decoded_sparse = self.rna_decode(logits, len_logits)
            decoded = tf.sparse_to_dense(
                sparse_indices=decoded_sparse.indices,
                output_shape=decoded_sparse.dense_shape,
                sparse_values=decoded_sparse.values,
                default_value=0,
                validate_indices=True)
            distribution = tf.nn.softmax(logits)

        return decoded, tensors_input.shape_batch, distribution

    def rna_loss(self, logits, len_logits, labels, len_labels, encoded=None, len_encoded=None):
        with tf.name_scope("rna_loss"):
            labels_sparse = dense_sequence_to_sparse(
                labels,
                len_labels)
            ctc_loss_batch = tf.nn.ctc_loss(
                labels_sparse,
                logits,
                sequence_length=len_logits,
                ctc_merge_repeated=False,
                ignore_longer_outputs_than_inputs=True,
                time_major=False)
            loss = ctc_loss_batch # utter-level ctc loss

        if self.args.model.confidence_penalty:
            logging.info('using confidence penalty')
            with tf.name_scope("confidence_penalty"):
                real_probs = tf.nn.softmax(logits)
                prevent_nan_constant = tf.constant(1e-10)
                real_probs += prevent_nan_constant

                neg_entropy = tf.reduce_sum(real_probs * tf.log(real_probs), axis=-1)
                ls_loss = self.args.model.confidence_penalty * tf.reduce_sum(neg_entropy, axis=-1)

            loss += ls_loss

        if self.args.model.policy_learning:
            rl_loss = self.policy_learning(logits, len_logits, labels, len_labels, encoded, len_encoded)
            loss += self.args.model.policy_learning * rl_loss

        if self.args.model.expected_loss:
            ep_loss = self.expected_loss(logits, len_logits, labels, len_labels)
            loss += self.args.model.expected_loss * ep_loss

        loss = tf.reduce_mean(loss)

        return loss

    def ocd_loss(self, logits, len_logits, labels, decoded):
        """
        the logits length is the sample_id length
        the len_labels is useless(??)
        """
        from tfModels.OCDLoss import OCD_with_blank_loss

        optimal_distributions, optimal_targets = OCD_with_blank_loss(
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

        if self.args.model.decoder.loss_on_blank:
            mask = pad_mask
        else:
            blank_id = self.args.dim_output-1
            blank_mask = tf.to_float(tf.not_equal(decoded, blank_id))
            mask = pad_mask * blank_mask
        loss = tf.reduce_sum(crossent * mask)/tf.reduce_sum(mask)
        # loss = tf.reduce_sum(crossent * mask, -1)
        # loss = tf.reduce_mean(loss)

        return loss


    def rna_decode(self, logits=None, len_logits=None, beam_reserve=False):
        beam_size = self.args.beam_size
        logits_timeMajor = tf.transpose(logits, [1, 0, 2])

        if beam_size == 1:
            decoded_sparse = tf.to_int32(tf.nn.ctc_greedy_decoder(
                logits_timeMajor,
                len_logits,
                merge_repeated=False)[0][0])
        else:
            if beam_reserve:
                decoded_sparse = tf.nn.ctc_beam_search_decoder(
                    logits_timeMajor,
                    len_logits,
                    beam_width=beam_size,
                    merge_repeated=False)[0]
            else:
                decoded_sparse = tf.to_int32(tf.nn.ctc_beam_search_decoder(
                    logits_timeMajor,
                    len_logits,
                    beam_width=beam_size,
                    merge_repeated=False)[0][0])

        return decoded_sparse
