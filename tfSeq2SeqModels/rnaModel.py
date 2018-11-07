import tensorflow as tf
import logging
from tfModels.tools import choose_device
from tfTools.tfTools import dense_sequence_to_sparse
from tfSeq2SeqModels.seq2seqModel import Seq2SeqModel


class RNAModel(Seq2SeqModel):
    num_Instances = 0

    def __init__(self, tensor_global_step, encoder, decoder, is_train, args,
                 batch=None, embed_table_encoder=None, embed_table_decoder=None,
                 name='RNA_Model'):
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
            self.sample_prob = decoder.sample_prob

            encoded, len_encoded = encoder(
                features=tensors_input.feature_splits[id_gpu],
                len_feas=tensors_input.len_fea_splits[id_gpu])

            if self.helper_type:
                decoder.build_helper(
                    type=self.helper_type,
                    encoded=encoded,
                    len_encoded=len_encoded)

            logits, sample_id, _ = decoder(encoded, len_encoded)

            if self.is_train:
                loss = self.rna_loss(
                    logits=logits,
                    len_logits=len_encoded,
                    labels=tensors_input.label_splits[id_gpu],
                    len_labels=tensors_input.len_label_splits[id_gpu],
                    encoded=encoded,
                    len_encoded=len_encoded)

                with tf.name_scope("gradients"):
                    gradients = self.optimizer.compute_gradients(loss)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        if self.is_train:
            return loss, gradients
        else:
            return logits, len_encoded, sample_id

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
            loss = tf.reduce_mean(ctc_loss_batch) # utter-level ctc loss

        if self.args.model.confidence_penalty:
            print('using confidence penalty')
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

    def policy_learning(self, logits, len_logits, labels, len_labels, encoded, len_encoded):
        assert (encoded is not None) and (len_encoded is not None)
        from tfModels.ctcModel import CTCModel
        from tfTools.tfTools import pad_to_same

        # with tf.variable_scope('policy_learning'):
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            decoder_sample = self.gen_decoder(
                is_train=False,
                embed_table=self.embed_table_decoder,
                global_step=self.global_step,
                args=self.args)
            decoder_sample.build_helper(
                type=self.args.model.decoder.sampleHelper,
                encoded=encoded,
                len_encoded=len_encoded)

            logits_sample, sample_id_sample, _ = decoder_sample(encoded, len_encoded)

        label_sparse = dense_sequence_to_sparse(labels, len_labels)

        # bias(gready) decode
        decoded_sparse = self.rna_decode(logits, len_logits)
        wer_bias = tf.edit_distance(decoded_sparse, label_sparse, normalize=True)
        wer_bias = tf.stop_gradient(wer_bias)

        # sample decode
        sample_sparse = self.rna_decode(logits_sample, len_logits)
        wer = tf.edit_distance(sample_sparse, label_sparse, normalize=True)
        sample = tf.sparse_to_dense(
            sparse_indices=sample_sparse.indices,
            output_shape=sample_sparse.dense_shape,
            sparse_values=sample_sparse.values,
            default_value=0,
            validate_indices=True)
        len_sample = tf.count_nonzero(sample, -1, dtype=tf.int32)
        # wer_bias = tf.Print(wer_bias, [len_sample], message='len_sample', summarize=1000)
        seq_sample, labels = pad_to_same([sample, labels])
        seq_sample = tf.where(len_sample<1, labels, seq_sample)
        len_sample = tf.where(len_sample<1, len_labels, len_sample)

        reward = wer_bias - wer

        rl_loss, _ = CTCModel.policy_ctc_loss(
            logits=logits_sample,
            len_logits=len_logits,
            flabels=sample,
            len_flabels=len_sample,
            batch_reward=reward,
            ctc_merge_repeated=False,
            args=self.args)

        return rl_loss

    def expected_loss(self, logits, len_logits, labels, len_labels):
        label_sparse = dense_sequence_to_sparse(labels, len_labels)
        list_decoded_sparse = self.rna_decode(logits, len_logits, beam_reserve=True)
        list_wer = []
        for decoded_sparse in list_decoded_sparse:
            decoded_sparse = tf.to_int32(decoded_sparse)
            list_wer.append(tf.edit_distance(decoded_sparse, label_sparse, normalize=True))
        wer_bias = tf.reduce_mean(list_wer)
        ep_loss = (list_wer - wer_bias)/len(list_wer)

        return ep_loss
