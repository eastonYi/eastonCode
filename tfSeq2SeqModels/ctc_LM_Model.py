import tensorflow as tf
import logging
import sys

from tfModels.tools import choose_device
from tfTools.tfTools import acoustic_shrink, dense_sequence_to_sparse
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
                args=self.args)
            self.decoder = decoder = self.gen_decoder2(
                is_train=self.is_train,
                embed_table=self.embedding_tabel,
                global_step=self.global_step,
                args=self.args,
                name='decoder2')

            hidden_output, len_hidden_output = self.encoder(
                features=tensors_input.feature_splits[id_gpu],
                len_feas=tensors_input.len_fea_splits[id_gpu])
            encoded, alignment, len_encoded = self.fc_decoder(hidden_output, len_hidden_output)

            if self.args.model.train_encoder:
                len_acoustic = len_encoded
            else:
                encoded = tf.stop_gradient(encoded)
                len_acoustic = tf.stop_gradient(len_encoded)
            distribution_acoustic = tf.nn.softmax(encoded)

            distribution_no_blank, len_no_blank = acoustic_shrink(
                distribution_acoustic=distribution_acoustic,
                len_acoustic=len_acoustic,
                dim_output=self.args.dim_output)
            if (not self.is_train) and (self.args.beam_size>1):
                print('beam search ...')
                with tf.variable_scope(decoder.name or 'decoder'):
                    logits, decoded, len_decode = decoder.beam_decode(distribution_no_blank, len_no_blank)
            else:
                print('greedy search ...')
                logits, decoded, len_decode = decoder(distribution_no_blank, len_no_blank)


            if self.is_train:
                loss = self.ocd_loss(
                    logits=logits,
                    len_logits=len_decode,
                    labels=tensors_input.label_splits[id_gpu],
                    decoded=decoded)
                if self.args.model.confidence_penalty:
                    ls_loss = self.args.model.confidence_penalty * confidence_penalty(logits, len_decode)
                    ls_loss = tf.reduce_mean(ls_loss)
                    loss += ls_loss
                if self.args.model.ctc_loss_for_acoutic:
                    ctc_loss = self.ctc_loss(
                        logits=encoded,
                        len_logits=len_encoded,
                        labels=tensors_input.label_splits[id_gpu],
                        len_labels=tensors_input.len_label_splits[id_gpu])
                    loss += ctc_loss

                with tf.name_scope("gradients"):
                    gradients = self.optimizer.compute_gradients(loss)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        if self.is_train:
            return loss, gradients, [decoded, tensors_input.label_splits[id_gpu], distribution_acoustic, len_acoustic, len_no_blank, distribution_no_blank]
            # return loss, gradients, tf.no_op()
        else:
            return logits, len_encoded, decoded

    def build_infer_graph(self):
        tensors_input = self.build_infer_input()

        with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model)):
            logits, len_logits, sample_id = self.build_single_graph(
                id_gpu=0,
                name_gpu=self.list_gpu_devices[0],
                tensors_input=tensors_input)

            # distribution = tf.nn.softmax(logits)
            distribution = logits

        return sample_id, tensors_input.shape_batch, distribution

    def ocd_loss(self, logits, len_logits, labels, decoded):
        """
        the logits length is the sample_id length
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

        if self.args.model.utt_level_loss:
            loss = tf.reduce_mean(tf.reduce_sum(crossent * pad_mask, -1)) # utt-level
        else:
            loss = tf.reduce_sum(crossent * pad_mask)/tf.reduce_sum(pad_mask) # token-level

        return loss

    def ctc_loss(self, logits, len_logits, labels, len_labels):
        """
        No valid path found: It is possible that no valid path is found if the
        activations for the targets are zero.
        """
        ctc_merge_repeated = self.args.model.decoder.ctc_merge_repeated
        with tf.name_scope("ctc_loss"):
            labels_sparse = dense_sequence_to_sparse(
                labels,
                len_labels)
            ctc_loss_batch = tf.nn.ctc_loss(
                labels_sparse,
                logits,
                sequence_length=len_logits,
                ctc_merge_repeated=ctc_merge_repeated,
                ignore_longer_outputs_than_inputs=True,
                time_major=False)
            loss = tf.reduce_mean(ctc_loss_batch) # utter-level ctc loss

        if self.args.model.confidence_penalty:
            ls_loss = self.args.model.confidence_penalty * confidence_penalty(logits, len_logits)
            ls_loss = tf.reduce_mean(ls_loss)
            loss += ls_loss

        return loss

    def get_embedding(self, embed_table, size_input, size_embedding):
        if size_embedding and (type(embed_table) is not tf.Variable):
            with tf.device("/cpu:0"):
                with tf.variable_scope(self.name, reuse=(self.__class__.num_Model > 0)):
                    embed_table = tf.get_variable(
                        "embedding", [size_input, size_embedding], dtype=tf.float32)

        return embed_table
