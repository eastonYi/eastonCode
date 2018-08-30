import tensorflow as tf
import logging
# from collections import namedtuple
# from tf.contrib.seq2seq import CustomHelper
# from tf.contrib.rnn import LSTMStateTuple
# from tf.contrib.framework import nest
from tfModels.tools import choose_device
from tfTools.tfTools import dense_sequence_to_sparse
from tfSeq2SeqModels.seq2seqModel import Seq2SeqModel


class RNAModel(Seq2SeqModel):
    num_Instances = 0

    def __init__(self, tensor_global_step, encoder, decoder, is_train, args,
                 batch=None, embed_table_encoder=None, embed_table_decoder=None,
                 name='RNA_Model'):
        super().__init__(tensor_global_step, encoder, decoder, is_train, args,
                     batch, embed_table_encoder, embed_table_decoder, name)

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):

        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            encoder = self.gen_encoder(
                is_train=self.is_train,
                embed_table=self.embed_table_encoder,
                args=self.args)
            decoder = self.gen_decoder(
                is_train=self.is_train,
                embed_table=self.embed_table_decoder,
                global_step=self.global_step,
                args=self.args)
            self.sample_prob = decoder.sample_prob

            encoder_input = encoder.build_input(
                id_gpu=id_gpu,
                tensors_input=tensors_input)
            encoded, len_encoded = encoder(encoder_input)

            decoder_input = decoder.build_input(
                id_gpu=id_gpu,
                encoded=encoded,
                len_encoded=len_encoded,
                tensors_input=tensors_input)
            decoder.build_helper(
                type=self.helper_type,
                labels=decoder_input.input_labels,
                len_labels=decoder_input.len_labels,
                encoded=encoded,
                len_encoded=len_encoded,
                batch_size=tf.size(len_encoded))

            logits, sample_id, _ = decoder(decoder_input)

            if self.is_train:
                loss = self.rna_loss(
                    logits=logits,
                    len_logits=len_encoded,
                    labels=tensors_input.label_splits[id_gpu],
                    len_labels=tensors_input.len_label_splits[id_gpu])

                with tf.name_scope("gradients"):
                    gradients = self.optimizer.compute_gradients(loss)

            self.__class__.num_Model += 1
            logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
                self.__class__.__name__, name_gpu, self.__class__.num_Model))

            return (loss, gradients) if self.is_train else sample_id

    def rna_loss(self, logits, len_logits, labels, len_labels):
        with tf.name_scope("rna_loss"):
            labels_sparse = dense_sequence_to_sparse(
                labels,
                len_labels)
            ctc_loss_batch = tf.nn.ctc_loss(
                labels_sparse,
                logits,
                sequence_length=len_logits,
                ignore_longer_outputs_than_inputs=True,
                time_major=False)
            loss = tf.reduce_mean(ctc_loss_batch) # utter-level ctc loss

        return loss

    # def rna_loss(seqs_input, seqs_labels, seq_input_lens, seq_label_lens, multi_forward, embedding, rna):
    #     """
    #     using dynamic programming, which generate a lattice form seq_input to seq_label_lens
    #     each node is a state of the decoder. Feeding an input and label to push a movement
    #     on the lattic.
    #     """
    #     def right_shift_rows(p, shift, pad):
    #         assert type(shift) is int
    #         return tf.concat([tf.ones((tf.shape(p)[0], shift), dtype=tf.float32)*pad,
    #                           p[:, :-shift]], axis=1)
    #
    #     def choose_state(multi_state, multi_state_pre, prob, prob_pre, forward_vars, forward_vars_pre):
    #         # sum_log(prob, forward_vars), sum_log(prob_pre, forward_vars_pre)
    #         # TODO
    #         return multi_state
    #
    #     def step(extend_forward_vars, step_input_x):
    #         # batch x len_label x (embedding+dim_feature)
    #         step_input = tf.concat([tf.tile(tf.expand_dims(step_input_x, 1), [1, N+1, 1]),
    #                                     blank_emb], 2)
    #         step_input_pre = tf.concat([tf.tile(tf.expand_dims(step_input_x, 1), [1, N+1, 1]),
    #                                     seqs_labels_emb], 2)
    #
    #         forward_vars, multi_state = extend_forward_vars
    #         forward_vars_pre = right_shift_rows(forward_vars, 1, LOG_ZERO)
    #
    #         # two distinct cell states are going to merge. Here we choose one of them.
    #         # distrib: batch x len_label x size_vocab
    #         distrib, multi_state = self.multi_forward(step_input, multi_state)
    #         distrib_pre, multi_state_pre = self.multi_forward(step_input_pre, multi_state)
    #
    #         prob = distrib[:, :, 0] # prob of blank: batch x len_label
    #         index_batch = range_batch([size_batch, N+1])
    #         index_len = range_batch([size_batch, N+1], False)
    #         prob_pre = tf.gather_nd(distrib_pre, tf.stack([index_batch, index_len, seqs_labels], -1))
    #
    #         multi_state = choose_state(multi_state, multi_state_pre, prob, prob_pre, forward_vars, forward_vars_pre)
    #         forward_vars = sum_log(forward_vars_pre + prob_pre, forward_vars + prob)
    #
    #         return [forward_vars, multi_state]
    #
    #     size_batch = tf.shape(seqs_input)[0]
    #     T = tf.shape(seqs_input)[1]
    #     N = tf.shape(seqs_labels)[1]
    #
    #     # data: batch x len_label x (embedding+dim_feature), at each time
    #     seqs_labels_endpad = tf.concat([seqs_labels, tf.zeros([size_batch, 1])], 1)
    #     seqs_labels_emb = tf.nn.embedding_lookup(embedding, seqs_labels_endpad)
    #     blank_emb = tf.nn.embedding_lookup(embedding, tf.zeros_like(seqs_labels, tf.int32))
    #     seqs_input_timeMajor = tf.transpose(seqs_input, ((1, 0, 2))) # actually, len major
    #
    #     # forward vars: batch x (len_label+1)
    #     tail = tf.ones((size_batch, N), dtype=tf.float32) * LOG_ZERO
    #     head = tf.ones((size_batch, 1), dtype=tf.float32) * LOG_ONE
    #     forward_vars_init = tf.concat([head, tail], -1)
    #
    #     # state: len_label
    #     multi_state_init = rna.zero_state(N+1, size_batch)
    #
    #     # forward loop
    #     forward_vars_steps = tf.scan(
    #         step,
    #         seqs_input_timeMajor,
    #         [forward_vars_init, multi_state_init])
