import tensorflow as tf
from tfTools.tfTools import dense_sequence_to_sparse

epsilon = 1e-10


def confidence_penalty(logits, len_logits):
    """
    H = - \sum(P * logP)
    loss = -H = \sum(P * logP)
    sequence-level loss
    the minimal value of cp is negative and is conditioned on the dim
    """
    with tf.name_scope("confidence_penalty"):
        real_probs = tf.nn.softmax(logits)+epsilon
        neg_entropy = tf.reduce_sum(real_probs * tf.log(real_probs), -1)
        pad_mask = tf.sequence_mask(
            len_logits,
            maxlen=tf.shape(logits)[1],
            dtype=logits.dtype)
        loss = tf.reduce_sum(neg_entropy*pad_mask, -1)

    return loss


def policy_learning(logits, len_logits, decoded_sparse, labels, len_labels, softmax_temperature, dim_output, args):
    from tfModels.CTCLoss import ctc_sample, ctc_reduce_map
    from tfTools.tfTools import sparse_shrink, pad_to_same
    print('using policy learning')
    with tf.name_scope("policy_learning"):
        label_sparse = dense_sequence_to_sparse(labels, len_labels)
        wer_bias = tf.edit_distance(decoded_sparse, label_sparse, normalize=True)
        wer_bias = tf.stop_gradient(wer_bias)

        sampled_align = ctc_sample(logits, softmax_temperature)
        sample_sparse = ctc_reduce_map(sampled_align, id_blank=dim_output-1)
        wer = tf.edit_distance(sample_sparse, label_sparse, normalize=True)
        seq_sample, len_sample, _ = sparse_shrink(sample_sparse)

        # ==0 is not success!!
        seq_sample, labels = pad_to_same([seq_sample, labels])
        seq_sample = tf.where(len_sample<1, labels, seq_sample)
        len_sample = tf.where(len_sample<1, len_labels, len_sample)

        reward = wer_bias - wer

        rl_loss, _ = policy_ctc_loss(
            logits=logits,
            len_logits=len_logits,
            flabels=seq_sample,
            len_flabels=len_sample,
            batch_reward=reward,
            args=args)
# def embedding_decentralization(embedding_tabel):


def policy_ctc_loss(logits, len_logits, flabels, len_flabels, batch_reward, args, ctc_merge_repeated=True):
    """
    flabels: not the ground-truth
    if len_flabels=None, means the `flabels` is sparse
    """
    from tfTools.math_tf import non_linear
    from tfTools.tfTools import dense_sequence_to_sparse

    with tf.name_scope("policy_ctc_loss"):
        if len_flabels is not None:
            flabels_sparse = dense_sequence_to_sparse(
                flabels,
                len_flabels)
        else:
            flabels_sparse = flabels

        ctc_loss_batch = tf.nn.ctc_loss(
            flabels_sparse,
            logits,
            sequence_length=len_logits,
            ignore_longer_outputs_than_inputs=True,
            ctc_merge_repeated=ctc_merge_repeated,
            time_major=False)
        ctc_loss_batch *= batch_reward
        ctc_loss_batch = non_linear(
            ctc_loss_batch,
            args.model.non_linear,
            args.model.min_reward)
        loss = tf.reduce_mean(ctc_loss_batch) # utter-level ctc loss

    return loss, ctc_loss_batch


# def policy_learning(self, logits, len_logits, labels, len_labels, encoded, len_encoded):
#     assert (encoded is not None) and (len_encoded is not None)
#     from tfModels.ctcModel import CTCModel
#     from tfTools.tfTools import pad_to_same
#
#     # with tf.variable_scope('policy_learning'):
#     with tf.variable_scope(tf.get_variable_scope(), reuse=True):
#         decoder_sample = self.gen_decoder(
#             is_train=False,
#             embed_table=self.embed_table_decoder,
#             global_step=self.global_step,
#             args=self.args)
#         decoder_sample.build_helper(
#             type=self.args.model.decoder.sampleHelper,
#             encoded=encoded,
#             len_encoded=len_encoded)
#
#         logits_sample, sample_id_sample, _ = decoder_sample(encoded, len_encoded)
#
#     label_sparse = dense_sequence_to_sparse(labels, len_labels)
#
#     # bias(gready) decode
#     decoded_sparse = self.rna_decode(logits, len_logits)
#     wer_bias = tf.edit_distance(decoded_sparse, label_sparse, normalize=True)
#     wer_bias = tf.stop_gradient(wer_bias)
#
#     # sample decode
#     sample_sparse = self.rna_decode(logits_sample, len_logits)
#     wer = tf.edit_distance(sample_sparse, label_sparse, normalize=True)
#     sample = tf.sparse_to_dense(
#         sparse_indices=sample_sparse.indices,
#         output_shape=sample_sparse.dense_shape,
#         sparse_values=sample_sparse.values,
#         default_value=0,
#         validate_indices=True)
#     len_sample = tf.count_nonzero(sample, -1, dtype=tf.int32)
#     # wer_bias = tf.Print(wer_bias, [len_sample], message='len_sample', summarize=1000)
#     seq_sample, labels = pad_to_same([sample, labels])
#     seq_sample = tf.where(len_sample<1, labels, seq_sample)
#     len_sample = tf.where(len_sample<1, len_labels, len_sample)
#
#     reward = wer_bias - wer
#
#     rl_loss, _ = policy_ctc_loss(
#         logits=logits_sample,
#         len_logits=len_logits,
#         flabels=sample,
#         len_flabels=len_sample,
#         batch_reward=reward,
#         ctc_merge_repeated=False,
#         args=self.args)
#
#     return rl_loss
#
#
# def expected_loss(self, logits, len_logits, labels, len_labels):
#     label_sparse = dense_sequence_to_sparse(labels, len_labels)
#     list_decoded_sparse = self.rna_decode(logits, len_logits, beam_reserve=True)
#     list_wer = []
#     for decoded_sparse in list_decoded_sparse:
#         decoded_sparse = tf.to_int32(decoded_sparse)
#         list_wer.append(tf.edit_distance(decoded_sparse, label_sparse, normalize=True))
#     wer_bias = tf.reduce_mean(list_wer)
#     ep_loss = (list_wer - wer_bias)/len(list_wer)
#
#     return ep_loss
