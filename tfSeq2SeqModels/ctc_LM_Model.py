import tensorflow as tf
import logging
import sys

from tfModels.tools import choose_device
from tfTools.tfTools import acoustic_shrink, pad_to, dense_sequence_to_sparse
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
        # the fc decdoer use the dim_ctc_output
        args.dim_ctc_output = len(args.phone.token2idx)
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
            acoustic, alignment, len_acoustic = self.fc_decoder(hidden_output, len_hidden_output)

            if not self.args.model.train_encoder:
                acoustic = tf.stop_gradient(acoustic)
                len_acoustic = tf.stop_gradient(len_acoustic)
            # used to guide the shrinking of the hidden_output
            distribution_acoustic = tf.nn.softmax(acoustic)

            # whether to shrink the hidden or the acoustic
            if self.args.model.shrink_hidden:
                # recommand
                hidden_shrinked, len_phone, acoustic_shrinked = acoustic_hidden_shrink(
                    distribution_acoustic=distribution_acoustic,
                    hidden=hidden_output,
                    len_acoustic=len_acoustic,
                    dim_output=self.args.dim_ctc_output,
                    hidden_size=self.args.model.encoder.num_cell_units,
                    num_avg=self.args.model.num_avg)
                phone = tf.argmax(acoustic_shrinked, -1)
            else:
                distribution_no_blank, len_phone = acoustic_shrink(
                    distribution_acoustic=distribution_acoustic,
                    len_acoustic=len_acoustic,
                    dim_output=self.args.dim_ctc_output)
                hidden_shrinked = distribution_no_blank

            if (not self.is_train) and (self.args.beam_size>1):
                # infer phrase
                if self.args.dirs.lm_checkpoint:
                    logging.info('beam search with language model ...')
                    with tf.variable_scope(decoder.name or 'decoder'):
                        if self.args.model.rerank:
                            decoded_beam, decoded, len_decode = decoder.beam_decode_rerank(
                                hidden_shrinked,
                                len_phone)
                        else:
                            logits, decoded, len_decode = decoder.beam_decode_lm(
                                hidden_shrinked,
                                len_phone)
                else:
                    logging.info('beam search ...')
                    with tf.variable_scope(decoder.name or 'decoder'):
                        logits, decoded, len_decode = decoder.beam_decode(
                            hidden_shrinked,
                            len_phone)
            else:
                # train phrase
                print('greedy search ...')
                logits, decoded, len_decode = decoder(hidden_shrinked, len_phone)
                decoded_beam = tf.no_op()

            if self.is_train:
                ocd_loss = self.ocd_loss(
                    logits=logits,
                    len_logits=len_decode,
                    labels=tensors_input.label_splits[id_gpu],
                    decoded=decoded,
                    len_decode=len_decode)

                ctc_loss = self.ctc_loss(
                    logits=acoustic,
                    len_logits=len_acoustic,
                    labels=tensors_input.phone_splits[id_gpu],
                    len_labels=tensors_input.len_phone_splits[id_gpu])
                loss = ocd_loss + ctc_loss

                if self.args.musk_update:
                    self.idx_update = self.deserve_idx(
                        decoded,
                        len_decode,
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
            [decoded, tensors_input.label_splits[id_gpu], distribution_acoustic, len_acoustic, phone, len_phone, ctc_loss, ocd_loss]
            # return loss, gradients, tf.no_op()
        else:
            return decoded, [decoded_beam, phone]

    def build_infer_graph(self):
        tensors_input = self.build_infer_input()

        with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model)):
            decoded, infer_log = self.build_single_graph(
                id_gpu=0,
                name_gpu=self.list_gpu_devices[0],
                tensors_input=tensors_input)

        return decoded, tensors_input.shape_batch, infer_log

    def build_tf_input(self):
        """
        stand training input
        """
        from collections import namedtuple
        tensors_input = namedtuple('tensors_input',
            'feature_splits, label_splits, phone_splits, \
             len_fea_splits, len_label_splits, len_phone_splits, \
             shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                # split input data alone batch axis to gpus
                tensors_input.feature_splits = tf.split(self.batch[0], self.num_gpus, name="feature_splits")
                tensors_input.label_splits = tf.split(self.batch[1], self.num_gpus, name="label_splits")
                tensors_input.phone_splits = tf.split(self.batch[2], self.num_gpus, name="phone_splits")
                tensors_input.len_fea_splits = tf.split(self.batch[3], self.num_gpus, name="len_fea_splits")
                tensors_input.len_label_splits = tf.split(self.batch[4], self.num_gpus, name="len_label_splits")
                tensors_input.len_phone_splits = tf.split(self.batch[5], self.num_gpus, name="len_phone_splits")
        tensors_input.shape_batch = tf.shape(self.batch[0])

        return tensors_input

    def ocd_loss(self, logits, len_logits, labels, decoded, len_decode):
        """
        the logits length is the sample_id length
        return batch shape loss
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
            loss = tf.reduce_sum(crossent * pad_mask, -1) # utt-level
        else:
            loss = tf.reduce_sum(crossent * pad_mask)/tf.reduce_sum(pad_mask) # token-level

        if self.args.model.decoder2.confidence_penalty:
            ls_loss = self.args.model.decoder2.confidence_penalty * \
                        confidence_penalty(logits, len_decode)
            loss += ls_loss

        return loss

    def ctc_loss(self, logits, len_logits, labels, len_labels):
        """
        No valid path found: It is possible that no valid path is found if the
        activations for the targets are zero.
        ctc_merge_repeated is always False
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
                ctc_merge_repeated=False,
                ignore_longer_outputs_than_inputs=True,
                time_major=False)

        if self.args.model.decoder.confidence_penalty:
            ls_loss = self.args.model.decoder.confidence_penalty * \
                        confidence_penalty(logits, len_logits)
            loss += ls_loss

        return loss

    def deserve_idx(self, decoded, len_decode, labels, len_labels):
        """
        if one sent is correct during training, then not to train on it
        """
        decoded_sparse = dense_sequence_to_sparse(
            seq=decoded,
            len_seq=len_decode)
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


def acoustic_hidden_shrink(distribution_acoustic, hidden, len_acoustic, dim_output, hidden_size, num_avg=1):
    """
    filter the hidden where blank_id dominants in distribution_acoustic.
    the blank_id default to be dim_output-1.
    incompletely tested
    the len_no_blank will be set one if distribution_acoustic is all blank dominanted
    shrink the hidden instead of distribution_acoustic
    """
    blank_id = dim_output - 1
    no_blank = tf.to_int32(tf.not_equal(tf.argmax(distribution_acoustic, -1), blank_id))
    mask_acoustic = tf.sequence_mask(len_acoustic, maxlen=tf.shape(distribution_acoustic)[1], dtype=no_blank.dtype)
    no_blank *= mask_acoustic
    len_no_blank = tf.reduce_sum(no_blank, -1)

    batch_size = tf.shape(no_blank)[0]
    seq_len = tf.shape(no_blank)[1]

    # the patch, the length of shrinked hidden is at least 1
    no_blank = tf.where(
        tf.not_equal(len_no_blank, 0),
        no_blank,
        tf.concat([tf.ones([batch_size, 1], dtype=tf.int32),
                   tf.zeros([batch_size, seq_len-1], dtype=tf.int32)], 1)
    )
    len_no_blank = tf.where(
        tf.not_equal(len_no_blank, 0),
        len_no_blank,
        tf.ones_like(len_no_blank, dtype=tf.int32)
    )

    batch_size = tf.size(len_no_blank)
    max_len = tf.reduce_max(len_no_blank)
    hidden_shrinked_init = tf.zeros([1, max_len, hidden_size])
    acoustic_shrinked_init = tf.zeros([1, max_len, dim_output])

    # average the hidden of n frames
    if num_avg == 3:
        hidden = (hidden + \
                tf.concat([hidden[:, 1:, :], hidden[:, -1:, :]], 1) + \
                tf.concat([hidden[:, :1, :], hidden[:, :-1, :]], 1)) / num_avg

    def step(i, hidden_shrinked, acoustic_shrinked):
        # loop over the batch
        shrinked = tf.gather(hidden[i], tf.reshape(tf.where(no_blank[i]>0), [-1]))
        shrinked_paded = pad_to(shrinked, max_len, axis=0)
        hidden_shrinked = tf.concat([hidden_shrinked,
                                       tf.expand_dims(shrinked_paded, 0)], 0)

        # acoustic shirinked
        shrinked = tf.gather(distribution_acoustic[i], tf.reshape(tf.where(no_blank[i]>0), [-1]))
        shrinked_paded = pad_to(shrinked, max_len, axis=0)
        acoustic_shrinked = tf.concat([acoustic_shrinked,
                                       tf.expand_dims(shrinked_paded, 0)], 0)

        return i+1, hidden_shrinked, acoustic_shrinked

    i, hidden_shrinked, acoustic_shrinked = tf.while_loop(
        cond=lambda i, *_: tf.less(i, batch_size),
        body=step,
        loop_vars=[0, hidden_shrinked_init, acoustic_shrinked_init],
        shape_invariants=[tf.TensorShape([]),
                          tf.TensorShape([None, None, hidden_size]),
                          tf.TensorShape([None, None, dim_output])]
    )
    hidden_shrinked = hidden_shrinked[1:, :, :]
    acoustic_shrinked = acoustic_shrinked[1:, :, :]

    return hidden_shrinked, len_no_blank, acoustic_shrinked
