import tensorflow as tf
import logging
import sys

from tfModels.tools import choose_device
from tfTools.tfTools import dense_sequence_to_sparse, pad_to
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

            encoded = tf.stop_gradient(encoded)
            len_acoustic = tf.stop_gradient(len_encoded)
            distribution_acoustic = tf.nn.softmax(encoded)

            distribution_no_blank, len_no_blank = self.acoustic_shrink(distribution_acoustic, len_acoustic)
            logits, decoded, len_decode = decoder(distribution_no_blank, len_no_blank)

            if self.is_train:
                loss = self.ocd_loss(
                    logits=logits,
                    len_logits=len_decode,
                    labels=tensors_input.label_splits[id_gpu],
                    decoded=decoded)

                with tf.name_scope("gradients"):
                    gradients = self.optimizer.compute_gradients(loss)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        if self.is_train:
            return loss, gradients, [decoded, tensors_input.label_splits[id_gpu], distribution_acoustic, len_acoustic, distribution_no_blank]
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

            distribution = tf.nn.softmax(logits)

        return sample_id, tensors_input.shape_batch, distribution

    def ocd_loss(self, logits, len_logits, labels, decoded):
        """
        the logits length is the sample_id length
        the len_labels is useless(??)
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

        if self.args.model.decoder.loss_on_blank:
            mask = pad_mask
        else:
            blank_id = self.args.dim_output-1
            blank_mask = tf.to_float(tf.not_equal(decoded, blank_id))
            mask = pad_mask * blank_mask
        # if all is blank, the sum of mask would be 0, and loss be NAN
        loss_batch = tf.reduce_sum(crossent * mask, -1)
        loss = tf.reduce_mean(loss_batch)

        return loss

    def acoustic_shrink(self, distribution_acoustic, len_acoustic):
        no_blank = tf.to_int32(tf.not_equal(tf.argmax(distribution_acoustic, -1), self.args.dim_output-1))
        mask_acoustic = tf.sequence_mask(len_acoustic, maxlen=tf.shape(distribution_acoustic)[1], dtype=no_blank.dtype)
        no_blank = mask_acoustic*no_blank
        len_no_blank = tf.reduce_sum(no_blank, -1)
        batch_size = tf.size(len_no_blank)
        max_len = tf.reduce_max(len_no_blank)
        acoustic_shrinked_init = tf.zeros([1, max_len, self.args.dim_output])

        def step(i, acoustic_shrinked):
            shrinked = tf.gather(distribution_acoustic[i], tf.where(no_blank[i]>0)[0])
            shrinked_paded = pad_to(shrinked, max_len, axis=0)
            acoustic_shrinked = tf.concat([acoustic_shrinked,
                                           tf.expand_dims(shrinked_paded, 0)], 0)
            return i+1, acoustic_shrinked

        _, acoustic_shrinked = tf.while_loop(
            cond=lambda i, *_: tf.less(i, batch_size),
            body=step,
            loop_vars=[0, acoustic_shrinked_init],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([None, None, self.args.dim_output])]
        )
        # acoustic_shrinked = tf.gather_nd(distribution_acoustic, tf.where(no_blank>0))
        acoustic_shrinked = acoustic_shrinked[1:, :, :]
        return acoustic_shrinked, len_no_blank

    def get_embedding(self, embed_table, size_input, size_embedding):
        if size_embedding and (type(embed_table) is not tf.Variable):
            with tf.device("/cpu:0"):
                with tf.variable_scope(self.name, reuse=(self.__class__.num_Model > 0)):
                    embed_table = tf.get_variable(
                        "embedding", [size_input, size_embedding], dtype=tf.float32)

        return embed_table
