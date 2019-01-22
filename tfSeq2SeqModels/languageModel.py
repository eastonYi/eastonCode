import tensorflow as tf
import logging
import sys
from collections import namedtuple
from tensorflow.python.util import nest

from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.seq2seq import sequence_loss
from tfTools.gradientTools import average_gradients, handle_gradients
from tfModels.regularization import confidence_penalty

from tfModels.tools import choose_device
from tfModels.layers import build_cell
from tfSeq2SeqModels.seq2seqModel import Seq2SeqModel


logging.basicConfig(level=logging.DEBUG, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class LanguageModel(Seq2SeqModel):
    """
    Intrinsic Evaluation: Perplexity;
    Extrinsic (task-based) Evaluation: Word Error Rate
    """

    def __init__(self, tensor_global_step, is_train, args, embed_table_decoder=None,
                 name='LanguageModel'):
        self.type = args.model.decoder.type
        self.num_cell_units = args.model.decoder.num_cell_units
        # self.dropout = args.model.decoder.dropout
        # self.keep_prob = 1 - args.model.decoder.dropout
        self.cell_type = args.model.decoder.cell_type
        self.num_layers = args.model.decoder.num_layers
        self.init_scale = args.model.decoder.init_scale
        self.rnn_mode = args.model.decoder.rnn_mode
        self.size_embedding = args.model.decoder.size_embedding
        self.global_step = tensor_global_step
        self.name = name
        self.initializer = tf.random_uniform_initializer(-self.init_scale, self.init_scale)

        super().__init__(tensor_global_step, None, None, is_train, args,
                         batch=None,
                         embed_table_encoder=None,
                         embed_table_decoder=embed_table_decoder,
                         name=name)

    def build_graph(self):
        # cerate input tensors in the cpu

        tensors_input = self.build_input()
        # create optimizer
        self.optimizer = self.build_optimizer()

        loss_step = []
        tower_grads = []
        # the outer scope is necessary for the where the reuse scope need to be limited whthin
        # or reuse=tf.get_variable_scope().reuse
        for id_gpu, name_gpu in enumerate(self.list_gpu_devices):
            with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model), initializer=self.initializer):
                loss, gradients = self.build_single_graph(id_gpu, name_gpu, tensors_input)
                loss_step.append(loss)
                tower_grads.append(gradients)

        loss = tf.reduce_sum(loss_step)
        # merge gradients, update current model
        with tf.device(self.center_device):
            # computation relevant to gradient
            averaged_grads = average_gradients(tower_grads)
            handled_grads = handle_gradients(averaged_grads, self.args)
            op_optimize = self.optimizer.apply_gradients(handled_grads, self.global_step)

        self.__class__.num_Instances += 1
        logging.info("built {} {} instance(s).".format(self.__class__.num_Instances, self.__class__.__name__))

        return loss, tensors_input.shape_batch, op_optimize

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):

            inputs = tensors_input.feature_splits[id_gpu]
            len_inputs = tensors_input.len_fea_splits[id_gpu]
            inputs.set_shape([None, None, self.size_embedding])

            if self.type == 'LSTM':
                from tfSeq2SeqModels.decoders.lm_decoder import LM_Decoder
                self.decoder = LM_Decoder(self.args, self.is_train)
                hidden_output, _ = self.decoder(inputs, len_inputs)
            elif self.type == 'SelfAttention':
                from tfSeq2SeqModels.decoders.self_attention_lm_decoder import SelfAttentionDecoder
                self.decoder= SelfAttentionDecoder(self.args, self.is_train, self.embed_table_decoder)
                # from tfSeq2SeqModels.decoders.self_attention_lm_decoder_lh import SelfAttentionDecoder_lh
                # decoder = SelfAttentionDecoder_lh(self.args, self.is_train, self.embed_table_decoder)
                hidden_output = self.decoder(inputs, len_inputs)
            # self.cell = self.make_multi_cell(self.num_layers)
            #
            # hidden_output, _ = tf.nn.dynamic_rnn(
            #     cell=self.cell,
            #     inputs=inputs,
            #     sequence_length=tensors_input.len_fea_splits[id_gpu],
            #     dtype=tf.float32)

            logits = hidden_output
            len_logits = tensors_input.len_label_splits[id_gpu]

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tensors_input.label_splits[id_gpu],
                logits=logits)
            loss *= tf.sequence_mask(
                tensors_input.len_label_splits[id_gpu],
                maxlen=tf.shape(logits)[1],
                dtype=logits.dtype)
            if self.args.model.confidence_penalty:
                ls_loss = self.args.model.confidence_penalty * confidence_penalty(logits, len_logits)
                ls_loss = tf.reduce_mean(ls_loss)
                loss += ls_loss

            # from tfModels.tensor2tensor.common_layers import padded_cross_entropy, weights_nonzero
            #
            # mask = tf.sequence_mask(
            #     tensors_input.len_label_splits[id_gpu],
            #     maxlen=tf.shape(logits)[1],
            #     dtype=logits.dtype)
            # batch_mask = tf.tile(tf.expand_dims(mask, -1), [1, 1, tf.shape(logits)[-1]])
            # loss, _ = padded_cross_entropy(
            #     logits* batch_mask,
            #     tensors_input.label_splits[id_gpu],
            #     0.0,
            #     weights_fn=weights_nonzero,
            #     reduce_sum=False)
            # loss = tf.Print(loss, [weight_sum], message='weight_sum', summarize=1000)

            if self.is_train:
                with tf.name_scope("gradients"):
                    gradients = self.optimizer.compute_gradients(loss)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        if self.is_train:
            return loss, gradients
        else:
            return loss

    def build_infer_graph(self):
        # cerate input tensors in the cpu
        tensors_input = self.build_infer_idx_input()

        # with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model)):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            loss = self.build_single_graph(
                id_gpu=0,
                name_gpu=self.list_gpu_devices[0],
                tensors_input=tensors_input)
            loss = tf.reduce_sum(loss)

        self.__class__.num_Instances += 1
        logging.info("built {} {} instance(s).".format(self.__class__.num_Instances, self.__class__.__name__))

        return loss, tensors_input.shape_batch

    def build_infer_idx_input(self):
        """
        used for token-input tasks such as nmt when the `self.embed_table_encoder` is given
        for the token inputs are easy to fentch form disk, there is no need to
        use tfdata.
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, label_splits, len_fea_splits, len_label_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                batch_src = tf.placeholder(tf.int32, [None, None], name='input_src')
                batch_ref = tf.placeholder(tf.int32, [None, None], name='input_ref')
                batch_src_lens = tf.placeholder(tf.int32, [None], name='input_src_lens')
                batch_ref_lens = tf.placeholder(tf.int32, [None], name='input_ref_lens')
                self.list_pl = [batch_src, batch_ref, batch_src_lens, batch_ref_lens]
                # split input data alone batch axis to gpus
                self.embed_table = self.embed_table_decoder
                batch_features = tf.nn.embedding_lookup(self.embed_table, batch_src)
                tensors_input.feature_splits = tf.split(batch_features, self.num_gpus, name="feature_splits")
                tensors_input.label_splits = tf.split(batch_ref, self.num_gpus, name="label_splits")
                tensors_input.len_fea_splits = tf.split(batch_src_lens, self.num_gpus, name="len_fea_splits")
                tensors_input.len_label_splits = tf.split(batch_ref_lens, self.num_gpus, name="len_label_splits")
        tensors_input.shape_batch = tf.shape(batch_features)

        return tensors_input

    def sample(self, token_init=None, state_init=None, max_length=50):
        with tf.variable_scope(self.decoder.scope, reuse=True):
            return self.decoder.sample(token_init, state_init, max_length)

    def score(self, decoder_input, len_seqs):
        """
        the input to the score is <sos> + sent + <eos>
        the input to the model is `<sos> + sent` and the score gather indices is `sent + <eos>`
        """
        with tf.variable_scope(self.decoder.scope, reuse=True):
            return self.decoder.score(decoder_input, len_seqs)

    def forward(self, preds, cache, stop_gradient=False):
        with tf.variable_scope(self.decoder.scope, reuse=True):

            return self.decoder.forward(preds, cache, stop_gradient=False)
