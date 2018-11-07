import tensorflow as tf
import logging
import sys
from collections import namedtuple

from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.seq2seq import sequence_loss
from tfTools.gradientTools import average_gradients, handle_gradients

from tfModels.tools import choose_device
from tfModels.layers import build_cell
from tfSeq2SeqModels.seq2seqModel import Seq2SeqModel

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class LanguageModel(Seq2SeqModel):

    def __init__(self, tensor_global_step, encoder, decoder, is_train, args,
                 embed_table_encoder=None, embed_table_decoder=None,
                 name='LanguageModel'):
        self.num_cell_units = args.model.decoder.num_cell_units
        self.dropout = args.model.decoder.dropout
        self.keep_prob = 1 - args.model.decoder.dropout
        self.cell_type = args.model.decoder.cell_type
        self.num_layers = args.model.decoder.num_layers
        self.init_scale = args.model.decoder.init_scale
        self.rnn_mode = args.model.decoder.rnn_mode
        self.size_embedding = args.model.decoder.size_embedding
        self.global_step = tensor_global_step
        self.name = name
        self.initializer = tf.random_uniform_initializer(-self.init_scale, self.init_scale)

        super().__init__(tensor_global_step, None, decoder, is_train, args,
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
            if self.is_train:
                inputs = tf.nn.dropout(inputs, self.keep_prob)

            inputs.set_shape([None, None, self.size_embedding])
            # cell = build_cell(
            #     num_units=self.num_cell_units,
            #     num_layers=self.num_layers,
            #     is_train=self.is_train,
            #     dropout=self.dropout,
            #     cell_type=self.cell_type)

            self.cell = self.make_multi_cell(self.num_layers)

            hidden_output, _ = tf.nn.dynamic_rnn(
                cell=self.cell,
                inputs=inputs,
                sequence_length=tensors_input.len_fea_splits[id_gpu],
                dtype=tf.float32)

            logits = hidden_output

            # logits = fully_connected(
            #     inputs=hidden_output,
            #     num_outputs=self.args.dim_output,
            #     scope='fc')

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tensors_input.label_splits[id_gpu],
                logits=logits)
            loss *= tf.cast(tf.sequence_mask(tensors_input.len_label_splits[id_gpu]), tf.float32)

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

        with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model), initializer=self.initializer):
            loss = self.build_single_graph(
                id_gpu=0,
                name_gpu=self.list_gpu_devices[0],
                tensors_input=tensors_input)
            loss = tf.reduce_sum(loss)

        self.__class__.num_Instances += 1
        logging.info("built {} {} instance(s).".format(self.__class__.num_Instances, self.__class__.__name__))

        return loss, tensors_input.shape_batch

    def _get_lstm_cell(self):
        if self.args.model.decoder.rnn_mode == 'BASIC':
            return tf.contrib.rnn.BasicLSTMCell(
                self.num_cell_units, forget_bias=0.0, state_is_tuple=True,
                reuse=not self.is_train)
        if self.args.model.decoder.rnn_mode == 'BLOCK':
            return tf.contrib.rnn.LSTMBlockCell(
                self.num_cell_units, forget_bias=0.0)
        if self.args.model.decoder.rnn_mode == 'CUDNN':
            return tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.num_cell_units)
        raise ValueError("rnn_mode %s not supported" % self.rnn_mode)

    def make_cell(self):
        cell = self._get_lstm_cell()
        if self.is_train and self.keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(
                cell, output_keep_prob=self.keep_prob)
        return cell

    def make_multi_cell(self, num_layers):
        list_cells = [self.make_cell() for _ in range(num_layers-1)]
        cell_proj = tf.contrib.rnn.OutputProjectionWrapper(
            cell=self.make_cell(),
            output_size=self.args.dim_output)
        list_cells.append(cell_proj)
        multi_cell = tf.contrib.rnn.MultiRNNCell(list_cells, state_is_tuple=True)

        return multi_cell

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
                embed_table = self.embed_table_encoder if self.embed_table_encoder else self.embed_table_decoder
                batch_features = tf.nn.embedding_lookup(embed_table, batch_src)
                tensors_input.feature_splits = tf.split(batch_features, self.num_gpus, name="feature_splits")
                tensors_input.label_splits = tf.split(batch_ref, self.num_gpus, name="label_splits")
                tensors_input.len_fea_splits = tf.split(batch_src_lens, self.num_gpus, name="len_fea_splits")
                tensors_input.len_label_splits = tf.split(batch_ref_lens, self.num_gpus, name="len_label_splits")
        tensors_input.shape_batch = tf.shape(batch_features)

        return tensors_input

    def zero_state(self, batch_size, dtype=tf.float32):
        return self.cell.zero_state(batch_size, dtype=tf.float32)

    def forward(self, input, state):
        output, state = tf.contrib.legacy_seq2seq.rnn_decoder(
            decoder_inputs=[input],
            initial_state=state,
            cell=self.cell)

        return output, state
