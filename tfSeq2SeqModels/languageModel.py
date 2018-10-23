import tensorflow as tf
import logging
import sys
from collections import namedtuple

from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.seq2seq import sequence_loss

from tfModels.tools import choose_device
from tfSeq2SeqModels.seq2seqModel import Seq2SeqModel

logging.basicself.args(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class LanguageModel(Seq2SeqModel):

    def __init__(self, tensor_global_step, decoder, is_train, args,
                 batch=None, embed_table_encoder=None, embed_table_decoder=None,
                 name='LanguageModel'):
        super().__init__(tensor_global_step, None, decoder, is_train, args,
                         batch, None, embed_table_decoder, name)

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        dropout = self.args.decoder.dropout
        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            inputs = tensors_input.feature_splits[id_gpu]
            if self.is_train:
                inputs = tf.nn.dropout(inputs, 1-dropout)
            hidden_output, state = self._build_rnn_graph(inputs)
            logits = fully_connected(
                inputs=hidden_output,
                num_outputs=self.args.dim_output)

            if self.is_train:
                loss = sequence_loss(
                    logits,
                    tensors_input.label_splits[id_gpu],
                    tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
                    average_across_timesteps=False,
                    average_across_batch=True)
                loss = tf.reduce_sum(loss)
                self._final_state = state

                with tf.name_scope("gradients"):
                    gradients = self.optimizer.compute_gradients(loss)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        if self.is_train:
            return loss, gradients
        else:
            return logits

    def build_infer_graph(self):
        # cerate input tensors in the cpu
        tensors_input = self.build_infer_input()

        with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model)):
            logits, len_logits = self.build_single_graph(
                id_gpu=0,
                name_gpu=self.list_gpu_devices[0],
                tensors_input=tensors_input)

        return

    def _build_rnn_graph_cudnn(self, inputs):
        """Build the inference graph using CUDNN cell."""
        inputs = tf.transpose(inputs, [1, 0, 2])
        self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=self.args.num_layers,
            num_units=self.args.hidden_size,
            input_size=self.args.hidden_size,
            dropout=1 - self.args.keep_prob if self.is_train else 0)
        params_size_t = self._cell.params_size()
        self._rnn_params = tf.get_variable(
            "lstm_params",
            initializer=tf.random_uniform(
                [params_size_t], -self.args.init_scale, self.args.init_scale),
            validate_shape=False)
        c = tf.zeros([self.args.num_layers, self.batch_size, self.args.hidden_size],
                     tf.float32)
        h = tf.zeros([self.args.num_layers, self.batch_size, self.args.hidden_size],
                     tf.float32)
        self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
        outputs, h, c = self._cell(inputs, h, c, self._rnn_params, self.is_train)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.reshape(outputs, [-1, self.args.hidden_size])

        return outputs, (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

    def _get_lstm_cell(self):
        if self.args.rnn_mode == 'BASIC':
            return tf.contrib.rnn.BasicLSTMCell(
                self.args.hidden_size, forget_bias=0.0, state_is_tuple=True,
                reuse=not self.is_train)
        if self.args.rnn_mode == 'BLOCK':
            return tf.contrib.rnn.LSTMBlockCell(
                self.args.hidden_size, forget_bias=0.0)
        raise ValueError("rnn_mode %s not supported" % self.args.rnn_mode)

    def _build_rnn_graph_lstm(self, inputs):
        """Build the inference graph using canonical LSTM cells."""
        def make_cell():
            cell = self._get_lstm_cell()
            if self.is_train and self.args.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=self.args.keep_prob)
            return cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(self.args.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(self.args.batch_size, tf.float32)
        state = self._initial_state
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, self.args.hidden_size])

        return output, state

    def build_idx_input(self):
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
                batch_features = tf.nn.embedding_lookup(self.embed_table_decoder, batch_src)
                tensors_input.feature_splits = tf.split(batch_features, self.num_gpus, name="feature_splits")
                tensors_input.label_splits = tf.split(batch_ref, self.num_gpus, name="label_splits")
                tensors_input.len_fea_splits = tf.split(batch_src_lens, self.num_gpus, name="len_fea_splits")
                tensors_input.len_label_splits = tf.split(batch_ref_lens, self.num_gpus, name="len_label_splits")
        tensors_input.shape_batch = tf.shape(batch_features)

        return tensors_input

    def build_infer_idx_input(self):
        """
        use the decoder table
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, len_fea_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                batch_src = tf.placeholder(tf.int32, [None, None], name='input_src')
                batch_src_lens = tf.placeholder(tf.int32, [None], name='input_src_lens')
                self.list_pl = [batch_src, batch_src_lens]
                # split input data alone batch axis to gpus
                batch_features = tf.nn.embedding_lookup(self.embed_table_decoder, batch_src)
                tensors_input.feature_splits = tf.split(batch_features, self.num_gpus, name="feature_splits")
                tensors_input.len_fea_splits = tf.split(batch_src_lens, self.num_gpus, name="len_fea_splits")

        tensors_input.shape_batch = tf.shape(batch_features)

        return tensors_input

    def get_embedding(self, embed_table):
        if embed_table is None:
            with tf.device("/cpu:0"):
                embed_table = tf.get_variable(
                    "embedding", [self.args.dim_output, self.size_embeding], dtype=tf.float32)

        return embed_table
