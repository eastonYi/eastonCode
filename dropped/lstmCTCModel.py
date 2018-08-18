import tensorflow as tf
import logging
import sys

from tensorflow.contrib.layers import fully_connected

from tfModels.tools import choose_device, l2_penalty
from tfModels.layers import layer_normalize, build_cell, cell_forward
from tfModels.lstmModel import LSTM_Model
from tfModels.CTCLoss import ctc_loss
from tfTools.tfTools import dense_sequence_to_sparse

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class LSTM_CTC_Model(LSTM_Model):

    def __init__(self, tensor_global_step, is_train, args, batch=None):
        super().__init__(tensor_global_step, is_train, args, batch=batch)
        # if not is_train:
        #     self.decoded = self.build_decoder_graph()
        #     self.list_run.append(self.decoded)

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        # build model in one device
        num_cell_units = self.args.model.num_cell_units
        num_cell_project = self.args.model.num_cell_project
        cell_type = self.args.model.cell_type
        dropout = self.args.model.dropout
        forget_bias = self.args.model.forget_bias
        use_residual = self.args.model.use_residual

        hidden_output = tensors_input.feature_splits[id_gpu]

        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            for i in range(self.args.model.num_lstm_layers):
                # build one layer: build block, connect block
                single_cell = build_cell(
                    num_units=num_cell_units,
                    num_layers=1,
                    is_train=self.is_train,
                    cell_type=cell_type,
                    dropout=dropout,
                    forget_bias=forget_bias,
                    use_residual=use_residual)
                hidden_output, _ = cell_forward(
                    cell=single_cell,
                    inputs=hidden_output,
                    index_layer=i)
                hidden_output = fully_connected(
                    inputs=hidden_output,
                    num_outputs=num_cell_project,
                    activation_fn=tf.nn.tanh,
                    scope='wx_b'+str(i))

                if self.args.model.use_layernorm:
                    hidden_output = layer_normalize(hidden_output, i)

            logits = fully_connected(inputs=hidden_output,
                                     num_outputs=self.args.dim_output,
                                     activation_fn=tf.identity,
                                     scope='fully_connected')

            # CTC loss
            with tf.name_scope("ctc_loss"):
                if self.args.model.ctc_loss_type == 'MyCTC':
                    distribution = tf.nn.softmax(logits)
                    batch_loss = ctc_loss(distribution,
                        tensors_input.label_splits[id_gpu],
                        tensors_input.len_fea_splits[id_gpu],
                        tensors_input.len_label_splits[id_gpu],
                        blank=self.args.ID_BLANK)
                    loss = tf.reduce_mean(batch_loss)

                elif self.args.model.ctc_loss_type == 'tfCTC':
                    sparse_targets = dense_sequence_to_sparse(
                        tensors_input.label_splits[id_gpu],
                        tensors_input.len_label_splits[id_gpu])
                    loss = tf.reduce_mean(tf.nn.ctc_loss(
                        sparse_targets,
                        logits,
                        sequence_length=tensors_input.len_fea_splits[id_gpu],
                        time_major=False))

            if self.is_train:
                with tf.name_scope("gradients"):
                    gradients = self.optimizer.compute_gradients(loss)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        return loss, gradients if self.is_train else logits

    def build_infer_graph(self):
        # cerate input tensors in the cpu
        self.tensors_input = self.build_input()

        # in the
        with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model)):
            loss, logits = self.build_single_graph(
                id_gpu=0,
                name_gpu=self.list_gpu_devices[0],
                tensors_input=self.tensors_input)

            distribution = tf.nn.softmax(logits)

        return loss, self.tensors_input.shape_batch, distribution

    def build_decoder_graph(self):
        # logits_timeMajor = tf.transpose(logits, [1, 0, 2])
        # with tf.name_scope('ctc_decoder'):
        #     result_sparse, _ = tf.nn.ctc_beam_search_decoder(
        #         logits_timeMajor,
        #         self.tensors_input.len_fea_splits[0],
        #         beam_width=self.args.beam_size)
        #     result = tf.sparse_tensor_to_dense(result_sparse[0])
        #
        # return result
        return tf.constant(1)
