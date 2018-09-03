import tensorflow as tf
import logging
import sys

from tensorflow.contrib.layers import fully_connected

from tfModels.tools import choose_device, l2_penalty
from tfModels.layers import layer_normalize, build_cell, cell_forward
from tfModels.lstmCTCModel import LSTM_CTC_Model
from tfModels.CTCLoss import ctc_loss
from tfTools.tfTools import dense_sequence_to_sparse


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class DeepSpeech(LSTM_CTC_Model):
    num_Instances = 0

    def __init__(self, tensor_global_step, is_train, args, batch=None):
        super().__init__(tensor_global_step, is_train, args, batch=batch)

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        # build model in one device
        dim_input = self.args.data.dim_input
        n_hidden = self.args.model.n_hidden
        relu_clip = self.args.model.relu_clip
        dropout = self.args.model.n_dropout
        num_cell_project = self.args.model.num_cell_project

        batch_x = tensors_input.feature_splits[id_gpu]
        size_batch = tf.shape(batch_x)[0]

        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            # batch_x = tf.reshape(batch_x, [-1, dim_input])

            # fc1
            output_fc1 = fully_connected(
                inputs=batch_x,
                num_outputs=n_hidden[1],
                activation_fn=tf.identity,
                scope='fc1')
            layer_1 = tf.minimum(tf.nn.relu(output_fc1), relu_clip)
            layer_1 = tf.nn.dropout(layer_1, (1.0 - dropout[1]))

            # fc2
            output_fc2 = fully_connected(inputs=layer_1,
                                         num_outputs=n_hidden[2],
                                         activation_fn=tf.identity,
                                         scope='fc2')
            layer_2 = tf.minimum(tf.nn.relu(output_fc2), relu_clip)
            layer_2 = tf.nn.dropout(layer_2, (1.0 - dropout[2]))

            # fc3
            output_fc3 = fully_connected(inputs=layer_2,
                                         num_outputs=n_hidden[3],
                                         activation_fn=tf.identity,
                                         scope='fc3')
            layer_3 = tf.minimum(tf.nn.relu(output_fc3), relu_clip)
            layer_3 = tf.nn.dropout(layer_3, (1.0 - dropout[3]))

            # hidden_output = tf.reshape(layer_3, [size_batch, -1, n_hidden[3]])
            hidden_output = layer_3
            for i in range(self.args.model.num_lstm_layers):
                # build one layer: build block, connect block
                single_cell = build_cell(n_hidden[4], 1, self.is_train, self.args)
                hidden_output, _ = cell_forward(single_cell, hidden_output, i)
                hidden_output = fully_connected(inputs=hidden_output,
                                                num_outputs=num_cell_project,
                                                activation_fn=tf.nn.tanh,
                                                scope='wx_b'+str(i))
                if self.args.model.use_layernorm:
                    hidden_output = layer_normalize(hidden_output, i)

            # layer_4 = tf.transpose(hidden_output, [1, 0, 2])
            # layer_4 = tf.reshape(layer_4, [-1, num_cell_project])
            layer_4 = hidden_output

            # fc5
            output_fc5 = fully_connected(inputs=layer_4,
                                         num_outputs=n_hidden[5],
                                         activation_fn=tf.identity,
                                         scope='fc5')
            layer_5 = tf.minimum(tf.nn.relu(output_fc5), relu_clip)
            layer_5 = tf.nn.dropout(layer_5, (1.0 - dropout[5]))

            # fc6
            output_fc6 = fully_connected(inputs=layer_5,
                                         num_outputs=n_hidden[6],
                                         activation_fn=tf.identity,
                                         scope='fc6')
            layer_6 = tf.nn.relu(output_fc6)

            self.logits = layer_6
            # self.logits = tf.transpose(
            #     tf.reshape(layer_6, [-1, size_batch, n_hidden[6]]),
            #     [1, 0, 2])

            with tf.name_scope("ctc_loss"):
                if self.args.model.ctc_loss_type == 'MyCTC':
                    distribution = tf.nn.softmax(self.logits)
                    batch_loss = ctc_loss(distribution,
                                          tensors_input.label_splits[id_gpu],
                                          tensors_input.len_fea_splits[id_gpu],
                                          tensors_input.len_label_splits[id_gpu])
                    loss = tf.reduce_mean(batch_loss)

                elif self.args.model.ctc_loss_type == 'tfCTC':
                    sparse_targets = dense_sequence_to_sparse(
                        tensors_input.label_splits[id_gpu],
                        tensors_input.len_label_splits[id_gpu])
                    loss = tf.reduce_mean(tf.nn.ctc_loss(
                        sparse_targets,
                        self.logits,
                        sequence_length=tensors_input.len_fea_splits[id_gpu],
                        time_major=False))

                # loss /= tf.cast(
                #     tf.reduce_sum(tensors_input.len_fea_splits[id_gpu]),
                #     dtype=tf.float32)

            # Calculate L2 penalty.
            if self.args.lamda_l2:
                loss += self.args.lamda_l2 * l2_penalty(tf.trainable_variables())

            if self.is_train:
                with tf.name_scope("gradients"):
                    gradients = self.optimizer.compute_gradients(loss)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        return loss, gradients if self.is_train else self.logits


    # def build_infer_graph(self):
    #     # cerate input tensors in the cpu
    #     self.tensors_input = self.build_input()
    #
    #     with tf.variable_scope('model', reuse=bool(self.__class__.num_Model)):
    #         loss, self.logits = self.build_single_graph(
    #             id_gpu=0,
    #             name_gpu=self.list_gpu_devices[0],
    #             tensors_input=self.tensors_input)
    #
    #         distribution = tf.nn.softmax(self.logits)
    #
    #     return loss, self.tensors_input.shape_batch, distribution

    # def build_decoder_graph(self):
    #     logits_timeMajor = tf.transpose(self.logits, [1, 0, 2])
    #     with tf.name_scope('ctc_decoder'):
    #         result_sparse, _ = tf.nn.ctc_beam_search_decoder(
    #             logits_timeMajor,
    #             self.tensors_input.len_fea_splits[0],
    #             beam_width=self.args.beam_size)
    #         result = tf.sparse_tensor_to_dense(result_sparse[0])
    #
    #     return result
