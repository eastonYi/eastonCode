import tensorflow as tf
import logging
import sys

from tensorflow.contrib.layers import fully_connected

from tfModels.tools import choose_device, l2_penalty
from tfModels.layers import layer_normalize, build_cell, cell_forward, conv_layer
from tfModels.lstmCTCModel2 import LSTM_CTC_Model
from tfTools.tfTools import dense_sequence_to_sparse
from tfTools.tfAudioTools import down_sample

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class DeepSpeech2(LSTM_CTC_Model):
    num_Instances = 0

    def __init__(self, tensor_global_step, mode, args, batch=None, name='DS2'):
        super().__init__(tensor_global_step, mode, args, batch=batch, name=name)

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        # build model in one device
        num_cell_hidden = self.args.model.num_cell_hidden
        num_cell_project = self.args.model.num_cell_project
        num_filter = self.args.model.num_filter

        batch_x = tensors_input.feature_splits[id_gpu]
        size_batch = tf.shape(batch_x)[0]

        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            conv_output = tf.expand_dims(batch_x, -1)
            # conv_output = tf.Print(conv_output, [tf.shape(conv_output)], message='conv before shape: ', summarize=100)
            for i in range(self.args.model.num_conv_layers):
                conv_output = conv_layer(
                    conv_output,
                    num_filter,
                    kernel=(32,8),
                    stride=(1,1),
                    scope='conv_'+str(i))

                conv_output = down_sample(conv_output, rate=2)
                # conv_output = tf.Print(conv_output, [tf.shape(conv_output)], message='conv during shape: ', summarize=100)
                # conv_output = tf.layers.max_pooling2d(
                #     inputs=conv_output,
                #     pool_size=[1,1],
                #     strides=(1,1),
                #     padding="same")
            # conv_output = tf.Print(conv_output, [tf.shape(conv_output)], message='conv after shape: ', summarize=100)

            hidden_output = tf.reshape(
                conv_output,
                [size_batch, -1, int(self.args.data.dim_input*num_filter)])
            # hidden_output = tf.Print(hidden_output, [tf.shape(hidden_output)], message='lstm input shape: ', summarize=100)
            for i in range(self.args.model.num_lstm_layers):
                # build one layer: build block, connect block
                single_cell = build_cell(num_cell_hidden, 1, self.is_train, self.args)
                hidden_output, _ = cell_forward(single_cell, hidden_output, i)
                hidden_output = fully_connected(inputs=hidden_output,
                                                num_outputs=num_cell_project,
                                                activation_fn=tf.nn.tanh,
                                                scope='wx_b'+str(i))

                if self.args.model.use_layernorm:
                    hidden_output = layer_normalize(hidden_output, i)

            self.logits = fully_connected(inputs=hidden_output,
                                     num_outputs=self.args.dim_output,
                                     activation_fn=tf.identity,
                                     scope='fully_connected')

            # CTC loss
            with tf.name_scope("ctc_loss"):
                if self.args.model.ctc_loss_type == 'MyCTC':
                    distribution = tf.nn.softmax(self.logits)
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
                        self.logits,
                        sequence_length=tensors_input.len_fea_splits[id_gpu],
                        time_major=False))

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
