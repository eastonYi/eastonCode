import tensorflow as tf
import logging
from collections import namedtuple
from tensorflow.contrib.layers import fully_connected
from tf.contrib.seq2seq import CustomHelper
from tf.contrib.rnn import LSTMStateTuple
from tf.contrib.framework import nest

from dataProcessing.gradientTools import average_gradients, handle_gradients
from tfModels.tools import warmup_exponential_decay, choose_device, l2_penalty
from tfModels.layers import layer_normalize, conv1d, build_cell, cell_forward
from tfModels.CTCLoss import rna_loss
from tfModels.lstmModel import LSTM_Model


class RNA(LSTM_Model):
    num_Instances = 0

    def __init__(self, tensor_global_step, is_train, args, batch=None):
        super().__init__(tensor_global_step, is_train, args, batch=batch)

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):

        encoder_output = self.encode(name_gpu, tensors_input.feature_splits[id_gpu], self.is_train)

        distribution = self.decoder(name_gpu, tensors_input.input_step, self.is_train)

        with tf.name_scope("rna_loss"):
            batch_loss = rna_loss(self.decoder,
                                  encoder_output,
                                  tensors_input.label_splits[id_gpu],
                                  tensors_input.seq_fea_lens_splits[id_gpu],
                                  tensors_input.seq_label_lens_splits[id_gpu])
            loss = tf.reduce_mean(batch_loss)
            # Calculate L2 penalty.
            if self.args.lamda_l2:
                loss += l2_penalty(tf.trainable_variables())

            tf.get_variable_scope().reuse_variables()

            if self.is_train:
                with tf.name_scope("gradients"):
                    gradients = self.optimizer.compute_gradients(loss)
            else:
                gradients = None

            logging.info('\tbuild model on {} succesfully!'.format(name_gpu))
            return loss, gradients

    def build_input(self):
        tensors_input = namedtuple('tensors_input', 'feature_splits, label_splits, mask_splits, seq_len, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                batch_features = tf.placeholder(tf.float32, [None, None, self.args.data.dim_input], name='input_feature')
                batch_labels = tf.placeholder(tf.int32, [None, None], name='input_labels')
                batch_fea_lens = tf.placeholder(tf.int32, [None], name='input_fea_lens')
                batch_label_lens = tf.placeholder(tf.int32, [None], name='input_label_lens')
                self.list_pl = [batch_features, batch_labels, batch_fea_lens, batch_label_lens]
                # split input data alone batch axis to gpus
                tensors_input.feature_splits = tf.split(batch_features, self.num_gpus, name="feature_splits")
                tensors_input.label_splits = tf.split(batch_labels, self.num_gpus, name="label_splits")
                tensors_input.mask_splits = tf.split(batch_fea_lens, self.num_gpus, name="mask_splits")
                tensors_input.seq_len = tf.split(batch_label_lens, self.num_gpus, name="seq_len")
        tensors_input.shape_batch = tf.shape(batch_features)

        return tensors_input

    def encode(self, name_gpu, tensors_input, is_train):
        hidden_output = tensors_input

        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            for i in range(self.args.model.num_encoder_layers):
                # build one layer: build block, connect block
                single_cell = build_cell(1, self.is_train, self.args)
                hidden_output = self.lstm_forward(single_cell, hidden_output, i)
                hidden_output = conv1d(hidden_output,
                                       filters=self.args.model.num_hidden_units,
                                       kernel_size=1,
                                       name='wx_b'+str(i),
                                       activation=tf.nn.tanh)
                if self.args.model.use_layernorm:
                    hidden_output = layer_normalize(hidden_output, i)

            encoded_input = fully_connected(inputs=hidden_output,
                                            num_outputs=self.args.dim_output,
                                            activation_fn=tf.identity,
                                            scope='fully_connected')
        return encoded_input

    def decoder(self, name_gpu, input_step, state):
        """
        the decoder consumes the input feature, so it is not a standerd decoder,
        which functions like a language model.
        the input to the decoder is at time t for a given alignment z is the
        concatenation of input vector x_t and one-hot embedded label vector z_{t-1}
        , this is just another rnn-ctc network!
        so the RNA removes the frame-level-output independent assumption by just
        concatenate the current feature input with embedded previous output label!
        the output of previous label is not known at current step so we need to "decode".

        the encoder is a forward-type build, each layer just comsumes tue input
        seq, while decoder cannot! we need to forward the decode step by step!
        """
        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            self.cell = build_cell(self.args.model.num_decoder_layers, self.is_train, self.args)
            logits_step, state = cell_forward(self.cell, input_step, state)
            distribution = tf.nn.softmax(logits_step)

        return distribution

    def multi_forward(self, input_step, multi_state, size_batch):
        """

        multi_state: tensor with shape:
                     num_cells x num_layers x 2 x batch x num_hidden
        state_tensor
        """
        shape_distrib = [size_batch, self.args.size_vocab]

        def step(var, inputs):
            input, stacked_single_state = inputs

            single_state = self.tensor2state(stacked_single_state)
            logits, state = cell_forward(self.cell, input, single_state)
            distrib = tf.nn.softmax(logits)
            state = tf.stack(nest.flatten(state), 0)

            return distrib, state

        distribution, multi_state = tf.scan(step,
                                            [input_step, multi_state],
                                            (tf.zeros(shape_distrib), self.state2tensor(self.cell.zero_state())))

        return distribution, multi_state

    def zero_state(self, num_cells, size_batch):
        single_zero_state = self.cell.zero_state(size_batch, dtype=tf.float32)
        single_zero_state_tensor = self.state2tensor(single_zero_state, size_batch)
        multi_zero_state = tf.tile(tf.expand_dims(single_zero_state_tensor, 0),
                                   [num_cells, 1, 1, 1])

        return multi_zero_state

    def state2tensor(self, state, size_batch):
        """
        state: 3 x 2 x (5x14)
        (LSTMStateTuple(c=<tf.Tensor shape=(5, 14) dtype=float32>, h=<tf.Tensor shape=(5, 14) dtype=float32>),
         LSTMStateTuple(c=<tf.Tensor shape=(5, 14) dtype=float32>, h=<tf.Tensor shape=(5, 14) dtype=float32>),
         LSTMStateTuple(c=<tf.Tensor shape=(5, 14) dtype=float32>, h=<tf.Tensor shape=(5, 14) dtype=float32>))

        tensor: num_layers x 2 x batch x num_hidden
        """
        num_layers = self.args.num_decoder_layers
        num_hidden = self.args.num_hidden_units
        tensor = tf.reshape(tf.stack(nest.flatten(state), 0),
                           [num_layers, 2, size_batch, num_hidden])
        return tensor

    def tensor2state(self, tensor):
        """
        tensor: num_layers x 2 x batch x num_hidden
        """
        num_hidden = tf.shape(tensor)[2]
        size_batch = tf.shape(tensor)[3]
        list_tensor = tf.unstack(tf.reshape(-1, size_batch, num_hidden))
        single_state = nest.pack_sequence_as(self.cell.zero_state(), list_tensor)

        return single_state
