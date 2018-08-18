import tensorflow as tf
import logging
import sys

from tensorflow.contrib.layers import fully_connected

from tfModels.tools import choose_device
from tfModels.lstmModel import LSTM_Model
from tfTools.tfTools import dense_sequence_to_sparse

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class LSTM_CTC_Model(LSTM_Model):

    def __init__(self, tensor_global_step, mode, args, batch=None, name='tf_CTC_Model'):
        self.name = name
        # Initialize some parameters
        self.mode = mode
        self.is_train = (mode == 'train')
        self.num_gpus = args.num_gpus if self.is_train else 1
        self.list_gpu_devices = args.list_gpus
        self.center_device = "/cpu:0"
        self.learning_rate = None
        self.args = args
        self.batch = batch
        self.build_input = self.build_tf_input if batch else self.build_pl_input

        self.list_pl = None

        self.global_step = tensor_global_step

        if mode == 'train':
            self.list_run = list(self.build_graph())
        elif mode == 'dev':
            self.list_run = list(self.build_dev_graph())
        elif mode == 'infer':
            self.list_run = list(self.build_infer_graph())
        else:
            raise "unknow mode!"

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        num_class = self.args.dim_output + 1
        Encoder = self.args.model.encoder.type

        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            # create encoder obj
            encoder = Encoder(
                is_train=self.is_train,
                args=self.args)
            encoder_input = encoder.build_input(
                id_gpu=id_gpu,
                tensors_input=tensors_input)

            # using encoder to encode the inout sequence
            hidden_output, len_logits = encoder(encoder_input)
            logits = fully_connected(
                inputs=hidden_output,
                num_outputs=num_class)

            # CTC loss
            loss = self.ctc_loss(
                logits=logits,
                len_logits=len_logits,
                labels=tensors_input.label_splits[id_gpu],
                len_labels=tensors_input.len_label_splits[id_gpu])

            if self.is_train:
                with tf.name_scope("gradients"):
                    gradients = self.optimizer.compute_gradients(loss)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        if self.is_train:
            return loss, gradients
        else:
            return loss, logits, len_logits

    def ctc_loss(self, logits, len_logits, labels, len_labels):
        with tf.name_scope("ctc_loss"):
            labels_sparse = dense_sequence_to_sparse(
                labels,
                len_labels)
            ctc_loss_batch = tf.nn.ctc_loss(
                labels_sparse,
                logits,
                sequence_length=len_logits,
                ignore_longer_outputs_than_inputs=True,
                time_major=False)
            loss = tf.reduce_mean(ctc_loss_batch) # utter-level ctc loss

        return loss

    def build_infer_graph(self):
        # cerate input tensors in the cpu
        beam_size = self.args.beam_size
        tensors_input = self.build_input()

        with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model)):
            loss, logits, len_logits = self.build_single_graph(
                id_gpu=0,
                name_gpu=self.list_gpu_devices[0],
                tensors_input=tensors_input)
            logits_timeMajor = tf.transpose(logits, [1, 0, 2])

            if beam_size == 1:
                decoded = tf.to_int32(tf.nn.ctc_greedy_decoder(
                    logits_timeMajor,
                    len_logits)[0][0])
            else:
                decoded = tf.to_int32(tf.nn.ctc_beam_search_decoder(
                    logits_timeMajor,
                    len_logits,
                    beam_width=beam_size)[0][0])

            decoded = tf.sparse_to_dense(
                sparse_indices=decoded.indices,
                output_shape=decoded.dense_shape,
                sparse_values=decoded.values,
                default_value=-1,
                validate_indices=True)
            distribution = tf.nn.softmax(logits)

        return loss, tensors_input.shape_batch, distribution, decoded

    def build_dev_graph(self):
        # just need loss
        tensors_input = self.build_input()

        with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model)):
            loss, logits, len_logits = self.build_single_graph(
                id_gpu=0,
                name_gpu=self.list_gpu_devices[0],
                tensors_input=tensors_input)

        return loss, tensors_input.shape_batch
