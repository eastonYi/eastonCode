import tensorflow as tf
import logging
import sys

from tensorflow.contrib.layers import fully_connected

from tfModels.tools import choose_device
from tfModels.lstmCTCModel2 import LSTM_CTC_Model
from tfSeq2SeqModels.encoders.cnn import CNN as Encoder_CNN


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class DeepSpeech2(LSTM_CTC_Model):

    def __init__(self, tensor_global_step, mode, args, batch=None, name='DeepSpeech2'):
        super().__init__(tensor_global_step, mode, args, batch, name)

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        num_class = self.args.dim_output + 1
        Encoder_LSTM = self.args.model.encoder.type

        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            encoder_cnn = Encoder_CNN(
                is_train=self.is_train,
                args=self.args)
            encoder_input = encoder_cnn.build_input(
                id_gpu=id_gpu,
                tensors_input=tensors_input)
            cnn_output, len_cnn_output = encoder_cnn(encoder_input)

            encoder_lstm = Encoder_LSTM(
                is_train=self.is_train,
                args=self.args)
            hidden_output, len_logits = encoder_lstm.encode(
                features=cnn_output,
                len_feas=len_cnn_output)

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
