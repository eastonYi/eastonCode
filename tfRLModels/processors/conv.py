import tensorflow as tf
import numpy as np

from tfModels.layers import residual, conv_lstm
from tfModels.tensor2tensor.common_layers import layer_norm
from processor import Processor


class CONV_Processor(Processor):
    def __init__(self, args, name):
        self.num_cell_units = args.model.processor.num_cell_units
        self.num_filters = args.model.processor.num_filters
        self.num_layers = args.model.processor.num_layers
        self.size_feat = args.data.dim_input
        name = name if name else 'conv_processor'
        super().__init__(args, name)

    def process(self, inputs, len_inputs):
        size_batch  = tf.shape(inputs)[0]
        size_length = tf.shape(inputs)[1]
        size_feat = int(self.size_feat/3)
        x = tf.reshape(inputs, [size_batch, size_length, size_feat, 3])

        with tf.variable_scope(self.name):
            x = self.normal_conv(
                inputs=x,
                filter_num=self.num_filters,
                kernel=(3,3),
                stride=(2,2),
                padding='SAME',
                use_relu=True,
                name="conv1",
                w_initializer=None,
                norm_type='layer')
            x = self.normal_conv(
                inputs=x,
                filter_num=self.num_filters,
                kernel=(3,3),
                stride=(2,2),
                padding='SAME',
                use_relu=True,
                name="conv2",
                w_initializer=None,
                norm_type='layer')
            x = self.normal_conv(
                inputs=x,
                filter_num=self.num_filters,
                kernel=(3,3),
                stride=(2,2),
                padding='SAME',
                use_relu=True,
                name="conv3",
                w_initializer=None,
                norm_type='layer')
            x = conv_lstm(
                inputs=x,
                kernel_size=(3,3),
                filters=self.num_filters)

        size_feat = int(np.ceil(self.size_feat/8)) * self.num_filters
        size_length = tf.cast(tf.ceil(tf.cast(size_length,tf.float32)/8), tf.int32)
        len_frames = tf.cast(tf.ceil(tf.cast(len_inputs,tf.float32)/8), tf.int32)
        frames = tf.reshape(x, [size_batch, size_length, size_feat])

        return frames, len_frames

    @staticmethod
    def normal_conv(inputs, filter_num, kernel, stride, padding, use_relu, name,
                    w_initializer=None, norm_type="batch"):
        with tf.variable_scope(name):
            net = tf.layers.conv2d(inputs, filter_num, kernel, stride, padding,
                               kernel_initializer=w_initializer, name="conv")
            if norm_type == "batch":
                net = tf.layers.batch_normalization(net, name="bn")
            elif norm_type == "layer":
                net = layer_norm(net)
            else:
                net = net
            output = tf.nn.relu(net) if use_relu else net

        return output

    @staticmethod
    def blstm(hidden_output, len_feas, num_cell_units, name, dropout=0.0, use_residual=False):
        num_cell_units /= 2

        with tf.variable_scope(name):
            f_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_cell_units)
            b_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_cell_units)

            x, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=f_cell,
                cell_bw=b_cell,
                inputs=hidden_output,
                dtype=tf.float32,
                time_major=False,
                sequence_length=len_feas)
            x = tf.concat(x, 2)

            if use_residual:
                x = residual(hidden_output, x, dropout)

        return x

    def pooling(self, x, len_sequence, type, name):

        x = tf.expand_dims(x, axis=2)
        x = self.normal_conv(
            x,
            self.num_cell_units,
            (1, 1),
            (1, 1),
            'SAME',
            'True',
            name="tdnn_"+str(name),
            norm_type='layer')

        if type == 'SAME':
            x = tf.layers.max_pooling2d(x, (1, 1), (1, 1), 'SAME')
        elif type == 'HALF':
            x = tf.layers.max_pooling2d(x, (2, 1), (2, 1), 'SAME')
            len_sequence = tf.cast(tf.ceil(tf.cast(len_sequence, tf.float32)/2), tf.int32)

        x = tf.squeeze(x, axis=2)

        return x, len_sequence
