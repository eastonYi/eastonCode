#!/usr/bin/env python
# encoding: utf-8

"""
@author: Linho
@file: dcommon_layers.py
@time: 2017/9/8 16:31
@desc:
"""

import tensorflow as tf
# from tensor2tensor.layers import common_layers
# from tensor2tensor.utils import expert_utils as eu
# from tensorflow.python.ops import array_ops
from .common_layers import layer_norm

# CNN
def gated_conv(inputs, filter_num, kernel, stride, padding, name):
  with tf.variable_scope(name):
    net = tf.layers.conv2d(inputs, filter_num, kernel, stride, padding, name="value_conv")
    gated = tf.layers.conv2d(inputs, filter_num, kernel, stride, padding, name="gated_conv")
    output = net * tf.nn.sigmoid(gated)
    return output

def normal_conv(inputs, filter_num, kernel, stride, padding, use_relu, name,
                w_initializer=None, norm_type="batch"):
  with tf.variable_scope(name):
    net = tf.layers.conv2d(inputs, filter_num, kernel, stride, padding,
                           kernel_initializer=w_initializer, name="conv")
    if norm_type == "batch":
      net = tf.layers.batch_normalization(net, name="bn")
    elif norm_type == "layer":
      # if name == 'conv':
      #   net = tf.Print(net, [net], message='net1: ', summarize=100)
      # net = tf.contrib.layers.layer_norm(net)
      # net = layer_normalize(net, name)
      net = layer_norm(net)
      # if name == 'conv':
      #   net = tf.Print(net, [net], message='net2: ', summarize=100)
    else:
      net = net
    output = tf.nn.relu(net) if use_relu else net
    return output

def normal_conv1d(inputs, filter_num, use_relu, name, norm_type="batch"):
  with tf.variable_scope(name):
    net = tf.layers.conv2d(inputs, filter_num, (1,1), (1,1), 'SAME', name="conv")
    if norm_type == "batch":
      net = tf.layers.batch_normalization(net, name="bn")
    elif norm_type == "layer":
      net = tf.contrib.layers.layer_norm(net)
    output = tf.nn.relu(net) if use_relu else net
    return output

def res_cnn(input, filter_num, kernel, stride, padding, name, dropout=0.0, norm_type="batch"):
  with tf.variable_scope(name):
    residual = input
    net = tf.layers.conv2d(input, filter_num, kernel, stride, padding, name="conv1")
    if norm_type == "batch":
      net = tf.layers.batch_normalization(net, name="bn1")
    elif norm_type == "layer":
      net = tf.contrib.layers.layer_norm(net)
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(net, filter_num, kernel, stride, padding, name="conv2")
    if norm_type == "batch":
      net = tf.layers.batch_normalization(net, name="bn2")
    elif norm_type == "layer":
      net = tf.contrib.layers.layer_norm(net)
    if dropout != 0.0:
      net = tf.nn.dropout(net, 1.0 - dropout)
    output= tf.nn.relu(net + residual)
    return output

def normal_pooling(input, size, stride, padding):
  net = tf.layers.max_pooling2d(input, size, stride, padding)
  return net

def res_dws_cnn_block(x, filters, kernel, stride, padding, name, w_initializer=None):
  with tf.variable_scope(name):
    y = tf.layers.separable_conv2d(x, filters, kernel, stride, padding,
                                   depthwise_initializer=w_initializer,
                                   pointwise_initializer=w_initializer, name="dws_conv1")
    y = tf.nn.relu(tf.layers.batch_normalization(y, name="bn1"))
    y = tf.layers.separable_conv2d(y, filters, kernel, stride, padding,
                                   depthwise_initializer=w_initializer,
                                   pointwise_initializer=w_initializer, name="dws_conv2")
    y = tf.layers.batch_normalization(y, name="bn2")
    return tf.nn.relu(x + y)

# def res_lstm_block(x, filters, kernel, padding, name):
#   with tf.variable_scope(name):
#     y = common_layers.conv_lstm(x, kernel, filters, padding=padding, name="conv_lstm1")
#     y = tf.nn.relu(tf.layers.batch_normalization(y, name="bn1"))
#     y = common_layers.conv_lstm(y, kernel, filters, padding=padding, name="conv_lstm2")
#     y = tf.layers.batch_normalization(y, name="bn2")
#     return tf.nn.relu(x + y)

def res_block(input, filter_num, kernel, stride, padding, name, dropout=0.0, w_initializer=None):
  with tf.variable_scope(name):
    if input.get_shape().as_list()[3] != filter_num:
      residual = normal_conv(input, filter_num, (1,1), (1,1), 'SAME', False,
                             "residual_proj", w_initializer=w_initializer)
    else:
      residual = input
    net = tf.layers.conv2d(input, filter_num, kernel, stride, padding,
                           name="conv1", kernel_initializer=w_initializer)
    net = tf.layers.batch_normalization(net, name="bn1")
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(net, filter_num, kernel, stride, padding,
                           name="conv2", kernel_initializer=w_initializer)
    net = tf.layers.batch_normalization(net, name="bn2")
    if dropout != 0.0:
      net = tf.nn.dropout(net, 1.0 - dropout)
    output= tf.nn.relu(net + residual)
    return output

def rescnn_for2dim(input, filter_num, kernel, stride, padding, name, w_initializer=None):
  with tf.variable_scope(name):
    net = tf.layers.conv2d(input, filter_num * 8, kernel, stride, padding,
                           name="conv1", kernel_initializer=w_initializer)
    net = tf.layers.batch_normalization(net, name="bn1")
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(net, filter_num, kernel, stride, padding,
                           name="conv2", kernel_initializer=w_initializer)
    net = tf.layers.batch_normalization(net, name="bn2")
    return net

def layer_preprocess(inputs, hparams=None):
  return tf.contrib.layers.layer_norm(inputs)

# RNN
def basic_lstm_cell(num_cells, add_residual=False):
  lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_cells, forget_bias=1.0,
                                           reuse=tf.get_variable_scope().reuse)
  if add_residual:
    lstm_cell = tf.contrib.rnn.ResidualWrapper(lstm_cell)

  return lstm_cell

def ln_lstm_cell(num_cells, add_residual=False, dropout=0.0):
  lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_cells, forget_bias=1.0,
                                                    layer_norm=True,
                                                    dropout_keep_prob=1.0 - dropout,
                                                    reuse=tf.get_variable_scope().reuse)
  if add_residual:
    lstm_cell = tf.contrib.rnn.ResidualWrapper(lstm_cell)

  return lstm_cell

def ln_lstm_cells(num_layers, num_cells, add_residual=False, dropout=0.0):
  lstm_cells = []
  for i in range(num_layers):
    lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_cells, forget_bias=1.0,
                                                      layer_norm=True,
                                                      dropout_keep_prob=1.0 - dropout,
                                                      reuse=tf.get_variable_scope().reuse)
    if add_residual:
      lstm_cell = tf.contrib.rnn.ResidualWrapper(lstm_cell)

    lstm_cells.append(lstm_cell)

  multi_lstm_cells = tf.contrib.rnn.MultiRNNCell(lstm_cells, state_is_tuple=True)
  return multi_lstm_cells

def blstm_cells(num_layers, num_cells, num_projs=None,
                add_residual=False, dropout=0.0):
  # use the half of the cell number for each direction.
  num_cells /= 2
  num_projs = num_projs/2 if num_projs else num_projs

  fwd_lstm_cells = []
  bwd_lstm_cells = []
  for i in range(num_layers):
    fwd_lstm_cell = tf.contrib.rnn.LSTMCell(num_cells, use_peepholes=True,
                                            num_proj=num_projs, cell_clip=50.0,
                                            forget_bias=1.0,
                                            reuse=tf.get_variable_scope().reuse)

    bwd_lstm_cell = tf.contrib.rnn.LSTMCell(num_cells, use_peepholes=True,
                                            num_proj=num_projs, cell_clip=50.0,
                                            forget_bias=1.0,
                                            reuse=tf.get_variable_scope().reuse)

    if dropout > 0.0:
      fwd_lstm_cell = tf.contrib.rnn.DropoutWrapper(fwd_lstm_cell,
                                                    state_keep_prob=1.0 - dropout)
      bwd_lstm_cell = tf.contrib.rnn.DropoutWrapper(bwd_lstm_cell,
                                                    state_keep_prob=1.0 - dropout)
    if add_residual:
      fwd_lstm_cell = tf.contrib.rnn.ResidualWrapper(fwd_lstm_cell)
      bwd_lstm_cell = tf.contrib.rnn.ResidualWrapper(bwd_lstm_cell)

    fwd_lstm_cells.append(fwd_lstm_cell)
    bwd_lstm_cells.append(bwd_lstm_cell)

  multi_fwd_lstm_cells = tf.contrib.rnn.MultiRNNCell(fwd_lstm_cells,
                                                     state_is_tuple=True)
  multi_bwd_lstm_cells = tf.contrib.rnn.MultiRNNCell(bwd_lstm_cells,
                                                     state_is_tuple=True)
  return multi_fwd_lstm_cells, multi_bwd_lstm_cells

def blstm_cell(num_cells, num_projs=None, add_residual=False, dropout=0.0):
  # use the half of the cell number for each direction.
  num_cells /= 2
  num_projs = num_projs / 2 if num_projs else num_projs

  # fwd_lstm_cell = tf.contrib.rnn.LSTMCell(num_cells, use_peepholes=True,
  #                                         num_proj=num_projs, cell_clip=50.0,
  #                                         forget_bias=1.0,
  #                                         reuse=tf.get_variable_scope().reuse)
  #
  # bwd_lstm_cell = tf.contrib.rnn.LSTMCell(num_cells, use_peepholes=True,
  #                                         num_proj=num_projs, cell_clip=50.0,
  #                                         forget_bias=1.0,
  #                                         reuse=tf.get_variable_scope().reuse)
  fwd_lstm_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_cells)
  bwd_lstm_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_cells)

  if dropout > 0.0:
    fwd_lstm_cell = tf.contrib.rnn.DropoutWrapper(fwd_lstm_cell,
                                                  state_keep_prob=1.0 - dropout)
    bwd_lstm_cell = tf.contrib.rnn.DropoutWrapper(bwd_lstm_cell,
                                                  state_keep_prob=1.0 - dropout)
  if add_residual:
    fwd_lstm_cell = tf.contrib.rnn.ResidualWrapper(fwd_lstm_cell)
    bwd_lstm_cell = tf.contrib.rnn.ResidualWrapper(bwd_lstm_cell)

  return fwd_lstm_cell, bwd_lstm_cell

def lstm_cells(num_layers, num_cells, num_projs=None,
               add_residual=False, dropout=0.0, initializer=None):
  lstm_cells = []
  initializer = tf.orthogonal_initializer() if initializer == "orthogonal" else None
  for i in range(num_layers):
    lstm_cell = tf.contrib.rnn.LSTMCell(num_cells, use_peepholes=True,
                                        cell_clip=50.0, forget_bias=1.0,
                                        num_proj=num_projs, initializer=initializer,
                                        reuse=tf.get_variable_scope().reuse)

    if dropout > 0.0:
      lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,
                                                state_keep_prob=1.0 - dropout)
    if add_residual:
      lstm_cell = tf.contrib.rnn.ResidualWrapper(lstm_cell)

    lstm_cells.append(lstm_cell)

  multi_lstm_cells = tf.contrib.rnn.MultiRNNCell(lstm_cells, state_is_tuple=True)
  return multi_lstm_cells

def lstm_cell(num_cells, num_projs=None, add_residual=False, dropout=0.0, initializer=None):
  initializer = tf.orthogonal_initializer() if initializer == "orthogonal" else None
  lstm_cell = tf.contrib.rnn.LSTMCell(num_cells, use_peepholes=True,
                                      cell_clip=50.0, forget_bias=1.0,
                                      num_proj=num_projs, initializer=initializer,
                                      reuse=tf.get_variable_scope().reuse)

  if dropout > 0.0:
    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,
                                              state_keep_prob=1.0 - dropout)
  if add_residual:
    lstm_cell = tf.contrib.rnn.ResidualWrapper(lstm_cell)

  return lstm_cell

def calculate_not_padding(inputs, cnn_frontend=False, audio_compression=2, stride3=False):
  if cnn_frontend:
    ishape = tf.shape(inputs)
    inputs = tf.reshape(inputs, [ishape[0], ishape[1], ishape[2] * ishape[3]])

    feat_sum = tf.reduce_sum(tf.abs(inputs), axis=-1)
    not_padding = tf.to_float(tf.not_equal(feat_sum, 0.0))
    not_padding = tf.expand_dims(tf.expand_dims(not_padding, 2), 3)
    stride = 3 if stride3 else 2
    not_padding = tf.layers.max_pooling2d(not_padding, [stride ** audio_compression, 1],
                                          [stride ** audio_compression, 1], 'SAME')
    not_padding = tf.to_int32(tf.squeeze(not_padding, [2,3]))

    return not_padding
  else:
    feat_sum = tf.reduce_sum(tf.abs(inputs), axis=-1)
    not_padding = tf.to_int32(tf.not_equal(feat_sum, 0.0))

    return not_padding

# Embedding
# def get_sharded_weights(vocab_size, hidden_dim, num_shards=1):
#   """Create or get concatenated embedding or softmax variable.
#
#   Args:
#     vocab_size: The size of vocabulary.
#     hidden_dim: The dimension of hidden layer.
#     num_shards: divide the vocabulary to dozens of shards.
#
#   Returns:
#      a list of self._num_shards Tensors.
#   """
#   shards = []
#   for i in range(num_shards):
#     shard_size = (vocab_size // num_shards) + (
#       1 if i < vocab_size % num_shards else 0)
#     var_name = "weights_%d" % i
#     shards.append(
#       tf.get_variable(
#         var_name, [shard_size, hidden_dim],
#         initializer=tf.random_normal_initializer(0.0, hidden_dim ** -0.5)))
#   if num_shards == 1:
#     ret = shards[0]
#   else:
#     ret = tf.concat(shards, 0)
#   # Convert ret to tensor.
#   ret = eu.convert_gradient_to_tensor(ret)
#   return ret

# Conv_LSTM
class ConvLSTMCell(tf.nn.rnn_cell.RNNCell):
  """A LSTM cell with convolutions instead of multiplications.

  Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting."
    Advances in Neural Information Processing Systems. 2015.
  """

  def __init__(self, shape, filters, kernel, forget_bias=1.0, activation=tf.tanh,
               normalize=True, peephole=True, data_format='channels_last', reuse=None):
    super(ConvLSTMCell, self).__init__(_reuse=reuse)
    self._kernel = kernel
    self._filters = filters
    self._forget_bias = forget_bias
    self._activation = activation
    self._normalize = normalize
    self._peephole = peephole
    if data_format == 'channels_last':
        self._size = tf.TensorShape(shape + [self._filters])
        self._feature_axis = self._size.ndims
        self._data_format = None
    elif data_format == 'channels_first':
        self._size = tf.TensorShape([self._filters] + shape)
        self._feature_axis = 0
        self._data_format = 'NC'
    else:
        raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

  @property
  def output_size(self):
    return self._size

  def call(self, x, state):
    c, h = state

    x = tf.concat([x, h], axis=self._feature_axis)
    n = x.shape[-1].value
    m = 4 * self._filters if self._filters > 1 else 4
    W = tf.get_variable('kernel', self._kernel + [n, m])
    y = tf.nn.convolution(x, W, 'SAME', data_format=self._data_format)
    if not self._normalize:
      y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
    j, i, f, o = tf.split(y, 4, axis=self._feature_axis)

    if self._peephole:
      i += tf.get_variable('W_ci', c.shape[1:]) * c
      f += tf.get_variable('W_cf', c.shape[1:]) * c

    if self._normalize:
      j = tf.contrib.layers.layer_norm(j)
      i = tf.contrib.layers.layer_norm(i)
      f = tf.contrib.layers.layer_norm(f)

    f = tf.sigmoid(f + self._forget_bias)
    i = tf.sigmoid(i)
    c = c * f + i * self._activation(j)

    if self._peephole:
      o += tf.get_variable('W_co', c.shape[1:]) * c

    if self._normalize:
      o = tf.contrib.layers.layer_norm(o)
      c = tf.contrib.layers.layer_norm(c)

    o = tf.sigmoid(o)
    h = o * self._activation(c)

    state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

    return h, state


class ConvGRUCell(tf.nn.rnn_cell.RNNCell):
  """A GRU cell with convolutions instead of multiplications."""

  def __init__(self, shape, filters, kernel, activation=tf.tanh,
               normalize=True, data_format='channels_last', reuse=None):
    super(ConvGRUCell, self).__init__(_reuse=reuse)
    self._filters = filters
    self._kernel = kernel
    self._activation = activation
    self._normalize = normalize
    if data_format == 'channels_last':
        self._size = tf.TensorShape(shape + [self._filters])
        self._feature_axis = self._size.ndims
        self._data_format = None
    elif data_format == 'channels_first':
        self._size = tf.TensorShape([self._filters] + shape)
        self._feature_axis = 0
        self._data_format = 'NC'
    else:
        raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return self._size

  @property
  def output_size(self):
    return self._size

  def call(self, x, h):
    channels = x.shape[self._feature_axis].value

    with tf.variable_scope('gates'):
      inputs = tf.concat([x, h], axis=self._feature_axis)
      n = channels + self._filters
      m = 2 * self._filters if self._filters > 1 else 2
      W = tf.get_variable('kernel', self._kernel + [n, m])
      y = tf.nn.convolution(inputs, W, 'SAME', data_format=self._data_format)
      if self._normalize:
        r, u = tf.split(y, 2, axis=self._feature_axis)
        r = tf.contrib.layers.layer_norm(r)
        u = tf.contrib.layers.layer_norm(u)
      else:
        y += tf.get_variable('bias', [m], initializer=tf.ones_initializer())
        r, u = tf.split(y, 2, axis=self._feature_axis)
      r, u = tf.sigmoid(r), tf.sigmoid(u)

    with tf.variable_scope('candidate'):
      inputs = tf.concat([x, r * h], axis=self._feature_axis)
      n = channels + self._filters
      m = self._filters
      W = tf.get_variable('kernel', self._kernel + [n, m])
      y = tf.nn.convolution(inputs, W, 'SAME', data_format=self._data_format)
      if self._normalize:
        y = tf.contrib.layers.layer_norm(y)
      else:
        y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
      h = u * h + (1 - u) * self._activation(y)

    return h, h
