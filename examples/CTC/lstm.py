#!/usr/bin/env python
# encoding: utf-8

"""
@author: Linho
@file: lstm.py
@time: 2017/9/25 11:46
@desc:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import dcommon_layers
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf

@registry.register_model("Lstm")
class Lstm(t2t_model.T2TModel):
  # Input: [batch_size, time_step, freq_dim, init_channel]
  # Output: [batch_size, time_step, hidden_dim]
  # Body: [2*2D-Conv(2), 3*projected-LSTM, 2*ffn]
  def model_fn_body(self, features):
    hparams = self._hparams
    w_initializer = None
    inputs = features["inputs"]
    inputs_len = features["scp/input_seq_len"]

    entry_output, compressed_inputs_len = lstm_entry_part(inputs, inputs_len, hparams, w_initializer)
    # entry_output = tf.Print(entry_output, ["entry_output in lstm's shape:", tf.shape(entry_output)])
    middle_output = lstm_middle_part(entry_output, compressed_inputs_len, hparams, w_initializer)
    # middle_output = tf.Print(middle_output, ["middle_output in lstm's shape:", tf.shape(middle_output)])
    exit_output = lstm_exit_part(middle_output, hparams, w_initializer)
    # exit_output = tf.Print(exit_output, ["middle_output in lstm's shape:", tf.shape(exit_output)])
    return exit_output

## lhdong ctc (copy from model_fn_body, for exporting model)
def model_fn_body(inputs, inputs_len, hparams):
  w_initializer = None
  ## input placeholder reshape
  ishape = tf.shape(inputs)
  ishape_static = inputs.get_shape()
  inputs = tf.reshape(inputs, [ishape[0], ishape[1], ishape[2]//3, 3])
  inputs.set_shape([ishape_static[0], ishape_static[1], ishape_static[2]//3, 3])
  ##
  entry_output, compressed_inputs_len = lstm_entry_part(inputs, inputs_len, hparams, w_initializer)
  middle_output = lstm_middle_part(entry_output, compressed_inputs_len, hparams, w_initializer)
  exit_output = lstm_exit_part(middle_output, hparams, w_initializer)

  return exit_output
##

def lstm_entry_part(inputs, inputs_len, hparams, w_initializer):
  # [2*2D-Conv(2), 1D-proj-Conv]
  with tf.variable_scope("entry_cnn_part"):
    # Using twice (2, 2) stride to compress and accelerate
    net = dcommon_layers.normal_conv(inputs, hparams.entry_filter_nums[0],
                                     hparams.entry_kernel_sizes[0], (2, 2),
                                     'SAME', True, "entry_compress_conv1", w_initializer)
    net = dcommon_layers.normal_conv(net, hparams.entry_filter_nums[1],
                                     hparams.entry_kernel_sizes[1], (2, 2),
                                     'SAME', True, "entry_compress_conv2", w_initializer)
    compressed_inputs_len = tf.div(tf.to_float(inputs_len), hparams.audio_compression)
    compressed_inputs_len = tf.to_int32(tf.ceil(compressed_inputs_len))

    # Reshape to dim3 and map to lstm projection size
    nshape = tf.shape(net)
    nshape_static = net.get_shape()
    net = tf.reshape(net, [nshape[0], nshape[1], 1, nshape[2] * nshape[3]])
    net.set_shape([nshape_static[0], nshape_static[1], 1, nshape_static[2] * nshape_static[3]])

    net = dcommon_layers.normal_conv(net, hparams.lstm_proj_size, (1, 1), (1, 1),
                                     'SAME', False, "entry_proj_conv", w_initializer)
    outputs = tf.transpose(tf.squeeze(net, 2), [1, 0, 2])

  return outputs, compressed_inputs_len

# all lstm layers is included in one multi_lstm_cells
def lstm_middle_part(inputs, inputs_len, hparams, w_initializer):
  # [3*projected-LSTM]
  # tf.logging.info(inputs.device)
  with tf.variable_scope("medium_lstm_part") as vs:
    # tf.logging.info(vs.caching_device)
    # vs.set_caching_device(inputs.device)
    # tf.logging.info(vs.caching_device)
    # inputs = tf.Print(inputs, ["before dynamic_rnn:", inputs])
      # Wrap previous state after reset
      # layer_states = []
      # for i in range(hparams.lstm_layer):
      #   layer_states.append(tf.zeros_initializer([tf.shape(inputs)[1], hparams.lstm_proj_size]))
      # initial_state = tuple(layer_states)

      # Get LSTM cells
    # with tf.device(inputs.device):
    multi_lstm_cells = dcommon_layers.ln_lstm_cells(hparams.lstm_layer,
                                                    hparams.lstm_cell_size,
                                                    hparams.lstm_proj_size)
    outputs, _ = tf.nn.dynamic_rnn(cell=multi_lstm_cells, inputs=inputs,
                                   dtype=tf.float32, time_major=True,
                                   sequence_length=inputs_len)
    # outputs = tf.Print(outputs, ["after dynamic_rnn:", tf.shape(outputs)])

    outputs = tf.transpose(outputs, [1, 0, 2])

  # for tensor in tf.trainable_variables():
  #   if "medium_lstm_part" in tensor.name:
  #     tf.logging.info(tensor.name + "  " + tensor.device)
  #
  # tf.logging.info(outputs.device)
  return outputs

def lstm_middle_part2(inputs, inputs_len, hparams, w_initializer):
  # [3*projected-LSTM]
  with tf.variable_scope("medium_lstm_part") as vs:
    outputs = tf.transpose(inputs, [1, 0, 2])
    return outputs

# every lstm layer is independent
def lstm_middle_part3(inputs, inputs_len, hparams, w_initializer):
  # [3*projected-LSTM]
  net = inputs
  with tf.variable_scope("medium_lstm_part"):
    # Get LSTM cells
    for i in range(hparams.lstm_layer):
      with tf.variable_scope("lstm_%d"%i):
        lstm_cell = dcommon_layers.ln_lstm_cell(hparams.lstm_cell_size,
                                                      hparams.lstm_proj_size)
        net = tf.Print(net, [net])
        net, _ = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=net,
                                   dtype=tf.float32, time_major=True,
                                   sequence_length=inputs_len)
        net = tf.Print(net, ["after dynamic_rnn:", tf.shape(net)])

    outputs = tf.transpose(net, [1, 0, 2])
    return outputs

def lstm_exit_part(inputs, hparams, w_initializer):
  # [3*ffn]
  net = tf.expand_dims(inputs, 2)
  with tf.variable_scope("exit_ffn_part"):
    for i in range(hparams.ffn_hidden_layer):
      with tf.variable_scope("ffn_layer%d" % i):
        ## ffn computation
        net = dcommon_layers.ffn_layer(net, hparams)
    # Get outputs
    outputs = tf.contrib.layers.layer_norm(tf.squeeze(net, 2))
  return outputs

@registry.register_hparams
def lstm_tone_new2():
  hparams = lstm_tone_1gpu()
  hparams.lstm_layer = 4
  hparams.ffn_hidden_size = 1
  hparams.ffn_filter_size = 1024
  hparams.batch_size = 144000
  return hparams

@registry.register_hparams
def lstm_tone_new():
  hparams = lstm_tone_1gpu()
  hparams.lstm_layer = 3
  hparams.ffn_hidden_size = 2
  hparams.ffn_filter_size = 1024
  hparams.batch_size = 160000
  return hparams

@registry.register_hparams
def lstm_4gpu_debug():
  hparams = lstm_base()
  hparams.lstm_layer = 3
  hparams.ffn_hidden_size = 2
  hparams.ffn_filter_size = 1024
  hparams.batch_size = 160000
  return hparams

@registry.register_hparams
def lstm_tone_exp2():
  hparams = lstm_tone_1gpu()
  hparams.lstm_layer = 3
  hparams.ffn_hidden_size = 2
  hparams.ffn_filter_size = 1024
  hparams.batch_size = 160000
  return hparams

@registry.register_hparams
def lstm_tone_exp1():
  hparams = lstm_tone_1gpu()
  hparams.lstm_layer = 3
  hparams.ffn_hidden_size = 2
  hparams.ffn_filter_size = 1024
  hparams.batch_size = 128000
  return hparams

@registry.register_hparams
def lstm_tone_1gpu():
  hparams = lstm_base()
  return hparams

@registry.register_hparams
def lstm_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  # add some new hparams
  # All hyperparameters ending in "dropout" are automatically set to 0.0
  # when not in training mode.
  hparams.add_hparam("entry_filter_nums", [64, 128])
  hparams.add_hparam("entry_kernel_sizes", [(7, 7), (3, 3)])
  hparams.add_hparam("lstm_layer", 3)
  hparams.add_hparam("lstm_cell_size", 800)
  hparams.add_hparam("lstm_proj_size", 512)
  hparams.add_hparam("ffn_hidden_layer", 3)
  hparams.add_hparam("ffn_hidden_size", 512)
  hparams.add_hparam("ffn_filter_size", 2048)
  hparams.add_hparam("ffn_layer", "conv_hidden_relu")

  hparams.add_hparam("relu_dropout", 0.0)
  hparams.add_hparam("residual_dropout", 0.1)
  hparams.add_hparam("pos", "timing")  # timing, none
  hparams.add_hparam("proximity_bias", int(False))
  hparams.add_hparam("min_length", 8)
  hparams.add_hparam("feat_dim", 129)
  hparams.add_hparam("boundaries", [(hparams.min_length + hparams.max_length) // 2])
  hparams.add_hparam("audio_compression", 4)

  # network structure
  hparams.entry_filter_nums = [32, 64]
  hparams.entry_kernel_sizes = [(7, 7), (3, 3)]
  hparams.lstm_layer = 4
  hparams.lstm_cell_size = 512
  hparams.lstm_proj_size = 512
  hparams.ffn_hidden_layer = 1
  hparams.ffn_hidden_size = hparams.lstm_proj_size
  hparams.hidden_size = hparams.ffn_hidden_size
  hparams.ffn_filter_size = 2048
  hparams.num_pinyins = 1385
  hparams.norm_type = "layer"
  hparams.shared_embedding_and_softmax_weights = 0

  # learning relatives
  hparams.batch_size = 25600
  hparams.max_length = 1200
  hparams.min_length = 20
  hparams.boundaries = [94, 125, 149, 173, 194, 214, 234, 252, 273, 293, 314, 335, 358, 383, 411, 444, 485, 540, 629]
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate_warmup_steps = 10000  ##
  hparams.learning_rate = 0.4  ##
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.weight_decay = 1e-5
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98

  hparams.residual_dropout = 0.1  ##
  hparams.relu_dropout = 0.1  ##

  # others
  hparams.symbol_modality_num_shards = 1
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.label_smoothing = 0.0
  hparams.audio_compression = 2 ** (len(hparams.entry_filter_nums))

  return hparams
