#!/usr/bin/env python
# encoding: utf-8

"""
@author: Linho
@file: RNA.py
@time: 2017/12/8 11:21
@desc: Recurrent Neural Aligner (RNA) for streaming ASR
  encoder: [uni-LSTM] × n
  decoder: [uni-LSTM] × k
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import dcommon_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import beam_search_rna
from tensor2tensor.layers import common_hparams

import tensorflow as tf

from tensorflow.python.util import nest

@registry.register_model
class RNA(t2t_model.T2TModel):
  def encode(self, inputs, hparams):
    """Encode RNA inputs.

    Args:
      inputs: RNA inputs.
        [batch_size, input_length, fbank_dim, fbank_channel]
      hparams: hyper-parameters for model.

    Returns:
      encoder_output: Encoder representation.
        [batch_size, input_length/stride, encoder_lstm_cells]
      not_padding: Used for loss computation and correct search length
        [batch_size, input_length/stride]
    """
    x = inputs

    # Front-end part
    if hparams.use_cnn_frontend:
      # CNN front-end
      with tf.variable_scope("encoder_frontend"):
        for i in range(hparams.frontend_cnn_layers):
          with tf.variable_scope("block%d"%(i+1)):
            filters = hparams.frontend_cnn_filters * (i + 1)
            if hparams.use_data745h:
              stride = (2, 1)
            else:
              stride = (3,2) if hparams.conv_time_stride3 else (2,2)
            x = dcommon_layers.normal_conv(x, filters, (3,3), stride,
                                           'SAME', True, "conv",
                                           norm_type=hparams.norm_type)
            if hparams.additional_module:
              if hparams.additional_module == "mu31":
                for _ in range(hparams.additional_module_layers):
                  x = common_layers.conv_lstm(x, (3,1), filters)
              elif hparams.additional_module == "mu33":
                for _ in range(hparams.additional_module_layers):
                  x = common_layers.conv_lstm(x, (3,3), filters)
              elif hparams.additional_module == "conv_lstm":
                not_padding = dcommon_layers.calculate_not_padding(inputs, True, i + 1)
                seq_lengths = tf.reduce_sum(not_padding, axis=-1)

                # conv_lstm_cell = dcommon_layers.ConvLSTMCell(shape=[x.get_shape().as_list()[2]],
                #                                              filters=filters, kernel=[3], peephole=False,
                #                                              normalize=False)
                # x, _ = tf.nn.dynamic_rnn(cell=conv_lstm_cell, inputs=x, dtype=tf.float32,
                #                          time_major=False, sequence_length=seq_lengths)

                conv_lstm_cell = dcommon_layers.ConvLSTMCell(shape=[x.get_shape().as_list()[2]],
                                                             filters=filters/2, kernel=[3], peephole=False,
                                                             normalize=False)
                x, _ = tf.nn.bidirectional_dynamic_rnn(conv_lstm_cell, conv_lstm_cell, inputs=x, dtype=inputs.dtype,
                                                       time_major=False, sequence_length=seq_lengths)
                x = tf.concat(x, 3)
              elif hparams.additional_module == "res_cnn":
                x = dcommon_layers.res_cnn(x, filters,
                                             (3, 3), (1, 1), 'SAME', 'rescnn', norm_type=hparams.norm_type)
              elif hparams.additional_module == "glu":
                x = dcommon_layers.normal_conv(x, filters, (3, 3), (1,1), 'SAME', True, "glu_conv1",
                                               norm_type=hparams.norm_type)
                x = x * tf.sigmoid(dcommon_layers.normal_conv(x, filters, (3, 3), (1,1), 'SAME',
                                                              True, "glu_conv2", norm_type=hparams.norm_type))
              else:
                tf.logging.debug("No such additional module!")

      xshape = tf.shape(x)
      xshape_static = x.get_shape()
      x = tf.reshape(x, [xshape[0], xshape[1], xshape[2] * xshape[3]])
      x.set_shape([xshape_static[0], xshape_static[1], xshape_static[2] * xshape_static[3]])

      # Obtain each utterance length in current batch
      not_padding = dcommon_layers.calculate_not_padding(inputs, True, hparams.frontend_cnn_layers, hparams.conv_time_stride3)
    else:
      # Consecutive frames
      # Flatten the original 4d inputs to 3d
      xshape = tf.shape(x)
      xshape_static = x.get_shape()
      x = tf.reshape(x, [xshape[0], xshape[1], xshape[2] * xshape[3]])
      x.set_shape([xshape_static[0], xshape_static[1], xshape_static[2] * xshape_static[3]])

      # Stacking and subsampling frames
      stacked = tf.concat([tf.pad(x[:, i:, :], [[0, 0], [0, i], [0, 0]])
                            for i in range(hparams.num_frame_stacking)], 2)
      x = tf.strided_slice(stacked, [0, 0, 0], tf.shape(stacked),
                           [1, hparams.num_frame_striding, 1])
      x.set_shape([None, None, stacked.get_shape()[2]])

      # Obtain each utterance length in current batch
      not_padding = dcommon_layers.calculate_not_padding(x)

    # Projection to prevent too big input dims
    if hparams.use_cnn_frontend:
      pass
    else:
      proj_dim = hparams.encoder_lstm_projs or hparams.encoder_lstm_cells
      x = dcommon_layers.normal_conv(tf.expand_dims(x, 2), proj_dim, (1,1), (1,1),
                                     'SAME', True, 'Proj', norm_type=hparams.norm_type)
      x = tf.squeeze(x, 2)

    # Lstm part
    with tf.variable_scope("encoder_lstms"):
      # Acquire the sequence length for dynamic rnn
      seq_lengths = tf.reduce_sum(not_padding, axis=-1)

      for i in range(hparams.encoder_lstm_layers):
        with tf.variable_scope("lstm%d" % (i + 1)):
          if hparams.encoder_use_blstm:
            fwd_lstm_cell, bwd_lstm_cell = \
              dcommon_layers.blstm_cell(hparams.encoder_lstm_cells,
                                        hparams.encoder_lstm_projs,
                                        hparams.encoder_add_residual,
                                        hparams.encoder_dropout)
            x, _ = tf.nn.bidirectional_dynamic_rnn(
              cell_fw=fwd_lstm_cell, cell_bw=bwd_lstm_cell, inputs=x,
              dtype=tf.float32, time_major=False, sequence_length=seq_lengths
            )
            x = tf.concat(x, 2)
          else:
            lstm_cell = dcommon_layers.lstm_cell(hparams.encoder_lstm_cells,
                                                 hparams.encoder_lstm_projs,
                                                 hparams.encoder_add_residual,
                                                 hparams.encoder_dropout, hparams.lstm_initializer)
            x, _ = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=x, dtype=tf.float32,
                                     time_major=False, sequence_length=seq_lengths)

          norm_type = hparams.norm_type if hparams.encoder_tdnn_use_norm else None
          use_relu = hparams.encoder_tdnn_use_relu

          if i == hparams.encoder_lstm_layers - 1:
            if hparams.encoder_output_tdnn:
              # handle the output of encoder
              x = tf.expand_dims(x, axis=2)
              x = dcommon_layers.normal_conv(x, hparams.decoder_lstm_cells,
                                             (hparams.tdnn_window_width, 1),
                                             (1, 1), 'SAME', use_relu, name="tdnn",
                                             norm_type=norm_type)
              if hparams.tdnn_stride > 1:
                x = dcommon_layers.normal_pooling(x, (hparams.tdnn_stride, 1),
                                                        (hparams.tdnn_stride, 1), 'SAME')
                # handle the padding mask
                not_padding = tf.expand_dims(tf.expand_dims(not_padding, 2), 3)
                not_padding = dcommon_layers.normal_pooling(not_padding, (hparams.tdnn_stride, 1),
                                                            (hparams.tdnn_stride, 1), 'SAME')
                not_padding = tf.squeeze(not_padding, [2, 3])
              x = tf.squeeze(x, axis=2)
          else:
            if hparams.encoder_use_NiN:
              x = tf.expand_dims(x, 2)
              if hparams.tdnn_window_width > 1:
                x = dcommon_layers.normal_conv(x, hparams.encoder_lstm_cells,
                                               (hparams.tdnn_window_width, 1),
                                               (1, 1), 'SAME', use_relu, name='interleaved_conv%d' % (i + 1),
                                               norm_type=norm_type)
              else:
                x = dcommon_layers.normal_conv1d(x, hparams.encoder_lstm_cells, use_relu,
                                               'interleaved_conv%d' % (i + 1),
                                                 norm_type=norm_type)

              if (i + 1) in hparams.lstm_pooling_layers:
                if (i + 1) == hparams.lstm_pooling_layers[0] and hparams.first_pooling_width3:
                  pooling_width = 3
                else:
                  pooling_width = 2
                x = dcommon_layers.normal_pooling(x, (pooling_width, 1),
                                                  (pooling_width, 1), 'SAME')
                # handle the padding mask
                not_padding = tf.expand_dims(tf.expand_dims(not_padding, 2), 3)
                not_padding = dcommon_layers.normal_pooling(not_padding, (pooling_width, 1),
                                                            (pooling_width, 1), 'SAME')
                not_padding = tf.squeeze(not_padding, [2, 3])
                seq_lengths = tf.reduce_sum(not_padding, axis=-1)

              x = tf.squeeze(x, 2)
      outputs = x

    return outputs, not_padding

  def decode(self, encoder_outputs, hparams):
    """Decode RNA outputs from encoder outputs and the decoder prediction of last step.

    Args:
      encoder_outputs: Encoder representation.
        [batch_size, input_length/stride, hidden_dim]
      hparams: hyper-parameters for model.

    Returns:
      decoder_output: Decoder representation.
        [batch_size, input_length/stride, 1, 1, vocab_size]
    """
    # Loop body
    def inner_loop(i, prev_ids, prev_states, logits, att_results=None):
      return symbols_to_logits_fn(i, prev_ids, prev_states, logits,
                                  encoder_outputs, hparams, att_results)

    # While loop initialization
    batch_size = common_layers.shape_list(encoder_outputs)[0]
    decode_length = common_layers.shape_list(encoder_outputs)[1]

    # common initial part
    if hparams.start_with_blank:
      initial_ids = tf.fill([batch_size], hparams.vocab_size - 1)
    else:
      initial_ids = tf.fill([batch_size], 1)  # <SOS>
    initial_logits = tf.zeros([batch_size, 1, hparams.vocab_size], dtype=tf.float32)

    # collect the initial states of lstms used in decoder.
    all_initial_states = {}

    # the initial states of lstm which model the symbol recurrence (lm).
    initial_states = []
    zero_states = tf.zeros([batch_size, hparams.decoder_lstm_cells], dtype=tf.float32)
    for i in range(hparams.decoder_lstm_layers):
      initial_states.append(tf.contrib.rnn.LSTMStateTuple(zero_states, zero_states))
    all_initial_states["lstm_states"] = tuple(initial_states)

    # the initial states of lstm which is on the upper of previous lstm,
    # which model the recurrence of the fusion of am and lm states (am + lm).
    if hparams.decoder_use_lstm2:
      lstm2_initial_states = []
      lstm2_zero_states = tf.zeros([batch_size, hparams.decoder_lstm2_cells],
                                   dtype=tf.float32)
      for i in range(hparams.decoder_lstm2_layers):
        lstm2_initial_states.append(tf.contrib.rnn.LSTMStateTuple(lstm2_zero_states,
                                                                  lstm2_zero_states))
      all_initial_states["lstm2_states"] = tuple(lstm2_initial_states)

    # the initial states of lm lstm used for extra symbol recurrence (extra lm).
    if hparams.lm_fusion:
      lm_initial_states = []
      lm_zero_states = tf.zeros([batch_size, hparams.lm_lstm_cells], dtype=tf.float32)
      for i in range(hparams.lm_lstm_layers):
        lm_initial_states.append(tf.contrib.rnn.LSTMStateTuple(lm_zero_states,
                                                               lm_zero_states))
      all_initial_states["lm_lstm_states"] = tuple(lm_initial_states)

    if hparams.use_attention:
      initial_att_results = tf.zeros([batch_size, hparams.decoder_lstm_cells], dtype=tf.float32)

      # The While loop
      _, _, _, logits, _ = tf.while_loop(
        lambda i, *_: tf.less(i, decode_length),
        inner_loop,
        [tf.constant(0), initial_ids, all_initial_states, initial_logits, initial_att_results],
        shape_invariants=[
          tf.TensorShape([]),
          tf.TensorShape([None]),
          nest.map_structure(lambda t: tf.TensorShape(t.shape), all_initial_states),
          tf.TensorShape([None, None, hparams.vocab_size]),
          tf.TensorShape([None, hparams.decoder_lstm_cells]),
        ]
      )
      # logits = tf.Print(logits, ["logits:", tf.shape(logits)])
    else:
      # The While loop
      _, _, _, logits = tf.while_loop(
        lambda i, *_: tf.less(i, decode_length),
        inner_loop,
        [tf.constant(0), initial_ids, all_initial_states, initial_logits],
        shape_invariants=[
          tf.TensorShape([]),
          tf.TensorShape([None]),
          nest.map_structure(lambda t: tf.TensorShape(t.shape), all_initial_states),
          tf.TensorShape([None, None, hparams.vocab_size])
        ]
      )

    logits = tf.expand_dims(tf.expand_dims(logits[:, 1:, :], 2), 3)

    return logits

  def model_fn_body(self, features):
    """RNA main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
        "inputs": RNA inputs.
          [batch_size, input_length, fbank_dim, fbank_channel]
        (USELESS KEY:)
        "targets": Target decoder outputs.
        "target_space_id": Scalar

    Returns:
      A dict contains the following:
        "logits": Final decoder representation.
          [batch_size, input_length/stride, 1, 1, vocab_size]
        "not_padding": Used for loss computation
          [batch_size]
    """
    hparams = self._hparams

    # Encoder part
    inputs = features["inputs"]
    encoder_outputs, not_padding = self.encode(inputs, hparams)

    # Decoder part
    logits = self.decode(encoder_outputs, hparams)

    # Return
    if hparams.using_label_smoothing:
      real_probs = tf.nn.softmax(tf.squeeze(logits, axis=[2, 3]))
      prevent_nan_constant = tf.constant(1e-10)
      real_probs += prevent_nan_constant

      if hparams.ls_type == "uniform":
        uniform_probs = tf.to_float([1 / hparams.vocab_size for _ in range(hparams.vocab_size)])
        kl_divergence = tf.reduce_sum(real_probs * tf.log(real_probs / uniform_probs), axis=-1)
        kl_divergence = kl_divergence * tf.to_float(not_padding)
        ls_loss = hparams.ls_lambda * tf.reduce_sum(kl_divergence, axis=-1)

      elif hparams.ls_type == "unigram":
        unigram_probs = []
        with open(hparams.unigram_txt_dir, 'r') as unigram_reader:
          for prob_line in unigram_reader:
            unigram_probs.append(float(prob_line))

        unigram_probs = tf.to_float(unigram_probs)
        kl_divergence = tf.reduce_sum(real_probs * tf.log(real_probs / unigram_probs), axis=-1)
        kl_divergence = kl_divergence * tf.to_float(not_padding)
        ls_loss = hparams.ls_lambda * tf.reduce_sum(kl_divergence, axis=-1)
      return {"logits": logits, "not_padding": not_padding, "ls_loss": ls_loss}
    elif hparams.using_confidence_penalty:
      real_probs = tf.nn.softmax(tf.squeeze(logits, axis=[2, 3]))
      prevent_nan_constant = tf.constant(1e-10)
      real_probs += prevent_nan_constant

      neg_entropy = tf.reduce_sum(real_probs * tf.log(real_probs), axis=-1)
      ls_loss = hparams.ls_lambda * tf.reduce_sum(neg_entropy, axis=-1)

      return {"logits": logits, "not_padding": not_padding, "ls_loss": ls_loss}
    else:
      return {"logits": logits, "not_padding": not_padding}


  def _greedy_infer(self, features, _):
    """Greedy decoding

    Args:
      features: Map of features. Should at least contain the following:
        "inputs": Model inputs. [batch_size, input_length, fbank_dim, fbank_channel]
        "targets": Target of utterance. [batch_size, target_length, 1, 1]

    Returns:
      decoded_ids: [batch_size, input_length/stride]
    """
    with tf.variable_scope(self.name):
      decoded_ids, _ = self._fast_decode(features)
      return decoded_ids, None, None


  def _beam_decode(self, features, useless1, beam_size, top_beams, useless2):
    """Beam search decoding.

    Args:
      features: an map of string to `Tensor`
      useless1(decode_length): an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      useless2(alpha): Float that controls the length penalty. larger the alpha, stronger
        the preference for slonger translations.

    Returns:
       samples: an integer `Tensor`. Top samples from the beam search
    """
    tf.logging.info("Here")
    with tf.variable_scope(self.name):
      decoded_ids, scores = self._fast_decode(features, beam_size, top_beams)
      return {"outputs": decoded_ids, "scores": scores}


  def _fast_decode(self,
                   features,
                   beam_size=1,
                   top_beams=1):
    """Decoding

    Implements both greedy and beam search decoding, uses beam search iff
    beam_size > 1, otherwise beam search relates arguments are ignored.

    Args:
      features: Map of features
      beam_size: Number of beams
      top_beams: An integet. How many of the beams to return.

    Return:
      decoded_ids: Decoding results.
      scores: Decoding probabilities.
    """
    hparams = self._hparams

    inputs = features["inputs"]
    inputs = tf.bitcast(inputs, tf.float32)
    inputs.set_shape([None, None, hparams.feat_dim // 3, 3])

    with tf.variable_scope("body", reuse=tf.AUTO_REUSE):
      # Encoder part
      encoder_outputs, not_padding = self.encode(inputs, hparams)

      if beam_size == 1:
        tf.logging.info("Using Greedy Decoding")
        # Using decoder part directly
        logits = self.decode(encoder_outputs, hparams)

        # Masking the redundant decoded_ids
        decoded_ids = tf.argmax(tf.squeeze(logits, [2, 3]), -1)
        decoded_ids = tf.multiply(tf.to_int32(decoded_ids), not_padding)
        scores = None
      else:  # Beam search
        tf.logging.info("Using Beam Search Decoding")
        ids, scores = beam_search_rna.beam_search(symbols_to_logits_fn, encoder_outputs,
                                                  not_padding, beam_size, hparams)
        if top_beams == 1:
          return ids[:, 0, 1:], scores[:, 0]
        else:
          return ids[:, :top_beams, 1:], scores[:, :top_beams]

    return decoded_ids, scores


# inner loop function of the decoder
def symbols_to_logits_fn(i, prev_ids, all_lstm_states, logits,
                         encoder_outputs, hparams, prev_att_results=None):
  """ Predicting current logits based on previous

  Args:
    i: loop index
    prev_ids: The ids of previous step. [batch_size]
    all_lstm_states: The dict of lstm states of previous step. Should contains:
      tuple of:
        "lstm1_states": ([batch_size, decode_lstm_cells],
                         [batch_size, decode_lstm_cells])
        "lstm2_states": ([batch_size, decode_lstm2_cells],
                         [batch_size, decode_lstm2_cells])
        "lm_lstm_states": ([batch_size, lm_lstm_cells],
                           [batch_size, lm_lstm_cells])
    logits: The concatanated logits of previous steps. [batch_size, None, vocab_size]
    encoder_outputs: The outputs of encoder.
    hparams: The set of hyper-parameters.
    prev_att_results: The previous attention results. [batch_size, decoder_lstm_cells]

  Returns:
    i, next_ids, next_states, logits.

  """
  if hparams.use_attention:
    padding = [[0,0], [hparams.att_window_width / 2, hparams.att_window_width / 2], [0,0]]
    encoder_outputs = tf.pad(encoder_outputs, tf.to_int32(padding), mode='SYMMETRIC')

  # Get the previous states of lstms.
  lstm_states = all_lstm_states["lstm_states"]

  lstm2_states = None
  if "lstm2_states" in all_lstm_states.keys():
    lstm2_states = all_lstm_states["lstm2_states"]

  lm_lstm_states = None
  if "lm_lstm_states" in all_lstm_states.keys():
    lm_lstm_states = all_lstm_states["lm_lstm_states"]

  # Get the embedding of previous prediction labels.
  with tf.variable_scope("decoder_emb"):
    var_emb = dcommon_layers.get_sharded_weights(hparams.vocab_size,
                                                 hparams.decoder_lstm_cells,
                                                 hparams.symbol_modality_num_shards)
    prev_emb = tf.gather(var_emb, prev_ids, axis=0)
    if hparams.multiply_embedding_mode == "sqrt_depth":
      prev_emb *= hparams.decoder_lstm_cells ** 0.5

    ###############################
    # Maybe useless----------------
    if hparams.mask_blank_emb:
      blank_id = hparams.vocab_size - 1
      not_blank_mask = tf.not_equal(prev_ids, tf.fill(tf.shape(prev_ids), blank_id))
      not_blank_masks = tf.stack([not_blank_mask for _ in range(hparams.decoder_lstm_cells)],
                                 axis=1)
      not_blank_masks = tf.to_float(not_blank_masks)
      prev_emb *= not_blank_masks
      # Maybe useless----------------
      ###############################

  if hparams.use_attention:
    decoder_inputs = tf.concat([prev_emb, prev_att_results], axis=1)
  else:
    # Concat the prediction embedding and the encoder_output
    if hparams.decoder_lstm_concat:
      eshape = tf.shape(encoder_outputs)
      initial_tensor = tf.zeros([eshape[0], eshape[2]])
      initial_tensor.set_shape([None, hparams.decoder_lstm_cells])
      prev_encoder_output = tf.cond(tf.equal(i, 0), lambda: initial_tensor,
                                    lambda: encoder_outputs[:, i - 1, :])
      decoder_inputs = tf.concat([prev_encoder_output, prev_emb], axis=1)
    else:
      decoder_inputs = prev_emb

  # Lstm part
  with tf.variable_scope("decoder_lstms"):
    multi_lstm_cells = dcommon_layers.lstm_cells(hparams.decoder_lstm_layers,
                                                 hparams.decoder_lstm_cells,
                                                 initializer=hparams.lstm_initializer,
                                                 dropout=hparams.decoder_dropout)
    prev_lstm_states = lstm_states
    lstm_outputs, lstm_states = \
      tf.contrib.legacy_seq2seq.rnn_decoder(decoder_inputs=[decoder_inputs],
                                            initial_state=lstm_states,
                                            cell=multi_lstm_cells)
  ############################
  # Maybe useless-------------
  if hparams.load_trained_lm_as_decoder:
    # if hparams.is_debug:
    #   i = tf.Print(i, ["i:", i])
    #   prev_ids = tf.Print(prev_ids, ["prev_ids:", prev_ids], summarize=30)

    # If previous id is <blank>, don't update the lstm states and its outputs.
    blank_id = hparams.vocab_size - 1
    prev_mask = tf.equal(prev_ids, tf.fill(tf.shape(prev_ids), blank_id))
    cur_mask = tf.not_equal(prev_ids, tf.fill(tf.shape(prev_ids), blank_id))

    # if hparams.is_debug:
    #   prev_mask = tf.Print(prev_mask, ["prev_mask", prev_mask], summarize=30)
    #   cur_mask = tf.Print(cur_mask, ["cur_mask", cur_mask], summarize=30)

    prev_masks = tf.stack([prev_mask for _ in range(hparams.lm_lstm_cells)], axis=1)
    cur_masks = tf.stack([cur_mask for _ in range(hparams.lm_lstm_cells)], axis=1)
    prev_masks = tf.to_float(prev_masks)
    cur_masks = tf.to_float(cur_masks)

    updated_lstm_states = []
    for layer in range(hparams.lm_lstm_layers):
      h_states = prev_masks * prev_lstm_states[layer].h + \
                 cur_masks * lstm_states[layer].h
      c_states = prev_masks * prev_lstm_states[layer].c + \
                 cur_masks * lstm_states[layer].c

      updated_lstm_states.append(tf.contrib.rnn.LSTMStateTuple(c_states, h_states))
    lstm_states = tuple(updated_lstm_states)

    lstm_outputs[0] = lstm_states[-1].h

    # if hparams.is_debug:
    #   lstm_outputs[0] = tf.Print(lstm_outputs[0], ["lstm_states:", lstm_outputs[0][0], lstm_outputs[0][1]])
  # Maybe useless--------------
  #############################

  if hparams.use_attention:
    att_context = encoder_outputs[:, i:i + hparams.att_window_width, :]

    s_i = tf.expand_dims(tf.expand_dims(lstm_outputs[0], axis=1), axis=2)
    h_t = tf.expand_dims(att_context, axis=2)
    s_i = dcommon_layers.normal_conv1d(s_i, hparams.decoder_lstm_cells, False,
                                       "att_proj1", norm_type=None)
    h_t = dcommon_layers.normal_conv1d(h_t, hparams.decoder_lstm_cells, False,
                                       "att_proj2", norm_type=None)
    s_i = tf.squeeze(s_i, axis=[2])
    h_t = tf.squeeze(h_t, axis=[2])

    if hparams.att_type == "dot_product":
      energy = tf.matmul(s_i, tf.transpose(h_t, [0, 2, 1]))
    elif hparams.att_type == "content":
      with tf.variable_scope("att_vector1"):
        bias = dcommon_layers.get_sharded_weights(1, hparams.decoder_lstm_cells)
      with tf.variable_scope("att_vector2"):
        w = dcommon_layers.get_sharded_weights(1, hparams.decoder_lstm_cells)

      bias = tf.expand_dims(bias, axis=0)
      zero_tensor = tf.fill(tf.shape(h_t), 0.0)
      w = zero_tensor + w

      energy = tf.reduce_sum(w * tf.tanh(h_t + s_i + bias), axis=-1)
      energy = tf.expand_dims(energy, axis=1)

    att_weight = tf.nn.softmax(energy)
    # if hparams.is_debug:
    #   att_weight = tf.Print(att_weight, ["att_weight:", att_weight[0], att_weight[1]], summarize=10)

    cur_att_results = tf.matmul(att_weight, att_context)
    cur_att_results = tf.squeeze(cur_att_results, axis=1)

    pre_softmax_inputs = tf.concat([lstm_outputs[0], cur_att_results], axis=1)
  else:
    # Optional lstm for the recurrence of the fusion of am and lm states
    if lstm2_states:
      lstm2_inputs = tf.concat([lstm_outputs[0], encoder_outputs[:, i, :]], axis=1)
      with tf.variable_scope("decoder_lstm2"):
        multi_lstm2_cells = dcommon_layers.lstm_cells(hparams.decoder_lstm2_layers,
                                                     hparams.decoder_lstm2_cells,
                                                      initializer=hparams.lstm_initializer,
                                                      dropout=hparams.decoder_dropout)

        lstm2_outputs, lstm2_states = \
          tf.contrib.legacy_seq2seq.rnn_decoder(decoder_inputs=[lstm2_inputs],
                                                initial_state=lstm2_states,
                                                cell=multi_lstm2_cells)
      pre_softmax_inputs = lstm2_outputs[0]
    else:
      if hparams.decoder_proj_add:
        pre_softmax_inputs = tf.add(lstm_outputs[0], encoder_outputs[:, i, :])
      else:
        pre_softmax_inputs = tf.concat([lstm_outputs[0], encoder_outputs[:, i, :]], axis=1)

  # Lm fusion or normal "mlp + softmax"
  if lm_lstm_states:
    cur_logits, lm_lstm_states = lm_fusion(prev_ids, lm_lstm_states,
                                          pre_softmax_inputs, hparams)
  else:
    # lstm_outputs: a list of outputs, using the element 0
    with tf.variable_scope("softmax", reuse=False):
      if hparams.decoder_use_lstm2 or hparams.decoder_proj_add:
        input_dim = hparams.decoder_lstm_cells
      else:
        input_dim = 2 * hparams.decoder_lstm_cells
      var_top = dcommon_layers.get_sharded_weights(hparams.vocab_size, input_dim,
                                                   hparams.symbol_modality_num_shards)
      cur_logits = tf.matmul(pre_softmax_inputs, var_top, transpose_b=True)

  # Refresh the elements
  cur_ids = tf.to_int32(tf.argmax(cur_logits, -1))
  logits = tf.concat([logits, tf.expand_dims(cur_logits, 1)], 1)

  all_lstm_states["lstm_states"] = lstm_states
  if lstm2_states:
    all_lstm_states["lstm2_states"] = lstm2_states
  if lm_lstm_states:
    all_lstm_states["lm_lstm_states"] = lm_lstm_states

  if hparams.use_attention:
    return i + 1, cur_ids, all_lstm_states, logits, cur_att_results
  else:
    return i + 1, cur_ids, all_lstm_states, logits

def lm_fusion(prev_ids, lm_prev_states, am_states, hparams):
  """
  Fused with language model.

  Args:
    i: The index of while loop.
    prev_ids: The previous character ids. [batch_size]
    lm_prev_states: The previous lstm states of language model.
      tuple of:
      ([batch_size, lm_cell_size],
      [batch_size, lm_cell_size])
    am_states: The RNA decoder output before MLP + softmax.
        [batch_size, 2*decoder_cell_size]
    hparams: The set of hyper-parameters.

  Returns:
    cur_logit: The current logit.
    lm_cur_states: The current lstm states of language model
  """
  norm_type = hparams.norm_type if hparams.lm_use_norm else None

  # Embedding part
  with tf.variable_scope("lm_part", reuse=tf.AUTO_REUSE):
    with tf.variable_scope("shared"):
      lm_var_emb = dcommon_layers.get_sharded_weights(hparams.vocab_size,
                                                      hparams.lm_lstm_cells)
      lm_prev_emb = tf.gather(lm_var_emb, prev_ids, axis=0)
      if hparams.multiply_embedding_mode == "sqrt_depth":
        lm_prev_emb *= hparams.lm_lstm_cells ** 0.5

    # Lstm part
    lm_multi_lstm_cells = dcommon_layers.lstm_cells(hparams.lm_lstm_layers,
                                                    hparams.lm_lstm_cells,
                                                    initializer=hparams.lstm_initializer,
                                                    dropout=hparams.lm_dropout)

    _, lm_cur_states = \
      tf.contrib.legacy_seq2seq.rnn_decoder(decoder_inputs=[lm_prev_emb],
                                            initial_state=lm_prev_states,
                                            cell=lm_multi_lstm_cells)

    # If previous id is <blank>, don't update the lstm states and its outputs.
    blank_id = hparams.vocab_size - 1
    prev_mask = tf.equal(prev_ids, tf.fill(tf.shape(prev_ids), blank_id))
    cur_mask = tf.not_equal(prev_ids, tf.fill(tf.shape(prev_ids), blank_id))
    prev_masks = tf.stack([prev_mask for _ in range(hparams.lm_lstm_cells)], axis=1)
    cur_masks = tf.stack([cur_mask for _ in range(hparams.lm_lstm_cells)], axis=1)
    prev_masks = tf.to_float(prev_masks)
    cur_masks = tf.to_float(cur_masks)

    lm_updated_states = []
    for layer in range(hparams.lm_lstm_layers):
      h_states = prev_masks * lm_prev_states[layer].h + \
                 cur_masks * lm_cur_states[layer].h
      c_states = prev_masks * lm_prev_states[layer].c + \
                 cur_masks * lm_cur_states[layer].c

      lm_updated_states.append(tf.contrib.rnn.LSTMStateTuple(c_states, h_states))
    lm_updated_states = tuple(lm_updated_states)

    # Top part
    with tf.variable_scope("shared", reuse=True):
      lm_var_top = dcommon_layers.get_sharded_weights(hparams.vocab_size,
                                                      hparams.lm_lstm_cells)
      lm_cur_logits = tf.matmul(lm_updated_states[-1].h, lm_var_top, transpose_b=True)

  with tf.variable_scope("fusion_part"):
    if hparams.linho_fusion:
      with tf.variable_scope("linho_fusion"):
        lm_states = lm_updated_states[-1].h
        concated_states = tf.concat([lm_states, am_states], axis=-1)
        concated_states = tf.expand_dims(tf.expand_dims(concated_states, 1), 1)

        lm_gates = dcommon_layers.normal_conv1d(concated_states, hparams.lm_lstm_cells,
                                                False, 'lm_states_proj', norm_type=norm_type)
        lm_gates = tf.nn.sigmoid(lm_gates)
        lm_gated_states = lm_states * tf.squeeze(lm_gates, axis=[1, 2])

        if hparams.linho_fusion_type == "concat":
          if hparams.lm_pre_softmax_proj:
            concated_states = tf.expand_dims(tf.expand_dims(tf.concat([lm_gated_states,
                                                                       am_states], axis=-1), 1), 1)
            projed_states = dcommon_layers.normal_conv1d(concated_states, hparams.lm_lstm_cells,
                                                    hparams.lm_use_relu, 'pre_softmax_proj', norm_type=norm_type)
            input_dim = hparams.lm_lstm_cells
            var_top = dcommon_layers.get_sharded_weights(hparams.vocab_size, input_dim)
            cur_logits = tf.matmul(tf.squeeze(projed_states, axis=[1, 2]), var_top, transpose_b=True)
          else:
            if hparams.decoder_use_lstm2 or hparams.decoder_proj_add:
              input_dim = hparams.decoder_lstm_cells + hparams.lm_lstm_cells
            else:
              input_dim = 2 * hparams.decoder_lstm_cells + hparams.lm_lstm_cells
            var_top = dcommon_layers.get_sharded_weights(hparams.vocab_size, input_dim)
            cur_logits = tf.matmul(tf.concat([lm_gated_states, am_states], axis=-1),
                                   var_top, transpose_b=True)
        elif hparams.linho_fusion_type == "add":
          input_dim = hparams.decoder_lstm_cells
          var_top = dcommon_layers.get_sharded_weights(hparams.vocab_size, input_dim)

          am_states = tf.expand_dims(tf.expand_dims(am_states, 1), 1)
          am_states = dcommon_layers.normal_conv1d(am_states, hparams.lm_lstm_cells,
                                                  hparams.lm_use_relu, 'am_states_proj', norm_type=norm_type)
          am_states = tf.squeeze(am_states, axis=[1, 2])
          cur_logits = tf.matmul(tf.add(am_states, lm_gated_states), var_top, transpose_b=True)
    else:
      # Cold fusion part. to acquire the current logits.
      with tf.variable_scope("cold_fusion"):
        # formulation (4a)
        if hparams.cold_fusion_type == "prob_proj":
          lm_cur_logits = lm_cur_logits - tf.expand_dims(tf.reduce_max(lm_cur_logits, axis=-1), axis=-1)
          lm_cur_logits = tf.expand_dims(tf.expand_dims(lm_cur_logits, 1), 1)
          h_lm = dcommon_layers.normal_conv1d(lm_cur_logits, hparams.lm_lstm_cells,
                                              hparams.lm_use_relu, "lm_logits_proj", norm_type=norm_type)
        elif hparams.cold_fusion_type == "lstm_state":
          h_lm = tf.expand_dims(tf.expand_dims(lm_updated_states[-1].h, axis=1), axis=1)

        # formulation (4b)
        am_states = tf.expand_dims(tf.expand_dims(am_states, 1), 1)
        concated_states = tf.concat([h_lm, am_states], axis=-1)
        pre_g = dcommon_layers.normal_conv1d(concated_states, hparams.lm_lstm_cells,
                                           False, "gate_proj", norm_type=norm_type)
        g = tf.nn.sigmoid(pre_g)

        # formulation (4c)
        s_cf = tf.concat([am_states, g * h_lm], axis=-1)

        # formulation (4d)
        # cur_logits = dcommon_layers.normal_conv1d(s_cf, hparams.vocab_size,
        #                                           hparams.lm_use_relu, "final_logits_proj",
        #                                           norm_type=norm_type)
        # cur_logits = tf.squeeze(cur_logits, axis=[1,2])
        s_cf = tf.squeeze(s_cf, axis=[1,2])
        input_dim = 2 * hparams.decoder_lstm_cells + hparams.lm_lstm_cells
        var_top = dcommon_layers.get_sharded_weights(hparams.vocab_size, input_dim)
        cur_logits = tf.matmul(s_cf, var_top, transpose_b=True)

  return cur_logits, lm_updated_states



##### HUITING #####
@registry.register_hparams
def rna_huiting_4gpu_0220_exp2():
  hparams = rna_huiting_4gpu_0213_base()
  hparams.learning_rate = 0.2
  return hparams

@registry.register_hparams
def rna_huiting_4gpu_0220_exp1():
  hparams = rna_huiting_4gpu_0213_base()
  hparams.learning_rate_warmup_steps = 16000
  return hparams

@registry.register_hparams
def rna_huiting_4gpu_0213_base():
  hparams = rna_hkust_4gpu_0203_bi_base()
  hparams.lstm_pooling_layers = [2]
  hparams.tdnn_stride = 2
  hparams.frontend_cnn_layers = 1
  hparams.additional_module = "mu33"
  hparams.batch_size = 20000

  hparams.vocab_size = 4765
  hparams.boundaries = [146, 158, 167, 175, 182, 188, 195, 201, 207, 214,
                        221, 228, 235, 244, 253, 264, 278, 297, 328]
  hparams.min_length = 50
  hparams.max_length = 1400
  hparams.feat_dim = 87
  return hparams

#####################
##### DATA_745H #####
#####################
@registry.register_hparams
def rna_data745h_4gpu_0227_lm7():
  hparams = rna_data745h_4gpu_0227_lm()
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 6000
  hparams.rna_dir = "./egs/data_745h/exp/scp_data745h_characters_rna/rna_data745h_4gpu_0227_exp8/models"
  hparams.ls_lambda = 0.3
  return hparams

@registry.register_hparams
def rna_data745h_4gpu_0227_lm6():
  hparams = rna_data745h_4gpu_0227_lm()
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 6000
  return hparams

@registry.register_hparams
def rna_data745h_4gpu_0227_lm4():
  hparams = rna_data745h_4gpu_0227_lm()
  hparams.using_confidence_penalty = False
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 6000
  hparams.rna_dir = "./egs/data_745h/exp/scp_data745h_characters_rna/rna_data745h_4gpu_0227_exp8/models"
  return hparams


@registry.register_hparams
def rna_data745h_4gpu_0227_lm3():
  hparams = rna_data745h_4gpu_0227_lm()
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 12000
  return hparams

@registry.register_hparams
def rna_data745h_4gpu_0227_lm2():
  hparams = rna_data745h_4gpu_0227_lm()
  hparams.using_confidence_penalty = False
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 6000
  return hparams

@registry.register_hparams
def rna_data745h_4gpu_0227_lm1():
  hparams = rna_data745h_4gpu_0227_lm()
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 6000
  return hparams

@registry.register_hparams
def rna_data745h_4gpu_0227_lm():
  hparams = rna_data745h_4gpu_0227_exp6()
  hparams.lm_fusion = True
  hparams.load_trained_lm_model = True
  hparams.load_trained_rna_model = True
  hparams.linho_fusion = True
  hparams.lm_dir = "./egs/data_745h/exp/lm_dir/data745h_char_lm_0314_exp7/models"
  hparams.rna_dir = "./egs/data_745h/exp/scp_data745h_characters_rna/rna_data745h_4gpu_0227_exp6/models"
  hparams.lm_lstm_cells = 640
  return hparams

@registry.register_hparams
def rna_data745h_4gpu_0227_exp9():
  hparams = rna_data745h_4gpu_0227_base()
  hparams.learning_rate = 0.16
  hparams.learning_rate_warmup_steps = 20000
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.3
  hparams.use_data745h = True
  hparams.decoder_lstm_cells = 480
  return hparams

@registry.register_hparams
def rna_data745h_4gpu_0227_exp8():
  hparams = rna_data745h_4gpu_0227_base()
  hparams.learning_rate = 0.16
  hparams.learning_rate_warmup_steps = 25000
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.3
  hparams.use_data745h = True
  return hparams

@registry.register_hparams
def rna_data745h_4gpu_0227_exp7():
  hparams = rna_data745h_4gpu_0227_base()
  hparams.learning_rate = 0.16
  hparams.learning_rate_warmup_steps = 25000
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.1
  hparams.use_data745h = True
  return hparams

@registry.register_hparams
def rna_data745h_4gpu_0227_exp6():
  hparams = rna_data745h_4gpu_0227_base()
  hparams.learning_rate = 0.16
  hparams.learning_rate_warmup_steps = 25000
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  hparams.use_data745h = True
  return hparams

@registry.register_hparams
def rna_data745h_4gpu_0227_exp5():
  hparams = rna_data745h_4gpu_0227_base()
  hparams.learning_rate = 0.16
  hparams.learning_rate_warmup_steps = 25000
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  return hparams

@registry.register_hparams
def rna_data745h_4gpu_0227_exp4():
  hparams = rna_data745h_4gpu_0227_base()
  hparams.learning_rate = 0.2
  hparams.learning_rate_warmup_steps = 16000
  return hparams

@registry.register_hparams
def rna_data745h_4gpu_0227_exp3():
  hparams = rna_data745h_4gpu_0227_base()
  hparams.learning_rate = 0.16
  hparams.learning_rate_warmup_steps = 10000
  return hparams

@registry.register_hparams
def rna_data745h_4gpu_0227_exp2():
  hparams = rna_data745h_4gpu_0227_base()
  hparams.learning_rate = 0.16
  hparams.learning_rate_warmup_steps = 25000
  return hparams

@registry.register_hparams
def rna_data745h_4gpu_0227_exp1():
  hparams = rna_data745h_4gpu_0227_base()
  hparams.learning_rate = 0.16
  return hparams

@registry.register_hparams
def rna_data745h_4gpu_0227_base():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.vocab_size = 4623
  hparams.boundaries = [102, 132, 156, 177, 198, 218, 238, 257, 277, 298,
                        319, 342, 366, 392, 421, 454, 496, 550, 634]
  hparams.min_length = 26
  hparams.max_length = 3174
  hparams.feat_dim = 42
  hparams.learning_rate = 0.1
  return hparams

#################
##### HKUST #####
#################

# last exploration of layer norm and relu
@registry.register_hparams
def rna_hkust_4gpu_0227_exp15():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp15_2():
  hparams = rna_hkust_4gpu_0227_exp15()
  hparams.encoder_tdnn_use_relu = False
  hparams.encoder_tdnn_use_norm = True
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp15_2_2():
  hparams = rna_hkust_4gpu_0227_exp15()
  hparams.encoder_tdnn_use_relu = False
  hparams.encoder_tdnn_use_norm = True
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp15_3():
  hparams = rna_hkust_4gpu_0227_exp15()
  hparams.encoder_tdnn_use_relu = True
  hparams.encoder_tdnn_use_norm = False
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp15_3_2():
  hparams = rna_hkust_4gpu_0227_exp15()
  hparams.encoder_tdnn_use_relu = True
  hparams.encoder_tdnn_use_norm = False
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp15_4():
  hparams = rna_hkust_4gpu_0227_exp15()
  hparams.encoder_tdnn_use_relu = False
  hparams.encoder_tdnn_use_norm = False
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp15_4_2():
  hparams = rna_hkust_4gpu_0227_exp15()
  hparams.encoder_tdnn_use_relu = False
  hparams.encoder_tdnn_use_norm = False
  return hparams



# uni-direction
@registry.register_hparams
def rna_hkust_4gpu_0227_exp14():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  hparams.encoder_use_blstm = False
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp14_2():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  hparams.encoder_use_blstm = False
  hparams.encoder_lstm_cells = 480
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp14_3():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  hparams.encoder_use_blstm = False
  hparams.encoder_lstm_cells = 480
  hparams.tdnn_window_width = 3
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp14_4():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  hparams.encoder_use_blstm = False
  hparams.encoder_lstm_cells = 480
  hparams.tdnn_window_width = 5
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp14_5():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  hparams.encoder_use_blstm = False
  hparams.encoder_lstm_cells = 480
  hparams.tdnn_window_width = 5
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp14_6():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  hparams.encoder_use_blstm = False
  hparams.encoder_lstm_cells = 384
  hparams.tdnn_window_width = 5
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp14_7():
  hparams = rna_hkust_4gpu_0227_exp14_5()
  return hparams


# label smoothing
@registry.register_hparams
def rna_hkust_4gpu_0227_exp13():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_label_smoothing = True
  hparams.ls_lambda = 0.05
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_2():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_label_smoothing = True
  hparams.ls_lambda = 0.02
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_3():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_label_smoothing = True
  hparams.ls_type = "unigram"
  hparams.ls_lambda = 0.05
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_4():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_label_smoothing = True
  hparams.ls_type = "unigram"
  hparams.ls_lambda = 0.02
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_5():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_label_smoothing = True
  hparams.ls_lambda = 0.02
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_6():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_label_smoothing = True
  hparams.ls_lambda = 0.01
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_7():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_label_smoothing = True
  hparams.ls_lambda = 0.02
  hparams.ls_type = "unigram"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_8():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_label_smoothing = True
  hparams.ls_lambda = 0.01
  hparams.ls_type = "unigram"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_9():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_10():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.05
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_11():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.5
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_12():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_13():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.1
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_14():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.05
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_15():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.02
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_16():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.decoder_dropout = 0.1
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_17():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.decoder_dropout = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_18():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.decoder_dropout = 0.3
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_19():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.decoder_dropout = 0.1
  hparams.encoder_dropout = 0.1
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_20():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.decoder_dropout = 0.2
  hparams.encoder_dropout = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_21():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_22():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.15
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_23():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.15
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_24():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.25
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_25():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.25
  return hparams

#####
@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_26():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_27():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  hparams.decoder_dropout = 0.2
  hparams.encoder_dropout = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_28():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  hparams.decoder_dropout = 0.2
  hparams.encoder_dropout = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_29():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  hparams.encoder_dropout = 0.2
  return hparams


@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_30():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  hparams.encoder_dropout = 0.2
  return hparams


@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_31():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_label_smoothing = True
  hparams.ls_lambda = 0.2
  hparams.ls_type = "unigram"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp13_32():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_label_smoothing = True
  hparams.ls_lambda = 0.2
  hparams.ls_type = "unigram"
  return hparams

# vtln
@registry.register_hparams
def rna_hkust_4gpu_0227_exp12():
  hparams = rna_hkust_4gpu_0227_exp9()
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_2():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.learning_rate_warmup_steps = 20000
  hparams.learning_rate = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_3():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.learning_rate_warmup_steps = 16000
  hparams.learning_rate = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_4():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.learning_rate_warmup_steps = 16000
  hparams.learning_rate = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_5():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 8000
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_6():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 8000
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_7():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.learning_rate_warmup_steps = 16000
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_8():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.learning_rate_warmup_steps = 16000
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_9():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.learning_rate_warmup_steps = 25000
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_10():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.learning_rate_warmup_steps = 25000
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_11():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.learning_rate_warmup_steps = 20000
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_12():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.learning_rate_warmup_steps = 20000
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_13():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.learning_rate_warmup_steps = 20000
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_14():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.learning_rate_warmup_steps = 20000
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_15():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.learning_rate_warmup_steps = 20000
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_16():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.learning_rate_warmup_steps = 20000
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_17():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.learning_rate_warmup_steps = 20000
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  hparams.decoder_proj_add = True
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_18():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.learning_rate_warmup_steps = 20000
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  hparams.decoder_proj_add = True
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_19():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  hparams.encoder_use_blstm = False
  hparams.encoder_lstm_cells = 480
  hparams.tdnn_window_width = 5
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_20():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  hparams.encoder_use_blstm = False
  hparams.encoder_lstm_cells = 480
  hparams.tdnn_window_width = 5
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm():
  hparams = rna_hkust_4gpu_0227_exp12()
  hparams.lm_fusion = True
  hparams.load_trained_lm_model = True
  hparams.load_trained_rna_model = True
  hparams.linho_fusion = True
  hparams.lm_dir = "./egs/hkust/exp/lm_dir/hkust_char_lm_0304_exp1/models"
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0227_exp12_2/models"
  hparams.lm_lstm_cells = 320
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm1():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm2():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm3():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 4000
  hparams.learning_rate = 0.1
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm4():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 4000
  hparams.learning_rate = 0.1
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm5():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm6():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm7():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.16
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm8():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.16
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm9():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  hparams.optimizing_rna = True
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm10():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  hparams.optimizing_rna = True
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm11():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  hparams.optimizing_rna = True
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm12():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  hparams.optimizing_rna = True
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm13():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0227_exp12_13/models"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm14():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0227_exp12_13/models"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm15():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0227_exp12_13/models"
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm15_2():
  hparams = rna_hkust_4gpu_0227_exp12_lm15()
  hparams.lm_dir = "./egs/hkust/exp/lm_dir/hkust_char_lm_0314_exp2/models"
  hparams.lm_lstm_cells = 640
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm15_3():
  hparams = rna_hkust_4gpu_0227_exp12_lm15()
  hparams.lm_dir = "./egs/hkust/exp/lm_dir/hkust_char_lm_0314_exp2/models"
  hparams.lm_lstm_cells = 640
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm15_4():
  hparams = rna_hkust_4gpu_0227_exp12_lm15()
  hparams.lm_dir = "./egs/hkust/exp/lm_dir/hkust_char_lm_0314_exp2/models"
  hparams.lm_lstm_cells = 640
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm15_5():
  hparams = rna_hkust_4gpu_0227_exp12_lm15()
  hparams.lm_dir = "./egs/hkust/exp/lm_dir/hkust_char_lm_0314_exp2/models"
  hparams.lm_lstm_cells = 640
  hparams.lm_pre_softmax_proj = True
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm15_6():
  hparams = rna_hkust_4gpu_0227_exp12_lm15()
  hparams.lm_dir = "./egs/hkust/exp/lm_dir/hkust_char_lm_0314_exp2/models"
  hparams.lm_lstm_cells = 640
  hparams.lm_pre_softmax_proj = True
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm15_7():
  hparams = rna_hkust_4gpu_0227_exp12_lm15()
  hparams.lm_dir = "./egs/hkust/exp/lm_dir/hkust_char_lm_0314_exp2/models"
  hparams.lm_lstm_cells = 640
  hparams.lm_pre_softmax_proj = True
  hparams.lm_use_norm = True
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm15_8():
  hparams = rna_hkust_4gpu_0227_exp12_lm15()
  hparams.lm_dir = "./egs/hkust/exp/lm_dir/hkust_char_lm_0314_exp2/models"
  hparams.lm_lstm_cells = 640
  hparams.lm_pre_softmax_proj = True
  hparams.lm_use_norm = True
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm15_9():
  hparams = rna_hkust_4gpu_0227_exp12_lm15()
  hparams.lm_dir = "./egs/hkust/exp/lm_dir/hkust_char_lm_0314_exp2/models"
  hparams.lm_lstm_cells = 640
  hparams.lm_pre_softmax_proj = True
  hparams.lm_use_relu = False
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm15_10():
  hparams = rna_hkust_4gpu_0227_exp12_lm15()
  hparams.lm_dir = "./egs/hkust/exp/lm_dir/hkust_char_lm_0314_exp2/models"
  hparams.lm_lstm_cells = 640
  hparams.lm_pre_softmax_proj = True
  hparams.lm_use_relu = False
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm16():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0227_exp12_13/models"
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm17():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0227_exp12_13/models"
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm18():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0227_exp12_14/models"
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm18_2():
  hparams = rna_hkust_4gpu_0227_exp12_lm18()
  hparams.lm_dir = "./egs/hkust/exp/lm_dir/hkust_char_lm_0314_exp2/models"
  hparams.lm_lstm_cells = 640
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm18_3():
  hparams = rna_hkust_4gpu_0227_exp12_lm18()
  hparams.lm_dir = "./egs/hkust/exp/lm_dir/hkust_char_lm_0314_exp2/models"
  hparams.lm_lstm_cells = 640
  return hparams


@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm19():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0227_exp12_14/models"
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm20():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0227_exp12_16/models"
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm20_2():
  hparams = rna_hkust_4gpu_0227_exp12_lm20()
  hparams.lm_dir = "./egs/hkust/exp/lm_dir/hkust_char_lm_0314_exp2/models"
  hparams.lm_lstm_cells = 640
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm20_3():
  hparams = rna_hkust_4gpu_0227_exp12_lm20()
  hparams.lm_dir = "./egs/hkust/exp/lm_dir/hkust_char_lm_0314_exp2/models"
  hparams.lm_lstm_cells = 640
  return hparams


@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm21():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0227_exp12_16/models"
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  return hparams

# @registry.register_hparams
# def rna_hkust_4gpu_0227_exp12_lm22():
#   hparams = rna_hkust_4gpu_0227_exp12_lm()
#   hparams.learning_rate_warmup_steps = 6000
#   hparams.learning_rate = 0.1
#   hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0227_exp12_16/models"
#   hparams.using_confidence_penalty = True
#   hparams.ls_lambda = 0.2
#   hparams.lm_gated_input_encoder_output = True
#   return hparams
#
# @registry.register_hparams
# def rna_hkust_4gpu_0227_exp12_lm23():
#   hparams = rna_hkust_4gpu_0227_exp12_lm()
#   hparams.learning_rate_warmup_steps = 6000
#   hparams.learning_rate = 0.1
#   hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0227_exp12_16/models"
#   hparams.using_confidence_penalty = True
#   hparams.ls_lambda = 0.2
#   hparams.lm_gated_input_encoder_output = True
#   return hparams

# @registry.register_hparams
# def rna_hkust_4gpu_0227_exp12_lm24():
#   hparams = rna_hkust_4gpu_0227_exp12_lm()
#   hparams.learning_rate_warmup_steps = 6000
#   hparams.learning_rate = 0.1
#   hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0227_exp12_16/models"
#   hparams.using_confidence_penalty = True
#   hparams.ls_lambda = 0.2
#   hparams.decoder_proj_add = True
#   return hparams
#
# @registry.register_hparams
# def rna_hkust_4gpu_0227_exp12_lm25():
#   hparams = rna_hkust_4gpu_0227_exp12_lm()
#   hparams.learning_rate_warmup_steps = 6000
#   hparams.learning_rate = 0.1
#   hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0227_exp12_16/models"
#   hparams.using_confidence_penalty = True
#   hparams.ls_lambda = 0.2
#   hparams.decoder_proj_add = True
#   return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm26():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  hparams.encoder_use_blstm = False
  hparams.encoder_lstm_cells = 480
  hparams.tdnn_window_width = 5
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0227_exp12_19/models"
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm26_2():
  hparams = rna_hkust_4gpu_0227_exp12_lm26()
  hparams.lm_dir = "./egs/hkust/exp/lm_dir/hkust_char_lm_0314_exp2/models"
  hparams.lm_lstm_cells = 640
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm26_3():
  hparams = rna_hkust_4gpu_0227_exp12_lm26()
  hparams.lm_dir = "./egs/hkust/exp/lm_dir/hkust_char_lm_0314_exp2/models"
  hparams.lm_lstm_cells = 640
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm27():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  hparams.encoder_use_blstm = False
  hparams.encoder_lstm_cells = 480
  hparams.tdnn_window_width = 5
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0227_exp12_19/models"
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm28():
  hparams = rna_hkust_4gpu_0227_exp12_lm()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  hparams.encoder_use_blstm = False
  hparams.encoder_lstm_cells = 480
  hparams.tdnn_window_width = 5
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0227_exp12_20/models"
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm28_2():
  hparams = rna_hkust_4gpu_0227_exp12_lm28()
  hparams.lm_dir = "./egs/hkust/exp/lm_dir/hkust_char_lm_0314_exp2/models"
  hparams.lm_lstm_cells = 640
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp12_lm28_3():
  hparams = rna_hkust_4gpu_0227_exp12_lm28()
  hparams.lm_dir = "./egs/hkust/exp/lm_dir/hkust_char_lm_0314_exp2/models"
  hparams.lm_lstm_cells = 640
  return hparams

# lm_fusion
@registry.register_hparams
def rna_hkust_4gpu_0227_exp11():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.lm_fusion = True
  hparams.load_trained_lm_model = True
  hparams.load_trained_rna_model = True
  hparams.linho_fusion = True
  hparams.lm_dir = "./egs/hkust/exp/lm_dir/hkust_char_lm_0314_exp2/models"
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0227_exp13_21/models"
  hparams.lm_lstm_cells = 640
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  hparams.using_confidence_penalty = True
  hparams.ls_lambda = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp11_2():
  hparams = rna_hkust_4gpu_0227_exp11()
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp11_3():
  hparams = rna_hkust_4gpu_0227_exp11()
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp11_4():
  hparams = rna_hkust_4gpu_0227_exp11()
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0227_exp13_9/models"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp11_5():
  hparams = rna_hkust_4gpu_0227_exp11()
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0227_exp13_12/models"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp11_6():
  hparams = rna_hkust_4gpu_0227_exp11()
  hparams.encoder_use_blstm = False
  hparams.encoder_lstm_cells = 480
  hparams.tdnn_window_width = 5
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0227_exp14_5/models"
  hparams.lm_dir = "./egs/hkust/exp/lm_dir/hkust_char_lm_0314_exp2/models"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp11_7():
  hparams = rna_hkust_4gpu_0227_exp11_6()
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp11_8():
  hparams = rna_hkust_4gpu_0227_exp11_6()
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0227_exp14_7/models"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp11_9():
  hparams = rna_hkust_4gpu_0227_exp11_6()
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0227_exp14_7/models"
  return hparams

# ctc
@registry.register_hparams
def rna_hkust_4gpu_0227_exp10():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.load_trained_pinyin_ctc_model = True
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp10_2():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.load_trained_pinyin_ctc_model = True
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp10_3():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.load_trained_pinyin_ctc_model = True
  hparams.pinyin_ctc_dir = "./egs/hkust_ctc/exp/scp_hkust_pinyins_ctc/ctc_hkust_4gpu_0227_base/models2"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp10_4():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.load_trained_pinyin_ctc_model = True
  hparams.learning_rate_warmup_steps = 6000
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp10_5():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.load_trained_pinyin_ctc_model = True
  hparams.learning_rate_warmup_steps = 4000
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp10_6():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.load_trained_pinyin_ctc_model = True
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp10_7():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.load_trained_pinyin_ctc_model = True
  hparams.learning_rate_warmup_steps = 2500
  hparams.learning_rate = 0.1
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp10_8():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.load_trained_pinyin_ctc_model = True
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp10_9():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.load_trained_pinyin_ctc_model = True
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  return hparams


# additional module
@registry.register_hparams
def rna_hkust_4gpu_0227_exp9():
  hparams = rna_hkust_4gpu_0227_exp6()
  hparams.additional_module = "mu33"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp9_2():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.learning_rate_warmup_steps = 4000
  hparams.learning_rate = 0.1
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp9_3():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp9_4():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.1
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp9_5():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.16
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp9_6():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.learning_rate_warmup_steps = 6000
  hparams.learning_rate = 0.16
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp9_7():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.learning_rate_warmup_steps = 8000
  hparams.learning_rate = 0.16
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp9_8():
  hparams = rna_hkust_4gpu_0227_exp9()
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp9_9():
  hparams = rna_hkust_4gpu_0227_exp9()
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp9_11():
  hparams = rna_hkust_4gpu_0227_exp9()
  hparams.additional_module_layers = 2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp9_12():
  hparams = rna_hkust_4gpu_0227_exp6()
  hparams.additional_module = "conv_lstm"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp9_13():
  hparams = rna_hkust_4gpu_0227_exp6()
  hparams.additional_module = "conv_lstm"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp9_14():
  hparams = rna_hkust_4gpu_0227_exp6()
  hparams.additional_module = "res_cnn"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp9_15():
  hparams = rna_hkust_4gpu_0227_exp6()
  hparams.additional_module = "res_cnn"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp9_16():
  hparams = rna_hkust_4gpu_0227_exp6()
  hparams.additional_module = "glu"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp9_17():
  hparams = rna_hkust_4gpu_0227_exp6()
  hparams.additional_module = "glu"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp9_18():
  hparams = rna_hkust_4gpu_0227_exp6()
  hparams.additional_module = "res_cnn"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp9_19():
  hparams = rna_hkust_4gpu_0227_exp6()
  hparams.additional_module = "res_cnn"
  return hparams


# subsample2 (conv + lstm-pooling)
@registry.register_hparams
def rna_hkust_4gpu_0227_exp8():
  hparams = rna_hkust_4gpu_0227_bi_base()
  hparams.frontend_cnn_layers = 3
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp8_2():
  hparams = rna_hkust_4gpu_0227_bi_base()
  hparams.frontend_cnn_layers = 3
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp8_2_2():
  hparams = rna_hkust_4gpu_0227_bi_base()
  hparams.frontend_cnn_layers = 3
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp7():
  hparams = rna_hkust_4gpu_0227_bi_base()
  hparams.lstm_pooling_layers = [2]
  hparams.frontend_cnn_layers = 2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp7_2():
  hparams = rna_hkust_4gpu_0227_bi_base()
  hparams.lstm_pooling_layers = [2]
  hparams.frontend_cnn_layers = 2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp7_2_2():
  hparams = rna_hkust_4gpu_0227_bi_base()
  hparams.lstm_pooling_layers = [2]
  hparams.frontend_cnn_layers = 2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp6():
  hparams = rna_hkust_4gpu_0227_bi_base()
  hparams.lstm_pooling_layers = [2]
  hparams.tdnn_stride = 2
  hparams.frontend_cnn_layers = 1
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp6_2():
  hparams = rna_hkust_4gpu_0227_bi_base()
  hparams.lstm_pooling_layers = [2]
  hparams.tdnn_stride = 2
  hparams.frontend_cnn_layers = 1
  return hparams

# subsample1 (different lstm-pooling setting)
@registry.register_hparams
def rna_hkust_4gpu_0227_exp5():
  hparams = rna_hkust_4gpu_0227_bi_base()
  hparams.lstm_pooling_layers = [1,2,3]
  hparams.tdnn_stride = 2
  hparams.frontend_cnn_layers = 0
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp5_2():
  hparams = rna_hkust_4gpu_0227_bi_base()
  hparams.lstm_pooling_layers = [1,2,3]
  hparams.tdnn_stride = 2
  hparams.frontend_cnn_layers = 0
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp5_3():
  hparams = rna_hkust_4gpu_0227_bi_base()
  hparams.lstm_pooling_layers = [1,2,3]
  hparams.tdnn_stride = 2
  hparams.frontend_cnn_layers = 0
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp4():
  hparams = rna_hkust_4gpu_0227_bi_base()
  hparams.lstm_pooling_layers = [1,2]
  hparams.tdnn_stride = 3
  hparams.frontend_cnn_layers = 0
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp4_2():
  hparams = rna_hkust_4gpu_0227_bi_base()
  hparams.lstm_pooling_layers = [1,2]
  hparams.first_pooling_width3 = True
  hparams.tdnn_stride = 2
  hparams.frontend_cnn_layers = 0
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp4_2_2():
  hparams = rna_hkust_4gpu_0227_bi_base()
  hparams.lstm_pooling_layers = [1,2]
  hparams.first_pooling_width3 = True
  hparams.tdnn_stride = 2
  hparams.frontend_cnn_layers = 0
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp3():
  hparams = rna_hkust_4gpu_0227_bi_base()
  hparams.lstm_pooling_layers = [1,2]
  hparams.tdnn_stride = 2
  hparams.frontend_cnn_layers = 0
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp3_2():
  hparams = rna_hkust_4gpu_0227_bi_base()
  hparams.lstm_pooling_layers = [1,2]
  hparams.tdnn_stride = 2
  hparams.frontend_cnn_layers = 0
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp2():
  hparams = rna_hkust_4gpu_0227_bi_base()
  hparams.lstm_pooling_layers = [2]
  hparams.tdnn_stride = 3
  hparams.frontend_cnn_layers = 0
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp2_2():
  hparams = rna_hkust_4gpu_0227_bi_base()
  hparams.lstm_pooling_layers = [2]
  hparams.first_pooling_width3 = True
  hparams.tdnn_stride = 2
  hparams.frontend_cnn_layers = 0
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp2_2_2():
  hparams = rna_hkust_4gpu_0227_bi_base()
  hparams.lstm_pooling_layers = [2]
  hparams.first_pooling_width3 = True
  hparams.tdnn_stride = 2
  hparams.frontend_cnn_layers = 0
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp2_2_3():
  hparams = rna_hkust_4gpu_0227_bi_base()
  hparams.lstm_pooling_layers = [2]
  hparams.first_pooling_width3 = True
  hparams.tdnn_stride = 2
  hparams.frontend_cnn_layers = 0
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp1_3():
  hparams = rna_hkust_4gpu_0227_bi_base()
  hparams.lstm_pooling_layers = [2]
  hparams.tdnn_stride = 2
  hparams.frontend_cnn_layers = 0
  return hparams


@registry.register_hparams
def rna_hkust_4gpu_0227_exp1_2():
  hparams = rna_hkust_4gpu_0227_bi_base()
  hparams.lstm_pooling_layers = [2]
  hparams.tdnn_stride = 2
  hparams.frontend_cnn_layers = 0
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_exp1():
  hparams = rna_hkust_4gpu_0227_bi_base()
  hparams.lstm_pooling_layers = [2]
  hparams.tdnn_stride = 2
  hparams.frontend_cnn_layers = 0
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0227_bi_base():
  hparams = rna_hkust_4gpu_0128_bi_base()
  hparams.tdnn_window_width = 1
  hparams.tdnn_stride = 1
  hparams.norm_type = "layer"
  hparams.batch_size = 20000
  hparams.hidden_size = 100
  hparams.learning_rate = 0.16
  return hparams

####################
# vtln
@registry.register_hparams
def rna_hkust_4gpu_0203_exp15_repair1_repeat3():
  hparams = rna_hkust_4gpu_0203_exp15()
  hparams.tdnn_window_width = 5
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp15_repair1_repeat2():
  hparams = rna_hkust_4gpu_0203_exp15()
  hparams.tdnn_window_width = 5
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp15_repair1_repeat1():
  hparams = rna_hkust_4gpu_0203_exp15()
  hparams.tdnn_window_width = 5
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp15_repair1():
  hparams = rna_hkust_4gpu_0203_exp15()
  hparams.tdnn_window_width = 5
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp15_repeat3():
  hparams = rna_hkust_4gpu_0203_exp15()
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp15_repeat2():
  hparams = rna_hkust_4gpu_0203_exp15()
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp15_repeat1():
  hparams = rna_hkust_4gpu_0203_exp15()
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp15_lm2():
  hparams = rna_hkust_4gpu_0203_exp15_lm()
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp15_lm():
  hparams = rna_hkust_4gpu_0203_exp15()
  hparams.lm_fusion = True
  hparams.load_trained_lm_model = True
  hparams.load_trained_rna_model = True
  hparams.linho_fusion = True
  hparams.lm_dir = "./egs/hkust/exp/lm_dir/hkust_char_lm_0203_exp1/models"
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0203_exp15/models"
  hparams.lm_lstm_cells = 320
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp15():
  hparams = rna_hkust_4gpu_0203_bi_base()
  hparams.lstm_pooling_layers = [2]
  hparams.tdnn_stride = 2
  hparams.frontend_cnn_layers = 1
  hparams.additional_module = "mu33"
  hparams.batch_size = 20000
  return hparams

################################
# language model
@registry.register_hparams
def rna_hkust_4gpu_0203_exp14():
  hparams = rna_hkust_4gpu_0203_bi_base()
  hparams.lstm_pooling_layers = [2]
  hparams.tdnn_stride = 2
  hparams.frontend_cnn_layers = 1
  hparams.additional_module = "mu33"
  hparams.lm_fusion = True
  hparams.load_trained_lm_model = True
  hparams.load_trained_rna_model = True
  hparams.linho_fusion = True
  hparams.lm_dir = "./egs/hkust/exp/lm_dir/hkust_char_lm_0203_exp1/models"
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0203_exp11/models"
  hparams.lm_lstm_cells = 320
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp14_repair1():
  hparams = rna_hkust_4gpu_0203_exp14()
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0203_exp11_repeat1/models"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp14_repair2():
  hparams = rna_hkust_4gpu_0203_exp14()
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0203_exp11_repeat1/models"
  hparams.learning_rate_warmup_steps = 6000
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp14_repair2_repair1():
  hparams = rna_hkust_4gpu_0203_exp14()
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0203_exp11_repeat1/models"
  hparams.learning_rate_warmup_steps = 6000
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp14_repair3():
  hparams = rna_hkust_4gpu_0203_exp14()
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0203_exp11_repair1/models"
  hparams.batch_size = 20000
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp14_repair4():
  hparams = rna_hkust_4gpu_0203_exp14()
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0203_exp10_repair2/models"
  hparams.additional_module = "mu31"
  hparams.batch_size = 20000
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp14_repair5():
  hparams = rna_hkust_4gpu_0203_exp14()
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0203_exp11_repair2/models"
  hparams.batch_size = 20000
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp14_repair6():
  hparams = rna_hkust_4gpu_0203_exp14()
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0203_exp11_repair2/models"
  hparams.batch_size = 20000
  hparams.linho_fusion_type = "add"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp14_repair7():
  hparams = rna_hkust_4gpu_0203_exp14()
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0203_exp11_repair2/models"
  hparams.batch_size = 20000
  hparams.linho_fusion = False
  hparams.cold_fusion_type = "prob_proj"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp14_repair7_repair1():
  hparams = rna_hkust_4gpu_0203_exp14_repair7()
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp14_repair8():
  hparams = rna_hkust_4gpu_0203_exp14()
  hparams.rna_dir = "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_4gpu_0203_exp11_repair2/models"
  hparams.batch_size = 20000
  hparams.linho_fusion = False
  hparams.cold_fusion_type = "lstm_state"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp14_repair8_repair1():
  hparams = rna_hkust_4gpu_0203_exp14_repair8()
  return hparams

#######################
# attention
@registry.register_hparams
def rna_hkust_4gpu_0203_exp13():
  hparams = rna_hkust_4gpu_0203_bi_base()
  hparams.lstm_pooling_layers = [2]
  hparams.encoder_output_tdnn = False
  hparams.frontend_cnn_layers = 1
  hparams.use_attention = True
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp13_repair1():
  hparams = rna_hkust_4gpu_0203_exp13()
  hparams.learning_rate = 0.2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp13_repair2():
  hparams = rna_hkust_4gpu_0203_exp13()
  hparams.clip_grad_norm = 10.0
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp13_repair3():
  hparams = rna_hkust_4gpu_0203_exp13()
  hparams.clip_grad_norm = 10.0
  hparams.encoder_output_tdnn = True
  hparams.tdnn_stride = 2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp13_repair4():
  hparams = rna_hkust_4gpu_0203_exp13()
  hparams.clip_grad_norm = 10.0
  hparams.att_type = "content"
  hparams.encoder_output_tdnn = True
  hparams.tdnn_stride = 2
  hparams.batch_size = 20000
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp13_repair5():
  hparams = rna_hkust_4gpu_0203_exp13()
  hparams.att_type = "content"
  hparams.encoder_output_tdnn = True
  hparams.tdnn_stride = 2
  hparams.batch_size = 20000
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp13_repair6():
  hparams = rna_hkust_4gpu_0203_exp13()
  hparams.att_type = "content"
  hparams.encoder_output_tdnn = True
  hparams.tdnn_stride = 2
  hparams.batch_size = 20000
  hparams.additional_module = "mu33"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp13_repair7():
  hparams = rna_hkust_4gpu_0203_exp13()
  hparams.att_type = "content"
  hparams.encoder_output_tdnn = True
  hparams.tdnn_stride = 2
  hparams.batch_size = 20000
  hparams.additional_module = "mu33"
  hparams.clip_grad_norm = 10.0
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp13_repair8():
  hparams = rna_hkust_4gpu_0203_exp13()
  hparams.att_type = "content"
  hparams.encoder_output_tdnn = True
  hparams.tdnn_stride = 2
  hparams.batch_size = 20000
  hparams.additional_module = "mu33"
  hparams.clip_grad_norm = 30.0
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp13_repair9():
  hparams = rna_hkust_4gpu_0203_exp13()
  hparams.att_type = "content"
  hparams.encoder_output_tdnn = True
  hparams.tdnn_stride = 2
  hparams.batch_size = 20000
  hparams.additional_module = "mu33"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp13_repair10():
  hparams = rna_hkust_4gpu_0203_exp13()
  hparams.att_type = "content"
  hparams.encoder_output_tdnn = True
  hparams.tdnn_stride = 2
  hparams.batch_size = 20000
  hparams.additional_module = "mu33"
  hparams.att_window_width = 10
  return hparams

######################
# additional module
@registry.register_hparams
def rna_hkust_4gpu_0203_exp12():
  hparams = rna_hkust_4gpu_0203_exp4()
  hparams.additional_module = "conv_lstm"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp11():
  hparams = rna_hkust_4gpu_0203_exp4()
  hparams.additional_module = "mu33"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp11_repeat1():
  hparams = rna_hkust_4gpu_0203_exp11()
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp11_repeat2():
  hparams = rna_hkust_4gpu_0203_exp11()
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp11_repair1():
  hparams = rna_hkust_4gpu_0203_exp11()
  hparams.batch_size = 20000
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp11_repair2():
  hparams = rna_hkust_4gpu_0203_exp11()
  hparams.batch_size = 20000
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp11_repair3():
  hparams = rna_hkust_4gpu_0203_exp11()
  hparams.batch_size = 20000
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp11_repair4():
  hparams = rna_hkust_4gpu_0203_exp11()
  hparams.batch_size = 10000
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp11_repair5():
  hparams = rna_hkust_4gpu_0203_exp11()
  hparams.batch_size = 20000
  hparams.clip_grad_norm = 10.0
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp11_repair6_repair1():
  hparams = rna_hkust_4gpu_0203_exp11()
  hparams.batch_size = 20000
  hparams.tdnn_window_width = 5
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp11_repair6_repair2():
  hparams = rna_hkust_4gpu_0203_exp11()
  hparams.batch_size = 20000
  hparams.tdnn_window_width = 5
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp10():
  hparams = rna_hkust_4gpu_0203_exp4()
  hparams.additional_module = "mu31"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp10_repeat1():
  hparams = rna_hkust_4gpu_0203_exp10()
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp10_repeat2():
  hparams = rna_hkust_4gpu_0203_exp10()
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp10_repair1():
  hparams = rna_hkust_4gpu_0203_exp10()
  hparams.decoder_use_lstm2 = True
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp10_repair2():
  hparams = rna_hkust_4gpu_0203_exp10()
  hparams.batch_size = 20000
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp10_repair3():
  hparams = rna_hkust_4gpu_0203_exp10()
  hparams.batch_size = 20000
  return hparams

# subsampling
@registry.register_hparams
def rna_hkust_4gpu_0203_exp9():
  hparams = rna_hkust_4gpu_0203_bi_base()
  hparams.lstm_pooling_layers = [2]
  hparams.tdnn_stride = 2
  hparams.frontend_cnn_layers = 2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp8():
  hparams = rna_hkust_4gpu_0203_bi_base()
  hparams.lstm_pooling_layers = [2]
  hparams.tdnn_stride = 2
  hparams.conv_time_stride3 = True
  hparams.frontend_cnn_layers = 1
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp7():
  hparams = rna_hkust_4gpu_0203_bi_base()
  hparams.lstm_pooling_layers = [2]
  hparams.tdnn_stride = 2
  hparams.num_frame_stacking = 5
  hparams.num_frame_striding = 3
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp6():
  hparams = rna_hkust_4gpu_0203_bi_base()
  hparams.frontend_cnn_layers = 3
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp5():
  hparams = rna_hkust_4gpu_0203_bi_base()
  hparams.lstm_pooling_layers = [2]
  hparams.frontend_cnn_layers = 2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp4():
  hparams = rna_hkust_4gpu_0203_bi_base()
  hparams.lstm_pooling_layers = [2]
  hparams.tdnn_stride = 2
  hparams.frontend_cnn_layers = 1
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp3():
  hparams = rna_hkust_4gpu_0203_bi_base()
  hparams.lstm_pooling_layers = [1,2]
  hparams.tdnn_stride = 2
  hparams.frontend_cnn_layers = 0
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp2():
  hparams = rna_hkust_4gpu_0203_bi_base()
  hparams.batch_size = 20000
  hparams.frontend_cnn_layers = 2
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_exp1():
  hparams = rna_hkust_4gpu_0203_bi_base()
  hparams.lstm_pooling_layers = [2]
  hparams.tdnn_stride = 2
  hparams.batch_size = 20000
  hparams.frontend_cnn_layers = 0
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0203_bi_base():
  hparams = rna_hkust_4gpu_0128_bi_base()
  hparams.tdnn_window_width = 1
  hparams.tdnn_stride = 1
  hparams.norm_type = "layer"
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0128_bi_base():
  hparams = rna_hkust_4gpu_0120_base()
  hparams.encoder_lstm_cells = 640
  hparams.encoder_lstm_layers = 4
  hparams.decoder_lstm_cells = 320
  hparams.encoder_use_blstm = True
  hparams.hidden_size = hparams.encoder_lstm_cells
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0128_base():
  hparams = rna_hkust_4gpu_0120_base()
  hparams.encoder_lstm_cells = 480
  hparams.encoder_lstm_layers = 4
  hparams.decoder_lstm_cells = 320
  return hparams

@registry.register_hparams
def rna_hkust_4gpu_0120_base():
  hparams = rna_hkust_0103_4gpu_base()
  hparams.learning_rate = 0.4
  hparams.learning_rate_warmup_steps = 10000
  hparams.vocab_size = 3674
  hparams.symbol_modality_num_shards = 1
  hparams.boundaries = [149, 190,  221, 248, 272, 294, 316, 337, 358, 379,
                        401, 424, 450, 477, 509, 545, 590, 651, 748]
  hparams.decoder_lstm_concat = True
  return hparams

@registry.register_hparams
def rna_hkust_0120_base():
  hparams = rna_hkust_0103_base()
  hparams.vocab_size = 3674
  hparams.symbol_modality_num_shards = 1
  hparams.decoder_lstm_concat = True
  hparams.boundaries = [149, 190,  221, 248, 272, 294, 316, 337, 358, 379,
                        401, 424, 450, 477, 509, 545, 590, 651, 748]
  return hparams

@registry.register_hparams
def rna_hkust_0103_4gpu_base():
  hparams = rna_hkust_base()
  hparams.batch_size = 40000
  hparams.learning_rate = 0.2
  hparams.learning_rate_warmup_steps = 6000
  hparams.hidden_size = hparams.encoder_lstm_cells

  hparams.frontend_cnn_layers = 2
  hparams.use_cnn_frontend = True
  hparams.encoder_output_tdnn = True
  hparams.shared_embedding_and_softmax_weights = False
  hparams.encoder_use_NiN = True
  hparams.decoder_lstm_cells = 256
  # Very important
  hparams.daisy_chain_variables = False
  return hparams

@registry.register_hparams
def rna_hkust_0103_base():
  hparams = rna_hkust_base()
  hparams.batch_size = 40000
  hparams.learning_rate = 0.2
  hparams.learning_rate_warmup_steps = 16000
  hparams.hidden_size = hparams.encoder_lstm_cells

  hparams.frontend_cnn_layers = 2
  hparams.use_cnn_frontend = True
  hparams.encoder_output_tdnn = True
  hparams.shared_embedding_and_softmax_weights = False
  hparams.encoder_use_NiN = True
  hparams.decoder_lstm_cells = 256
  return hparams

@registry.register_hparams
def rna_hkust_base():
  hparams = rna_base()
  hparams.encoder_add_residual = False
  hparams.encoder_lstm_cells = 512
  hparams.encoder_lstm_layers = 3
  hparams.decoder_lstm_cells = 512
  hparams.num_frame_stacking = 16
  hparams.num_frame_striding = 8

  hparams.hidden_size = hparams.encoder_lstm_cells
  hparams.learning_rate = 0.2
  hparams.dropout = 0.0
  hparams.learning_rate_warmup_steps = 6000

  # Hkust setup
  hparams.batch_size = 20000
  hparams.boundaries = [94, 125, 149, 173, 194, 214, 234, 252, 273, 293,
                        314, 335, 358, 383, 411, 444, 485, 540, 629]
  hparams.max_length = 1200
  hparams.min_length = 20

  hparams.shared_embedding_and_softmax_weights = True
  hparams.symbol_modality_num_shards = 10
  hparams.feat_dim = 120
  hparams.vocab_size = 4349

  return hparams

@registry.register_hparams
def rna_wsj_base():
  hparams = rna_base()
  hparams.encoder_add_residual = False
  hparams.encoder_lstm_cells = 512
  hparams.encoder_lstm_layers = 3
  hparams.decoder_lstm_cells = 512
  hparams.num_frame_stacking = 16
  hparams.num_frame_striding = 8

  hparams.hidden_size = hparams.encoder_lstm_cells
  hparams.learning_rate_warmup_steps = 6000
  hparams.dropout = 0.0
  hparams.learning_rate = 0.2

  # Wsj setup
  hparams.batch_size = 64000
  hparams.boundaries = [309, 370, 415, 453, 489, 521, 550, 580, 608, 635,
                        663, 690, 715, 741, 768, 794, 820, 844, 871, 898,
                        927, 957, 990, 1025, 1063, 1109, 1161, 1234, 1343]
  hparams.max_length = 1703
  hparams.min_length = 186

  hparams.shared_embedding_and_softmax_weights = False
  hparams.symbol_modality_num_shards = 1
  hparams.feat_dim = 240
  hparams.vocab_size = 32

  return hparams

@registry.register_hparams
def rna_base():
  """Set of hyper-parameters."""
  hparams = common_hparams.basic_params1()

  # Self-defined hparams
  # frontend part
  # cnn
  hparams.add_hparam("use_cnn_frontend", False)
  # hparams.add_hparam("cnn_frontend_scheme", 1)
  hparams.add_hparam("frontend_cnn_layers", 2)
  hparams.add_hparam("frontend_cnn_filters", 64)
  # hparams.add_hparam("additional_layers", 1)
  hparams.add_hparam("additional_module", None)
  hparams.add_hparam("additional_module_layers", 1)
  hparams.add_hparam("conv_time_stride3", False)
  # stack & stride
  hparams.add_hparam("num_frame_stacking", 8)
  hparams.add_hparam("num_frame_striding", 3)
  # lstm part
  hparams.add_hparam("encoder_lstm_layers", 3)
  hparams.add_hparam("decoder_lstm_layers", 1)
  hparams.add_hparam("encoder_lstm_cells", 512)
  hparams.add_hparam("encoder_lstm_projs", None)
  hparams.add_hparam("decoder_lstm_cells", 256)
  # hparams.add_hparam("decoder_lstm_projs", 512)
  # extra lstm setup
  hparams.add_hparam("encoder_use_blstm", False)
  hparams.add_hparam("encoder_add_residual", False)
  hparams.add_hparam("encoder_use_NiN", False)
  hparams.add_hparam("encoder_use_ln_lstm", False)
  hparams.add_hparam("is_debug", False)
  hparams.add_hparam("first_pooling_width3", False)
  # dropout
  hparams.add_hparam("encoder_dropout", 0.0)
  hparams.add_hparam("decoder_dropout", 0.0)
  hparams.add_hparam("lm_dropout", 0.0)

  # tdnn
  hparams.add_hparam("encoder_output_tdnn", False)
  hparams.add_hparam("tdnn_window_width", 3)
  hparams.add_hparam("tdnn_stride", 2)
  hparams.add_hparam("lstm_pooling_layers", [4])
  # attention
  hparams.add_hparam("use_attention", False)
  hparams.add_hparam("att_window_width", 5)
  hparams.add_hparam("att_type", "dot_product")

  # extra
  # the vocabulary contains (<pad>:0; chars_in_dict; <blank>:vocab_size-1)
  hparams.add_hparam("vocab_size", 32)
  hparams.add_hparam("use_rna_model", True)
  hparams.add_hparam("feat_dim", 240)
  # hparams.add_hparam("newbob_decay_time", 0)
  hparams.add_hparam("use_newbob", False)
  hparams.add_hparam("decoder_lstm2_cells", 320)
  hparams.add_hparam("decoder_lstm2_layers", 1)
  hparams.add_hparam("decoder_lstm_concat", False)
  hparams.add_hparam("decoder_use_lstm2", False)
  hparams.add_hparam("mask_blank_emb", False)
  hparams.add_hparam("start_with_blank", False)
  hparams.add_hparam("lstm_initializer", None)

  # lm relatives
  hparams.add_hparam("lm_fusion", False)
  hparams.add_hparam("lm_lstm_cells", 320)
  hparams.add_hparam("lm_lstm_layers", 1)

  hparams.add_hparam("linho_fusion", False)
  hparams.add_hparam("load_trained_rna_model", False)
  hparams.add_hparam("load_trained_lm_model", False)
  hparams.add_hparam("load_trained_lm_as_decoder", False)
  hparams.add_hparam("lm_dir", "./egs/hkust/exp/lm_dir/hkust_char_lm_0122_exp1/models")
  hparams.add_hparam("rna_dir", "./egs/hkust/exp/scp_hkust_characters_rna/rna_hkust_0122_exp4/models")
  hparams.add_hparam("optimizing_rna", False)

  hparams.add_hparam("linho_fusion_type", "concat")
  hparams.add_hparam("cold_fusion_type", "prob_proj")
  hparams.add_hparam("decoder_proj_add", False)
  hparams.add_hparam("lm_pre_softmax_proj", False)
  hparams.add_hparam("lm_use_norm", False)
  hparams.add_hparam("lm_use_relu", True)
  hparams.add_hparam("encoder_tdnn_use_norm", True)
  hparams.add_hparam("encoder_tdnn_use_relu", True)

  # pinyin ctc model initialization
  hparams.add_hparam("load_trained_pinyin_ctc_model", False)
  hparams.add_hparam("pinyin_ctc_dir", "./egs/hkust_ctc/exp/scp_hkust_pinyins_ctc/ctc_hkust_4gpu_0227_base/models")

  # label smoothing
  hparams.add_hparam("using_label_smoothing", False)
  hparams.add_hparam("ls_type", 'uniform')
  hparams.add_hparam("ls_lambda", 0.05)
  hparams.add_hparam("unigram_txt_dir", "egs/hkust/data_char_new/stat_freq/unigram.txt")
  hparams.add_hparam("using_confidence_penalty", False)

  # for data745h
  hparams.add_hparam("use_data745h", False)

  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 4000
  hparams.initializer_gain = 1.0
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.num_sampled_classes = 0
  hparams.label_smoothing = 0
  hparams.dropout = 0.0
  # hparams.factored_logits = False
  hparams.multiply_embedding_mode = "sqrt_depth"
  # hparams.sampling_method = "argmax"
  hparams.norm_type = "batch"

  return hparams