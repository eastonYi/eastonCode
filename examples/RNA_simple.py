#!/usr/bin/env python
# encoding: utf-8

from tensor2tensor.layers import common_layers
from tensor2tensor.layers import dcommon_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

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
    "lstm1_states": ([batch_size, decode_lstm_cells],
                     [batch_size, decode_lstm_cells])
    logits: The concatanated logits of previous steps. [batch_size, None, vocab_size]
    encoder_outputs: The outputs of encoder.
    hparams: The set of hyper-parameters.

  Returns:
    i, next_ids, next_states, logits.

  """

  # Get the previous states of lstms.
  lstm_states = all_lstm_states["lstm_states"]

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

  # Concat the prediction embedding and the encoder_output
  eshape = tf.shape(encoder_outputs)
  initial_tensor = tf.zeros([eshape[0], eshape[2]])
  initial_tensor.set_shape([None, hparams.decoder_lstm_cells])
  prev_encoder_output = tf.cond(tf.equal(i, 0), lambda: initial_tensor,
                                lambda: encoder_outputs[:, i - 1, :])
  decoder_inputs = tf.concat([prev_encoder_output, prev_emb], axis=1)


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

  if hparams.decoder_proj_add:
    pre_softmax_inputs = tf.add(lstm_outputs[0], encoder_outputs[:, i, :])
  else:
    pre_softmax_inputs = tf.concat([lstm_outputs[0], encoder_outputs[:, i, :]], axis=1)

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


  return i + 1, cur_ids, all_lstm_states, logits
