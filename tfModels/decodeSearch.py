'''
in progress
'''

import tensorflow as tf


def greedy_search(encoder_output, decode_fn, sos_id):
    """Greedy search in graph.
    sos_id: <sos> in seq2seq; <blk> in rna
    """
    batch_size = tf.shape(encoder_output)[0]

    preds = tf.ones([batch_size, 1], dtype=tf.int32) * sos_id
    scores = tf.zeros([batch_size], dtype=tf.float32)

    def step(i, finished, preds, scores, cache):
        # Call decoder and get predictions.
        decoder_output = decode_fn(preds, encoder_output)

        _, next_preds, next_scores = test_output(decoder_output, reuse=reuse)
        next_preds = next_preds[:, None, 0]
        next_scores = next_scores[:, 0]

        # Update.
        scores = scores + next_scores
        preds = tf.concat([preds, next_preds], axis=1)

        # Whether sequences finished.
        has_eos = tf.equal(next_preds[:, 0], 3)
        finished = tf.logical_or(finished, has_eos)

        return i+1, finished, preds, scores, cache

    def not_finished(i, finished, preds, scores, cache):
        return tf.logical_and(
            tf.reduce_any(tf.logical_not(finished)),
            tf.less_equal(
                i,
                tf.reduce_min([tf.shape(encoder_output)[1] + 50, self._config.test.max_target_length])
            )
        )

    i, finished, preds, scores, cache = \
        tf.while_loop(cond=not_finished,
                      body=step,
                      loop_vars=[0, finished, preds, scores, cache],
                      shape_invariants=[
                          tf.TensorShape([]),
                          tf.TensorShape([None]),
                          tf.TensorShape([None, None]),
                          tf.TensorShape([None]),
                          tf.TensorShape([None, None, None, None])],
                      back_prop=False)

    preds = preds[:, 1:]  # remove <S> flag
    return preds


def beam_search(encoder_output, use_cache, reuse):
        """Beam search in graph."""
        beam_size, batch_size = self._config.test.beam_size, tf.shape(encoder_output)[0]
        inf = 1e10

        if beam_size == 1:
            return self.greedy_search(encoder_output, use_cache, reuse)

        def get_bias_scores(scores, bias):
            """
            If a sequence is finished, we only allow one alive branch. This function aims to give one branch a zero score
            and the rest -inf score.
            Args:
                scores: A real value array with shape [batch_size * beam_size, beam_size].
                bias: A bool array with shape [batch_size * beam_size].

            Returns:
                A real value array with shape [batch_size * beam_size, beam_size].
            """
            bias = tf.to_float(bias)
            b = tf.constant([0.0] + [-inf] * (beam_size - 1))
            b = tf.tile(b[None, :], multiples=[batch_size * beam_size, 1])
            return scores * (1 - bias[:, None]) + b * bias[:, None]

        def get_bias_preds(preds, bias):
            """
            If a sequence is finished, all of its branch should be </S> (3).
            Args:
                preds: A int array with shape [batch_size * beam_size, beam_size].
                bias: A bool array with shape [batch_size * beam_size].

            Returns:
                A int array with shape [batch_size * beam_size].
            """
            bias = tf.to_int32(bias)
            return preds * (1 - bias[:, None]) + bias[:, None] * 3

        # Prepare beam search inputs.
        # [batch_size, 1, *, hidden_units]
        encoder_output = encoder_output[:, None, :, :]
        # [batch_size, beam_size, *, hidden_units]
        encoder_output = tf.tile(encoder_output, multiples=[1, beam_size, 1, 1])
        encoder_output = tf.reshape(encoder_output, [batch_size * beam_size, -1, encoder_output.get_shape()[-1].value])
        # [[<S>, <S>, ..., <S>]], shape: [batch_size * beam_size, 1]
        preds = tf.ones([batch_size * beam_size, 1], dtype=tf.int32) * 2
        scores = tf.constant([0.0] + [-inf] * (beam_size - 1), dtype=tf.float32)  # [beam_size]
        scores = tf.tile(scores, multiples=[batch_size])  # [batch_size * beam_size]
        lengths = tf.zeros([batch_size * beam_size], dtype=tf.float32)
        bias = tf.zeros_like(scores, dtype=tf.bool)

        if use_cache:
            cache = tf.zeros([batch_size * beam_size, 0, self._config.num_blocks, self._config.hidden_units])
        else:
            cache = tf.zeros([0, 0, 0, 0])

        def step(i, bias, preds, scores, lengths, cache):
            # Where are we.
            i += 1
            # Call decoder and get predictions.
            if use_cache:
                decoder_output, cache = \
                    self.decoder_with_caching(preds, cache, encoder_output, is_training=False, reuse=reuse)
            else:
                decoder_output = self.decoder(preds, encoder_output, is_training=False, reuse=reuse)

            _, next_preds, next_scores = self.test_output(decoder_output, reuse=reuse)

            next_preds = get_bias_preds(next_preds, bias)
            next_scores = get_bias_scores(next_scores, bias)

            # Update scores.
            scores = scores[:, None] + next_scores  # [batch_size * beam_size, beam_size]
            scores = tf.reshape(scores, shape=[batch_size, beam_size ** 2])  # [batch_size, beam_size * beam_size]

            # LP scores.
            lengths = lengths[:, None] + tf.to_float(tf.not_equal(next_preds, 3))  # [batch_size * beam_size, beam_size]
            lengths = tf.reshape(lengths, shape=[batch_size, beam_size ** 2])  # [batch_size, beam_size * beam_size]
            lp = tf.pow((5 + lengths) / (5 + 1), self._config.test.lp_alpha)  # Length penalty
            lp_scores = scores / lp  # following GNMT

            # Pruning
            _, k_indices = tf.nn.top_k(lp_scores, k=beam_size)
            base_indices = tf.reshape(tf.tile(tf.range(batch_size)[:, None], multiples=[1, beam_size]), shape=[-1])
            base_indices *= beam_size ** 2
            k_indices = base_indices + tf.reshape(k_indices, shape=[-1])  # [batch_size * beam_size]

            # Update lengths.
            lengths = tf.reshape(lengths, [-1])
            lengths = tf.gather(lengths, k_indices)

            # Update scores.
            scores = tf.reshape(scores, [-1])
            scores = tf.gather(scores, k_indices)

            # Update predictions.
            next_preds = tf.gather(tf.reshape(next_preds, shape=[-1]), indices=k_indices)
            preds = tf.gather(preds, indices=k_indices / beam_size)
            if use_cache:
                cache = tf.gather(cache, indices=k_indices / beam_size)
            preds = tf.concat((preds, next_preds[:, None]), axis=1)  # [batch_size * beam_size, i]

            # Whether sequences finished.
            bias = tf.equal(preds[:, -1], 3)  # </S>?

            return i, bias, preds, scores, lengths, cache

        def not_finished(i, bias, preds, scores, lengths, cache):
            return tf.logical_and(
                tf.reduce_any(tf.logical_not(bias)),
                tf.less_equal(
                    i,
                    tf.reduce_min([tf.shape(encoder_output)[1] + 50, self._config.test.max_target_length])
                )
            )

        i, bias, preds, scores, lengths, cache = \
            tf.while_loop(cond=not_finished,
                          body=step,
                          loop_vars=[0, bias, preds, scores, lengths, cache],
                          shape_invariants=[
                              tf.TensorShape([]),
                              tf.TensorShape([None]),
                              tf.TensorShape([None, None]),
                              tf.TensorShape([None]),
                              tf.TensorShape([None]),
                              tf.TensorShape([None, None, None, None])],
                          back_prop=False)

        scores = tf.reshape(scores, shape=[batch_size, beam_size])
        preds = tf.reshape(preds, shape=[batch_size, beam_size, -1])  # [batch_size, beam_size, max_length]

        max_indices = tf.to_int32(tf.argmax(scores, axis=-1))  # [batch_size]
        max_indices += tf.range(batch_size) * beam_size
        preds = tf.reshape(preds, shape=[batch_size * beam_size, -1])

        final_preds = tf.gather(preds, indices=max_indices)
        final_preds = final_preds[:, 1:]  # remove <S> flag
        return final_preds
