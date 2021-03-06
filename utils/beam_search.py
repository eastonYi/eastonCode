'''
    search methods
    use sess.sun and feed_dict to run the model
    for debug and understanding
'''
import numpy as np

inf = 10e10
def beam_decode_rerank(model,
                       beam_size,
                       lambda_lm):
    """
    beam search rerank at end with language model integration (self-attention model)
    the input to te score is <sos> + tokens !!!
    """
    # beam search Initialize
    # repeat each sample in batch along the batch axis [1,2,3,4] -> [1,1,2,2,3,3,4,4]
    encoded = tf.tile(encoded[:, None, :, :],
                      multiples=[1, beam_size, 1, 1]) # [batch_size, beam_size, *, hidden_units]
    encoded = tf.reshape(encoded,
                         [batch_size * beam_size, -1, encoded.get_shape()[-1].value])
    # [[<blk>, <blk>, ..., <blk>,]], shape: [batch_size * beam_size, 1]
    token_init = tf.fill([batch_size * beam_size, 1], self.args.sos_idx)
    logits_init = tf.zeros([batch_size * beam_size, 1, self.dim_output], dtype=tf.float32)
    # the score must be [0, -inf, -inf, ...] at init, for the preds in beam is same in init!!!
    scores = tf.constant([0.0] + [-inf] * (beam_size - 1), dtype=tf.float32)  # [beam_size]
    scores = tf.tile(scores, multiples=[batch_size])  # [batch_size * beam_size]
    cache_init = tf.zeros([batch_size*beam_size,
                           0,
                           self.lm.args.model.decoder.num_blocks,
                           self.lm.args.model.decoder.num_cell_units])

    # create decoder cell
    self.cell = self.create_cell()

    # collect the initial states of lstms used in decoder.
    all_initial_states = {}
    all_initial_states["state_decoder"] = self.zero_state(batch_size * beam_size, dtype=tf.float32)
    base_indices = tf.reshape(tf.tile(tf.range(batch_size)[:, None], multiples=[1, beam_size]), shape=[-1])

    def step(i, preds, scores, all_states, logits, cache):
        """
        the cache has no specific shape, so no can be put in the all_states
        """
        state_decoder = all_states["state_decoder"]
        prev_emb = self.embedding(preds[:, -1])
        decoder_input = tf.concat([encoded[:, i, :], prev_emb], axis=1)
        decoder_input.set_shape([None, self.size_embedding + self.num_cell_units_en])

        # Lstm part
        with tf.variable_scope("decoder_lstms"):
            output_decoder, state_decoder = tf.contrib.legacy_seq2seq.rnn_decoder(
                decoder_inputs=[decoder_input],
                initial_state=state_decoder,
                cell=self.cell)
            all_states["state_decoder"] = state_decoder
            # output_decoder = [tf.concat([output_decoder[0], encoded[:, i, :]], axis=1)]

        cur_logit = tf.layers.dense(
            inputs=output_decoder[0],
            units=self.dim_output,
            activation=None,
            use_bias=False,
            name='fully_connected')

        logits = tf.concat([logits, cur_logit[:, None]], 1)
        z = tf.nn.log_softmax(cur_logit) # [batch*beam, size_output]
        # the langueage model infer
        if self.args.model.shallow_fusion:
            preds_emb = self.lm.decoder.embedding(preds)

            with tf.variable_scope(self.args.top_scope, reuse=True):
                with tf.variable_scope(self.args.lm_scope):
                    lm_output, cache = self.lm.decoder.decoder_with_caching_impl(preds_emb, cache)
                    logit_lm = dense(
                        inputs=lm_output[:, -1, :],
                        units=self.dim_output,
                        kernel=tf.transpose(self.lm.decoder.fully_connected),
                        use_bias=False)
            z_lm = lambda_lm * tf.nn.log_softmax(logit_lm) # [batch*beam, size_output]
        else:
            z_lm = tf.zeros_like(z)

        # rank the combined scores
        next_scores, next_preds = tf.nn.top_k(z+z_lm, k=beam_size, sorted=True)
        next_preds = tf.to_int32(next_preds)

        # beamed scores & Pruning
        scores = scores[:, None] + next_scores  # [batch_size * beam_size, beam_size]
        scores = tf.reshape(scores, shape=[batch_size, beam_size * beam_size])

        _, k_indices = tf.nn.top_k(scores, k=beam_size)
        k_indices = base_indices * beam_size * beam_size + tf.reshape(k_indices, shape=[-1])  # [batch_size * beam_size]
        # Update scores.
        scores = tf.reshape(scores, [-1])
        scores = tf.gather(scores, k_indices)
        # Update predictions.
        next_preds = tf.gather(tf.reshape(next_preds, shape=[-1]), indices=k_indices)
        # k_indices: [0~batch*beam*beam], preds: [0~batch*beam]
        preds = tf.gather(preds, indices=k_indices // beam_size)
        preds = tf.concat((preds, next_preds[:, None]), axis=1)  # [batch_size * beam_size, i]

        return i+1, preds, scores, all_states, logits, cache

    _, preds, scores_am, _, logits, cache = tf.while_loop(
        cond=lambda i, *_: tf.less(i, tf.shape(encoded)[1]),
        body=step,
        loop_vars=[0, token_init, scores, all_initial_states, logits_init, cache_init],
        shape_invariants=[tf.TensorShape([]),
                          tf.TensorShape([None, None]),
                          tf.TensorShape([None]),
                          nest.map_structure(lambda t: tf.TensorShape(t.shape), all_initial_states),
                          tf.TensorShape([None, None, self.dim_output]),
                          tf.TensorShape([None, None, None, None])]
        )

    len_encoded = tf.reshape(tf.tile(len_encoded[:, None], multiples=[1, beam_size]), [-1])
    with tf.variable_scope(self.args.top_scope, reuse=True):
        with tf.variable_scope(self.args.lm_scope):
            scores_lm, distribution = self.lm.decoder.score(preds, len_encoded)

    # [batch_size * beam_size, ...]
    preds = preds[:, 1:]
    logits = logits[:, 1:, :]
    not_padding = tf.to_int32(tf.sequence_mask(len_encoded, maxlen=tf.shape(encoded)[1]))
    preds *= not_padding

    # [batch_size , beam_size, ...]
    scores_lm = self.args.lambda_rerank * scores_lm
    scores = scores_am + scores_lm
    # tf.nn.top_k is used to sort `scores`
    scores_sorted, sorted = tf.nn.top_k(tf.reshape(scores, [batch_size, beam_size]),
                                        k=beam_size,
                                        sorted=True)

    sorted = base_indices * beam_size + tf.reshape(sorted, shape=[-1])  # [batch_size * beam_size]

    # [batch_size * beam_size, ...]
    logits_sorted = tf.gather(logits, sorted)
    preds_sorted = tf.gather(preds, sorted)
    scores_lm_sorted = tf.gather(scores_lm, sorted)
    scores_am_sorted = tf.gather(scores_am, sorted)

    # [batch_size, beam_size, ...]
    scores_lm_sorted = tf.reshape(scores_lm_sorted, shape=[batch_size, beam_size])
    scores_am_sorted = tf.reshape(scores_am_sorted, shape=[batch_size, beam_size])
    preds_sorted = tf.reshape(preds_sorted, shape=[batch_size, beam_size, -1])  # [batch_size, beam_size, max_length]
    logits = tf.reshape(logits_sorted, [batch_size, beam_size, -1, self.dim_output])
    len_encoded = tf.reshape(len_encoded, [batch_size, beam_size])

    # return logits, final_preds, len_encoded
    return [preds_sorted, scores_am_sorted, scores_lm_sorted], preds_sorted[:, 0, :], len_encoded
