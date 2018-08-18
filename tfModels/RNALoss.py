import tensorflow as tf

from tfModels.math_tf import sum_log, range_batch


ID_BLANK = tf.convert_to_tensor(0)
LOG_ZERO = tf.convert_to_tensor(-1e29) # can not be too small in tf
LOG_ONE = tf.convert_to_tensor(0.0)


def rna_loss(seqs_input, seqs_labels, seq_input_lens, seq_label_lens, multi_forward, embedding, rna):
    """
    using dynamic programming, which generate a lattice form seq_input to seq_label_lens
    each node is a state of the decoder. Feeding an input and label to push a movement
    on the lattic.
    """
    def right_shift_rows(p, shift, pad):
        assert type(shift) is int
        return tf.concat([tf.ones((tf.shape(p)[0], shift), dtype=tf.float32)*pad,
                          p[:, :-shift]], axis=1)

    def choose_state(multi_state, multi_state_pre, prob, prob_pre, forward_vars, forward_vars_pre):
        # sum_log(prob, forward_vars), sum_log(prob_pre, forward_vars_pre)
        # TODO
        return multi_state

    def step(extend_forward_vars, step_input_x):
        # batch x len_label x (embedding+dim_feature)
        step_input = tf.concat([tf.tile(tf.expand_dims(step_input_x, 1), [1, N+1, 1]),
                                    blank_emb], 2)
        step_input_pre = tf.concat([tf.tile(tf.expand_dims(step_input_x, 1), [1, N+1, 1]),
                                    seqs_labels_emb], 2)

        forward_vars, multi_state = extend_forward_vars
        forward_vars_pre = right_shift_rows(forward_vars, 1, LOG_ZERO)

        # two distinct cell states are going to merge. Here we choose one of them.
        # distrib: batch x len_label x size_vocab
        distrib, multi_state = rna.multi_forward(step_input, multi_state)
        distrib_pre, multi_state_pre = rna.multi_forward(step_input_pre, multi_state)

        prob = distrib[:, :, 0] # prob of blank: batch x len_label
        index_batch = range_batch([size_batch, N+1])
        index_len = range_batch([size_batch, N+1], False)
        prob_pre = tf.gather_nd(distrib_pre, tf.stack([index_batch, index_len, seqs_labels], -1))

        multi_state = choose_state(multi_state, multi_state_pre, prob, prob_pre, forward_vars, forward_vars_pre)
        forward_vars = sum_log(forward_vars_pre + prob_pre, forward_vars + prob)

        return [forward_vars, multi_state]

    size_batch = tf.shape(seqs_input)[0]
    T = tf.shape(seqs_input)[1]
    N = tf.shape(seqs_labels)[1]

    # data: batch x len_label x (embedding+dim_feature), at each time
    seqs_labels_endpad = tf.concat([seqs_labels, tf.zeros([size_batch, 1])], 1)
    seqs_labels_emb = tf.nn.embedding_lookup(embedding, seqs_labels_endpad)
    blank_emb = tf.nn.embedding_lookup(embedding, tf.zeros_like(seqs_labels, tf.int32))
    seqs_input_timeMajor = tf.transpose(seqs_input, ((1, 0, 2))) # actually, len major

    # forward vars: batch x (len_label+1)
    tail = tf.ones((size_batch, N), dtype=tf.float32) * LOG_ZERO
    head = tf.ones((size_batch, 1), dtype=tf.float32) * LOG_ONE
    forward_vars_init = tf.concat([head, tail], -1)

    # state: len_label
    multi_state_init = rna.zero_state(N+1, size_batch)

    # forward loop
    forward_vars_steps = tf.scan(step,
                                 seqs_input_timeMajor,
                                 [forward_vars_init, multi_state_init])


if __name__ == '__main__':
    pass
