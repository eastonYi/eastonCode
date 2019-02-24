import tensorflow as tf
import numpy as np


def pad_to(list_array, max_len, size_pad):
    """
    concate list of same size (size_pad) array pad to a larger array
    """
    pad = np.zeros((max_len-len(list_array), size_pad), np.float32)
    if list_array:
        res = np.concatenate([np.array(list_array), pad], 0)
    else:
        res = pad

    return res


def pad_to_same(list_array):
    """
    concate list of same size (size_pad) array pad to a larger array
    """
    list_sents = []
    if list_array:
        max_len = max(len(array) for array in list_array)

        for array in list_array:
            pad = np.zeros((max_len-len(array)), dtype=np.int32)
            list_sents.append(np.concatenate([array, pad]))

    return np.array(list_sents, dtype=np.int32)


def add_avg_hidden(list_hidden, list_one_no_blank):
    if list_one_no_blank:
        h = np.mean(list_one_no_blank, 0)
        # for i in range(1, len(list_one_no_blank), 1):
        #     print(np.sum(np.square(list_one_no_blank[i]-list_one_no_blank[i-1])))
        # print('\n')
        list_hidden.append(h)
        del list_one_no_blank[:]


def add_concate_hidden(list_hidden, list_one_no_blank, num=4):
    if list_one_no_blank:
        null = np.zeros_like(list_one_no_blank[0])
        h = np.concatenate(list_one_no_blank[:num] + [null]*(num-len(list_one_no_blank)))
        list_hidden.append(h)
        del list_one_no_blank[:]


def acoustic_hidden_shrink(distribution_acoustic, hidden, len_acoustic, blank_id, num_post=0, num_frames=4):
    """
    distribution_acoustic: [b x t x v]
    hidden: [b x t x h]
    num_post: the number of posterior hidden frame that add to the current one
    """
    alignments = np.argmax(distribution_acoustic, -1) # alignments: [b x t]
    list_batch = []
    list_len = []
    # batch loop
    for i, _hidden in enumerate(hidden):
        list_hidden = []
        token_pre = None
        list_one_no_blank = []
        # time loop
        for t, h in zip(range(len_acoustic[i]), _hidden):
            if alignments[i][t] == token_pre:
                # a repeated no blank
                list_one_no_blank.append(h)

            elif alignments[i][t] == blank_id:
                # a blank
                token_pre = None
                if num_post>0 and list_one_no_blank:
                    list_one_no_blank.extend([*_hidden[t+1: min(t+num_post+1, len_acoustic[i]-1)]])
                # add_avg_hidden(list_hidden, list_one_no_blank)
                add_concate_hidden(list_hidden, list_one_no_blank, num_frames)
            else:
                # a new no blank
                # add_avg_hidden(list_hidden, list_one_no_blank)
                add_concate_hidden(list_hidden, list_one_no_blank, num_frames)
                list_one_no_blank = [h]
                token_pre = alignments[i][t]

        # add_avg_hidden(list_hidden, list_one_no_blank)
        add_concate_hidden(list_hidden, list_one_no_blank, num_frames)

        list_batch.append(list_hidden)
        list_len.append(len(list_hidden))

    list_hidden_padded = []
    for hidden_shrinked in list_batch:
        # the hidden is at least with length of 1
        hidden_padded = pad_to(hidden_shrinked, max(list_len+[1]), num_frames * hidden.shape[-1])
        list_hidden_padded.append(hidden_padded)
    acoustic_shrinked = np.stack(list_hidden_padded, 0)
    # the hidden is at least with length of 1
    list_len = [x if x != 0 else 1 for x in list_len]

    del list_batch, list_hidden_padded

    return acoustic_shrinked, np.array(list_len, np.int32)


def acoustic_hidden_shrink_tf(distribution_acoustic, hidden, len_acoustic, blank_id, num_post):
    """
    NOTATION: the gradient will not pass over the input vars
    """
    hidden_shrinked, len_no_blank = tf.py_func(acoustic_hidden_shrink, [distribution_acoustic, hidden, len_acoustic, blank_id, num_post],
                      (tf.float32, tf.int32))
    hidden_shrinked.set_shape([None, None, 4*hidden.get_shape()[-1]])
    len_no_blank.set_shape([None])

    return hidden_shrinked, len_no_blank


def analysis_alignments(alignments, len_acoustic, blank_id):
    """
    t: acoustic length
    u: label length
    input:
        alignments: [b, t]
    output:
        repeated_frames: [b, u]
        len_label: [b]
        musk_repeated: [bxu, t]
    """
    # not_blank = np.not_equal(alignments, blank_id)
    lists_num_tokens = []
    list_token_len = []
    list_token_musks = []

    for align, len_time in zip(alignments, len_acoustic):
        list_num_tokens = []
        num_repeated = 0
        prev_is_blank = True
        # the end of align is set to blank
        align = np.concatenate([align, [blank_id]], -1)
        for i, token in zip(range(len_time+1), align):
            if token != blank_id:
                # none of current and previous token is blank
                num_repeated +=1
                prev_is_blank = False
                continue
            elif token == blank_id and not prev_is_blank:
                # current char ends
                list_num_tokens.append(num_repeated)
                musk = np.zeros_like(align, dtype=np.float32)
                musk[i-num_repeated: i] = np.ones([num_repeated], dtype=np.float32)
                list_token_musks.append(musk)
                num_repeated = 0
            prev_is_blank = True

        lists_num_tokens.append(list_num_tokens)
        list_token_len.append(len(list_num_tokens))

    repeated_frames = pad_to_same(lists_num_tokens)
    len_label = np.array(list_token_len, np.int32)
    musk_repeated = pad_to_same(list_token_musks)

    return repeated_frames, len_label, musk_repeated

def analysis_alignments_tf(alignments, len_acoustic, blank_id):
    repeated_frames, len_label, musk_repeated = tf.py_func(
        analysis_alignments, [alignments, len_acoustic, blank_id],
        (tf.int32, tf.int32, tf.int32))
    repeated_frames.set_shape([None, None])
    len_label.set_shape([None])
    musk_repeated.set_shape([None, alignments.get_shape()[-1]])

    return repeated_frames, len_label, musk_repeated


def constrain_repeated(alignments, hidden, len_acoustic, blank_id):
    """
    alignments: [b, t]
    hidden: [b x t x h]
    repeated_frames: [b x u]
    musk_repeated: [bxu, t]

    asserts : 1. step_in_batch should be ended as bxu-1
    """
    repeated_frames, len_label, musk_repeated = analysis_alignments_tf(alignments, len_acoustic, blank_id)
    batch_size = tf.shape(hidden)[0]

    def sent(i, step_in_batch, loss):

        def step(j, i, step_in_batch, loss_sent):
            indices = tf.where(musk_repeated[step_in_batch] > 0)
            repeated_frames = tf.gather(hidden[i], indices)
            mean, variance = tf.nn.moments(repeated_frames, 0)
            loss_sent += tf.reduce_sum(variance)
            # i = tf.Print(i, [i, j, step_in_batch], message='i, j, step_in_batch: ', summarize=1000)

            return j+1, i, step_in_batch+1, loss_sent

        _, _, step_in_batch, loss_sent = tf.while_loop(
            cond=lambda j, *_: tf.less(j, len_label[i]),
            body=step,
            loop_vars=[0, i, step_in_batch, 0.0],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([]),
                              tf.TensorShape([]),
                              tf.TensorShape([])])
        loss += loss_sent

        return i+1, step_in_batch, loss

    _, _, loss = tf.while_loop(
        cond=lambda i, *_: tf.less(i, batch_size),
        body=sent,
        loop_vars=[0, 0, 0.0],
        shape_invariants=[tf.TensorShape([]),
                          tf.TensorShape([]),
                          tf.TensorShape([])])

    return loss


def repeated_constrain_loss(distribution_acoustic, hidden, len_acoustic, blank_id):
    """
    constrain the repeated hidden representations to be the same
    """
    alignments = tf.argmax(distribution_acoustic, -1) # alignments: [b x t]
    loss = constrain_repeated(
        alignments=alignments,
        hidden=hidden,
        len_acoustic=len_acoustic,
        blank_id=blank_id)

    return loss


def test_acoustic_hidden_shrink():
    distribution_acoustic = np.array(
        [[[0.04,0.01, 0.05, 0.9],
          [0.3, 0.8, 0.05, 0.05],
          [0.1, 0.8, 0.05, 0.05],
          [0.05,0.01, 0.9, 0.04]],
         [[0.9, 0.01, 0.05, 0.04],
          [0.1, 0.05, 0.05, 0.9],
          [0.1, 0.05, 0.05, 0.9],
          [0.05,0.01, 0.9, 0.04]]])
    len_acoustic = [2, 4]
    res = acoustic_hidden_shrink(distribution_acoustic, distribution_acoustic, len_acoustic, 3)
    print(res)


def test_shrink_tf():
    distribution_acoustic = tf.Constant(
        [[[0.04,0.01, 0.05, 0.9],
          [0.3, 0.8, 0.05, 0.05],
          [0.1, 0.8, 0.05, 0.05],
          [0.05,0.01, 0.9, 0.04]],
         [[0.9, 0.01, 0.05, 0.04],
          [0.1, 0.05, 0.05, 0.9],
          [0.1, 0.05, 0.05, 0.9],
          [0.05,0.01, 0.9, 0.04]]])
    len_acoustic = [2, 4]
    res = acoustic_hidden_shrink(distribution_acoustic, distribution_acoustic, len_acoustic, 3)
    print(res)


def test_analysis_alignments():
    """
    (array([[2, 1, 3],
            [2, 1, 0]], dtype=int32),
     array([
       [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
       [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=int32))
    """
    alignments = np.array(
    [[9,1,2,9,3,9,4,5,6,9],
     [9,1,2,9,3,9,0,0,0,0]])

    print(analysis_alignments(alignments, [10,6], 9))


def test_constrain_repeated():
    """
    [(None,
        <tf.Variable 'Variable:0' shape=(2, 10) dtype=int32_ref>),
    (<tf.Tensor 'gradients/while/while/strided_slice_2/Enter_grad/b_acc_3:0' shape=(2, 10, 3) dtype=float32>,
        <tf.Variable 'Variable_1:0' shape=(2, 10, 3) dtype=float32_ref>)]
    the gradient can flow the hidden representation rather the alignment, which
    means the model could learn the acoutic representaton but it can only depend
    on the CTC loss to learning the alignment
    """

    # alignments = tf.Variable(
    #     [[9,1,2,9,3,9,4,5,6,9],
    #      [9,1,2,9,3,9,0,0,0,0],
    #      [9,9,9,9,9,9,0,0,0,0]],
    #     dtype=np.int32)
    alignments = tf.Variable(
        [[9,9,9,9,9,9,9,9,9,9],
         [9,9,9,9,9,9,0,0,0,0],
         [9,9,9,9,9,9,0,0,0,0]],
        dtype=np.int32)
    hidden = tf.Variable([
        [[1,2,3],[3,3,3],[3,3,3],[1,2,3],[2,10,20],[1,2,3],[2,20,2],[2,20,2],[2,20,2],[1,2,3]],
        [[1,2,3],[2,2,2],[2,2,2],[1,2,3],[2,2,2],[1,2,3],[2,2,2],[2,2,2],[3,3,2],[1,2,3]],
        [[1,2,3],[2,2,2],[2,2,2],[1,2,3],[2,2,2],[1,2,3],[2,2,2],[2,2,2],[3,3,2],[1,2,3]]],
        dtype=tf.float32)

    loss = constrain_repeated(alignments, hidden, len_acoustic=[10,6,6], blank_id=9)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    # gradients = optimizer.compute_gradients(loss)
    # print(gradients)
    sess=tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    print(sess.run(loss))

if __name__ == '__main__':
    # test_acoustic_hidden_shrink()
    # test_analysis_alignments()
    test_constrain_repeated()
