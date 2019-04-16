import tensorflow as tf
from tfTools.tfTools import dense_sequence_to_sparse
import numpy as np
import time


def editDistance(hyp, ref):
    '''
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Input:
        hyp: the list of words produced by splitting hypothesis sentence.
        ref: the list of words produced by splitting reference sentence.
        d:   r e f
           h
           y
           p
    '''
    # assert (len(hyp) < 200) and (len(ref) < 200)
    d = np.zeros((len(hyp)+1, len(ref)+1), dtype=np.uint8)
    d[0, :] = np.arange(len(ref)+1)
    d[:, 0] = np.arange(len(hyp)+1)
    for i in range(1, len(hyp)+1):
        for j in range(1, len(ref)+1):
            if hyp[i-1] == ref[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d

def ED_tf(hyp, ref):
    return tf.py_func(editDistance, [hyp, ref], tf.uint8)


def editDistance_batch(hyp, ref):
    '''
    This function is to calculate the edit distance of reference sentence and
    the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Input:
        hyp: the list of words produced by splitting hypothesis sentence.
        ref: the list of words produced by splitting reference sentence.
        d:   r e f
           h
           y
           p
    '''
    batch_size, len_hyp = hyp.shape
    _, len_ref = ref.shape

    # initialization
    d = np.zeros((batch_size, len_hyp+1, len_ref+1), dtype=np.uint8)
    for i in range(batch_size):
        d[i, 0, :] = np.arange(len_ref+1)
        d[i, :, 0] = np.arange(len_hyp+1)

    # dynamic programming
    for i in range(1, len_hyp+1):
        for j in range(1, len_ref+1):
            sub = d[:, i-1, j-1] + 1
            ins = d[:, i, j-1] + 1
            det = d[:, i-1, j] + 1
            d[:, i, j] = np.where(
                hyp[:, i-1] == ref[:, j-1],
                d[:, i-1, j-1],
                np.amin(np.stack([sub, ins, det]), 0)
            )
    return d

def ED_batch_tf(hyp, ref):
    return tf.py_func(editDistance_batch, [hyp, ref], tf.uint8)


def optimal_completion_targets(hyp, ref):
    '''
    OCD targsts
    Optimal Completion Distillation for Sequence Learning
    http://arxiv.org/abs/1810.01398
    this is for no eos
    return the 3d mask tensor of the distance table
    the function only used to compute the target distributions, no gradients need
    to pass through, so the py_func is OK here.

    NOTATION:
        the `hyp` should not be None, i.e. `len_hyp` should larger than 0
    '''
    batch_size, len_hyp = hyp.shape
    _, len_ref = ref.shape
    # initialization
    d = np.zeros((batch_size, len_hyp, len_ref), dtype=np.int32)
    d[:, 0, :] = np.tile(np.expand_dims(np.arange(len_ref), 0), [batch_size, 1])
    d[:, :, 0] = np.tile(np.expand_dims(np.arange(len_hyp), 0), [batch_size, 1])

    # dynamic programming
    for i in range(1, len_hyp):
        for j in range(1, len_ref):
            sub = d[:, i-1, j-1] + 1
            ins = d[:, i, j-1] + 1
            det = d[:, i-1, j] + 1
            d[:, i, j] = np.where(
                hyp[:, i-1] == ref[:, j-1],
                d[:, i-1, j-1],
                np.amin(np.stack([sub, ins, det]), 0))
    m = np.amin(d, -1)
    mask_min = np.equal(d, np.stack([m]*d.shape[-1], -1)).astype(np.int32)
    del d

    return mask_min, m

def optimal_completion_targets_with_blank(hyp, ref, blank_id):
    '''
    OCD targsts for RNA structure where blanks are inserted between the ref labels
    '''
    batch_size, len_hyp = hyp.shape
    _, len_ref = ref.shape

    # initialization
    d = np.zeros((batch_size, len_hyp+1, len_ref+1), dtype=np.int32)
    d[:, 0, :] = np.tile(np.expand_dims(np.arange(len_ref+1), 0), [batch_size, 1])
    blank_batch = np.ones([batch_size], dtype=np.int32)*blank_id
    real_count = np.ones([batch_size], dtype=np.int32)
    for i in range(1, len_hyp+1):
        cond = (hyp[:,i-1] != blank_batch)
        d[:,i,0] = np.where(
            cond,
            real_count,
            real_count-1)
        real_count += cond

    # dynamic programming
    for i in range(1, len_hyp+1):
        for j in range(1, len_ref+1):
            sub = d[:, i-1, j-1] + 1
            ins = d[:, i, j-1] + 1
            det = d[:, i-1, j] + 1
            d[:, i, j] = np.where(
                hyp[:, i-1] == ref[:, j-1],
                d[:, i-1, j-1],
                np.amin(np.stack([sub, ins, det]), 0))
            # the line of blank is kept as previous line
            d[:, i, j] = np.where(
                hyp[:, i-1] == blank_batch,
                d[:, i-1, j],
                d[:, i, j])

    m = np.amin(d, -1)
    mask_min = np.equal(d, np.stack([m]*d.shape[-1], -1)).astype(np.int32)

    return mask_min

def optimal_completion_targets_with_blank_v2(hyp, ref, blank_id):
    '''
    OCD targsts for RNA structure where blanks are inserted between the ref labels
    we add the blank id as the end column
    randomly add balnk to the optimal_targets
    '''
    batch_size, len_hyp = hyp.shape
    _, len_ref = ref.shape

    len_hyp_real = np.sum(hyp.astype(bool) , -1)
    len_ref_real = np.sum(ref.astype(bool) , -1)

    # initialization & blank target assignment
    d = np.zeros((batch_size, len_hyp, len_ref+1), dtype=np.int32)
    d[:, 0, :] = np.tile(np.expand_dims(np.arange(len_ref+1), 0), [batch_size, 1])
    target_blank = np.zeros([batch_size, len_hyp], dtype=np.bool)
    for i in range(len_hyp):
        if i == 0:
            real_count = np.zeros([batch_size], dtype=np.int32)
            batch_blank = np.ones([batch_size], dtype=np.int32)*blank_id
            num_blanks_left = len_hyp_real-len_ref_real
            d[:,i,0] = real_count
        else:
            cond = (hyp[:,i-1] != batch_blank)
            d[:,i,0] = np.where(
                cond,
                real_count+1,
                real_count)
            real_count += cond
            num_blanks_left -= (1 - cond)

        p = (num_blanks_left/(len_hyp_real-i+1e-5)).clip(min=0, max=1)
        target_blank[:, i] = np.where(
            np.random.binomial(n=[1]*batch_size, p=p),
            np.ones([batch_size], dtype=np.bool),
            np.zeros([batch_size], dtype=np.bool))

    # dynamic programming
    for i in range(1, len_hyp):
        for j in range(1, len_ref+1):
            sub = d[:, i-1, j-1] + 1
            ins = d[:, i, j-1] + 1
            det = d[:, i-1, j] + 1
            d[:, i, j] = np.where(
                hyp[:, i-1] == ref[:, j-1],
                d[:, i-1, j-1],
                np.amin(np.stack([sub, ins, det]), 0))
            # the line of blank is kept as previous line
            d[:, i, j] = np.where(
                hyp[:, i-1] == batch_blank,
                d[:, i-1, j],
                d[:, i, j])

    m = np.amin(d, -1)
    mask_min = np.equal(d, np.stack([m]*d.shape[-1], -1))
    mask_min[:, :, -1] = target_blank

    return mask_min.astype(np.int32)


def optimal_completion_targets_tf(hyp, ref):
    '''
    OCD targsts
    Optimal Completion Distillation for Sequence Learning
    http://arxiv.org/abs/1810.01398
    this is for no eos
    return the 3d mask tensor of the distance table
    the function only used to compute the target distributions, no gradients need
    to pass through, so the py_func is OK here.

    NOTATION:
        the `hyp` should not be None, i.e. `len_hyp` should larger than 0
        d0, d2
        d1, d
    '''
    batch_size = tf.shape(hyp)[0]
    len_hyp = tf.shape(hyp)[1]
    len_ref = tf.shape(ref)[1]

    # initialization
    d_init = tf.tile(tf.range(len_ref)[None, None, :], [batch_size, 1, 1])

    def sent(i, d):
        def step(j, i, d_prev, d_cur):
            d0 = d_prev[:, j-1]
            d2 = d_prev[:, j]

            sub = d0 + 1
            ins = d_cur[:, -1] + 1
            det = d2 + 1
            d = tf.where(
                hyp[:, i-1] == ref[:, j-1],
                d0,
                tf.reduce_min(tf.stack([sub, ins, det]), 0))
            d_cur = tf.concat([d_cur, d[:, None]], -1)

            return j+1, i, d_prev, d_cur

        _, _, _, d_line = tf.while_loop(
            cond=lambda j, *_: tf.less(j, len_ref),
            body=step,
            loop_vars=[1, i, d[:, -1, :], tf.tile([[i]], [batch_size, 1])],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([]),
                              tf.TensorShape([None, None]),
                              tf.TensorShape([None, None])])
        d = tf.concat([d, d_line[:, None, :]], 1)

        return i+1, d

    _, d = tf.while_loop(
        cond=lambda i, *_: tf.less(i, len_hyp),
        body=sent,
        loop_vars=[1, d_init],
        shape_invariants=[tf.TensorShape([]),
                          tf.TensorShape([None, None, None])])

    m = tf.reduce_max(d, -1)
    mask_min = tf.to_int32(tf.equal(d, tf.tile(m[:, :, None], [1, 1, len_ref])))

    return mask_min


def Qvalue(hyp, ref):
    '''
    [[0, 1, 1, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 1]]
    '''
    d = tf.py_func(editDistance_batch, [hyp, ref], tf.uint8)
    q = -tf.to_int32(tf.reduce_min(d, -1))
    # q = q[:, 1:] - q[:, :-1]

    return q

def OCD(hyp, ref, vocab_size):
    """
    make sure the padding id is 0!
    ref: * * * * <eos> <pad> <pad>
    the gradient is stoped at mask_min
    """
    mask_min = tf.py_func(optimal_completion_targets, [hyp, ref], tf.int32)
    # mask_min = optimal_completion_targets_tf(hyp, ref)

    tiled_ref = tf.tile(tf.expand_dims(ref, 1), [1, tf.shape(mask_min)[1], 1])
    optimal_targets = tiled_ref * mask_min

    # optimal_targets = tf.Print(optimal_targets, [optimal_targets[0]], message='optimal_targets: ', summarize=1000)
    # optimal_targets = tf.Print(optimal_targets, [ref[0]], message='ref: ', summarize=1000)
    # optimal_targets = tf.Print(optimal_targets, [hyp[0]], message='hyp: ', summarize=1000)
    # labels: batch x time x vocab
    # we do not need the optimal targrts when the prefix is complete.
    labels = tf.one_hot(optimal_targets, vocab_size)
    optimal_distribution = tf.reduce_sum(labels, -2)

    # ignore the label 0 (pad id) in labels
    batch_size = tf.shape(optimal_distribution)[0]
    time_size = tf.shape(optimal_distribution)[1]
    optimal_distribution = tf.concat([tf.zeros(shape=[batch_size, time_size, 1], dtype=tf.float32),
                                      optimal_distribution[:, :, 1:]], -1)
    # average over all the optimal targets
    optimal_distribution = tf.nn.softmax(optimal_distribution*1024)
    # optimal_distribution = tf.stop_gradient(optimal_distribution)

    return optimal_distribution, optimal_targets
    # return optimal_distribution


def OCD_with_blank_loss(hyp, ref, vocab_size):
    """
    make sure the padding id is 0!
    randomly add blank_id as optimal target
    """
    from tfTools.tfTools import batch_pad
    blank_id = vocab_size - 1

    # mask_min = tf.py_func(optimal_completion_targets, [hyp, ref], tf.int32)
    # mask_min = tf.py_func(optimal_completion_targets_with_blank, [hyp, ref, blank_id], tf.int32)
    mask_min = tf.py_func(optimal_completion_targets_with_blank_v2, [hyp, ref, blank_id], tf.int32)
    # replace the ref's pad with blank since it is the end of the sequence
    ref = tf.to_int32(tf.ones_like(ref)*blank_id)*tf.to_int32(tf.equal(ref, 0)) + ref
    shifted_ref = batch_pad(p=ref, length=1, pad=blank_id, direct='tail')
    tiled_ref = tf.tile(tf.expand_dims(shifted_ref, 1), [1, tf.shape(mask_min)[1], 1])
    optimal_targets = tiled_ref * mask_min

    # labels: batch x time x vocab
    # we do not need the optimal targrts when the prefix is complete.
    labels = tf.one_hot(optimal_targets, vocab_size)
    optimal_distribution = tf.reduce_sum(labels, -2)

    # ignore the label 0 in labels
    batch_size = tf.shape(optimal_distribution)[0]
    time_size = tf.shape(optimal_distribution)[1]
    optimal_distribution = tf.concat([tf.zeros(shape=[batch_size, time_size, 1], dtype=tf.float32),
                                      optimal_distribution[:, :, 1:]], -1)
    # average over all the optimal targets
    optimal_distribution = tf.nn.softmax(optimal_distribution*1024)

    # return optimal_distribution, optimal_targets
    return optimal_distribution, optimal_targets


def test_ED_tf():
    list_vocab = list('_SATRYUNDP')
    value_hyp = np.array([list_vocab.index(s) for s in 'SATURDAY'], dtype=np.uint8)
    value_ref = np.array([list_vocab.index(s) for s in 'SUNDAY'], dtype=np.uint8)

    # build graph
    hyp = tf.placeholder(tf.uint8)
    ref = tf.placeholder(tf.uint8)
    table_distance = ED_tf(hyp, ref)

    # run graph
    with tf.Session() as sess:
        distance = sess.run(table_distance, {hyp: value_hyp,
                                             ref: value_ref})
        print(distance)

def test_ED_batch():
    list_vocab = list('_SATRYUNDP')
    # value_hyp = np.array([list_vocab.index(s) for s in 'SATRAPY'], dtype=np.uint8)
    # value_ref = np.array([list_vocab.index(s) for s in 'SUNDAY_'], dtype=np.uint8)
    value_hyp = np.array([[list_vocab.index(s) for s in 'SATURDAY'],
                          [list_vocab.index(s) for s in 'SUNDAY__']],
                         dtype=np.int32)

    value_ref = np.array([[list_vocab.index(s) for s in 'SUNDAY'],
                          [list_vocab.index(s) for s in 'SUNDAY']],
                         dtype=np.int32)

    # print(np.array([value_hyp, value_ref]))
    # print(editDistance_batch(np.array([value_hyp, value_ref]), np.array([value_ref, value_hyp])))

    # build graph
    hyp = tf.placeholder(tf.uint8)
    ref = tf.placeholder(tf.uint8)
    table_distance = ED_batch_tf(hyp, ref)

    # run graph
    with tf.Session() as sess:
        distance = sess.run(table_distance, {hyp: value_hyp,
                                             ref: value_ref})
        print(distance)


def test_Qvalue():
    """
    """
    list_vocab = list('_SATRYUNDP-')

    value_hyp = np.array([[list_vocab.index(s) for s in 'SATURDAY-'],
                          [list_vocab.index(s) for s in 'SUNDAY-__']],
                         dtype=np.int32)

    value_ref = np.array([[list_vocab.index(s) for s in 'SUNDAY-'],
                          [list_vocab.index(s) for s in 'SUNDAY-']],
                         dtype=np.int32)

    # build graph
    hyp = tf.placeholder(tf.int32)
    ref = tf.placeholder(tf.int32)

    m = Qvalue(hyp, ref)
    print('graph has built...')

    # run graph
    with tf.Session() as sess:
        feed_dict = {hyp: value_hyp,
                     ref: value_ref}
        m_ = sess.run([m], feed_dict)
        print(m_)

def test_OCD_loss():
    """
    the pad will be removed from optimal target in the end ,and we do not need
    to care about the length of hyp and ref since we will finally mask the loss.
          S  U  N  D  A  Y
     [ 1  0  0  0  0  0  0]
    S[ 0  6  0  0  0  0  0]
    A[ 0  6  7  0  0  0  0]
    T[ 0  6  7  8  0  0  0]
    R[ 0  6  7  8  2  0  0]
    A[ 0  0  0  0  0  5  0]
    P[ 0  0  0  0  0  5 10]
    Y[ 0  0  0  0  0  0 10]

          S  U  N  D  A  _
     [ 1  0  0  0  0  0  0]
    S[ 0  6  0  0  0  0  0]
    A[ 0  6  7  0  0  0  0]
    T[ 0  6  7  8  0  0  0]
    R[ 0  6  7  8  2  0  0]
    A[ 0  0  0  0  0  0  0]
    P[ 0  0  0  0  0  0 10]
    _[ 0  0  0  0  0  0 10]

    output:
    sample 0:
    [[ 1  0  0  0  0  0  0]
     [ 0  6  0  0  0  0  0]
     [ 0  6  7  0  0  0  0]
     [ 0  6  7  8  0  0  0]
     [ 0  6  7  8  2  0  0]
     [ 0  0  0  0  0  5  0]
     [ 0  0  0  0  0  5 10]
     [ 0  0  0  0  0  0 10]
     [ 0  0  0  0  0  0 10]]
    S
    U
    U, N
    U, N, D
    U, N, D, A
    Y
    Y, -
    -
    [[ 0.    1.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
     [ 0.    0.    0.    0.    0.    0.    1.    0.    0.    0.    0.]
     [ 0.    0.    0.    0.    0.    0.    0.5   0.5   0.    0.    0.]
     [ 0.    0.    0.    0.    0.    0.    0.33  0.33  0.33  0.    0.]
     [ 0.    0.    0.25  0.    0.    0.    0.25  0.25  0.25  0.    0.]
     [ 0.    0.    0.    0.    0.    1.    0.    0.    0.    0.    0.]
     [ 0.    0.    0.    0.    0.    0.5   0.    0.    0.    0.    0.5]]

    sample 1:
    [[ 1  0  0  0  0  0  0]
     [ 0  6  0  0  0  0  0]
     [ 0  6  7  0  0  0  0]
     [ 0  6  7  8  0  0  0]
     [ 0  6  7  8  2  0  0]
     [ 0  0  0  0  0 10  0]
     [ 0  0  0  0  0 10 10]
     [ 0  0  0  0  0  0 10]]
    S
    U
    U, N
    U, N, D
    U, N, D, A
    -
    -

    [[ 0.    1.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
     [ 0.    0.    0.    0.    0.    0.    1.    0.    0.    0.    0.]
     [ 0.    0.    0.    0.    0.    0.    0.5   0.5   0.    0.    0.]
     [ 0.    0.    0.    0.    0.    0.    0.33  0.33  0.33  0.    0.]
     [ 0.    0.    0.25  0.    0.    0.    0.25  0.25  0.25  0.    0.]
     [ 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    1.]
     [ 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    1.]]

    decoded[0]: [524 156 241 464 280 299]
    labels[0]:  [524 42  239 241 101 464 280 299]
    optimal_targets[0]: [
    [524 0 0 0 0 0 0 0 0 0 0]
    [0 42 0 0 0 0 0 0 0 0 0]
    [0 42 239 0 0 0 0 0 0 0 0]
    [0 42 239 241 101 0 0 0 0 0 0]
    [0 42 239 241 101 464 280 0 0 0 0]
    [0 0 0 0 0 0 0 299 0 0 0]
    """
    list_vocab = list('_SATRYUNDP-')

    value_hyp = np.array([[list_vocab.index(s) for s in 'SATURDAY'],
                          [list_vocab.index(s) for s in 'SUNDAY__']],
                         dtype=np.int32)

    value_ref = np.array([[list_vocab.index(s) for s in 'SUNDAY'],
                          [list_vocab.index(s) for s in 'SUNDAY']],
                         dtype=np.int32)

    # list_vocab = list('_abcdefgmnop-')
    # value_hyp = np.array([[list_vocab.index(s) for s in 'abcdef']],
    #                      dtype=np.int32)
    #
    # value_ref = np.array([[list_vocab.index(s) for s in 'amncodef']], #
    #                      dtype=np.int32)
    # print(optimal_completion_targets(value_hyp, value_ref))

    # build graph
    hyp = tf.placeholder(tf.int32)
    ref = tf.placeholder(tf.int32)

    optimal_distribution_T, optimal_targets_T = OCD(hyp, ref, vocab_size=len(list_vocab))
    print('graph has built...')

    # run graph
    with tf.Session() as sess:
        feed_dict = {hyp: value_hyp,
                     ref: value_ref}
        optimal_distribution, optimal_targets = sess.run([optimal_distribution_T, optimal_targets_T], feed_dict)
        print(optimal_targets)

        batch_size, time_size, len_target = optimal_targets.shape
        for i in range(batch_size):
            print('\nsample {}:\n{}'.format(i, optimal_targets[i]))
            for t in range(time_size):
                targets = optimal_targets[i][t]
                print(', '.join(list_vocab[token] for token in targets[targets>0]))
            print(optimal_distribution[i])

def test_OCD_with_blank_loss():
    """
    optimal_targets:
     [[[ 1  0  0  0  0  0 10]
      [ 1  0  0  0  0  0  0]
      [ 0  6  0  0  0  0 10]
      [ 0  6  0  0  0  0  0]
      [ 0  6  7  0  0  0 10]
      [ 0  6  7  0  0  0 10]
      [ 0  6  7  0  0  0  0]
      [ 0  6  7  8  0  0 10]
      [ 0  6  7  8  2  0 10]
      [ 0  6  7  8  2  0 10]
      [ 0  0  0  0  0  5 10]
      [ 0  0  0  0  0  5 10]
      [ 0  0  0  0  0  5 10]
      [ 0  0  0  0  0  5 10]
      [ 0  0  0  0  0  5 10]]

     [[ 1  0  0  0  0  0 10]
      [ 0  6  0  0  0  0 10]
      [ 0  6  0  0  0  0 10]
      [ 0  6  7  0  0  0 10]
      [ 0  6  7  8  0  0  0]
      [ 0  6  7  8  0  0 10]
      [ 0  6  7  8  2  0 10]
      [ 0  6  7  8  2  0 10]
      [ 0  0  0  0  0 10 10]
      [ 0  0  0  0  0 10 10]
      [ 0  0  0  0  0 10 10]
      [ 0  0  0  0  0 10 10]
      [ 0  0  0  0  0  0  0]
      [ 0  0  0  0  0  0  0]
      [ 0  0  0  0  0  0  0]]]

    sample 0:
    [[ 1  0  0  0  0  0 10]
     [ 1  0  0  0  0  0  0]
     [ 0  6  0  0  0  0 10]
     [ 0  6  0  0  0  0  0]
     [ 0  6  7  0  0  0 10]
     [ 0  6  7  0  0  0 10]
     [ 0  6  7  0  0  0  0]
     [ 0  6  7  8  0  0 10]
     [ 0  6  7  8  2  0 10]
     [ 0  6  7  8  2  0 10]
     [ 0  0  0  0  0  5 10]
     [ 0  0  0  0  0  5 10]
     [ 0  0  0  0  0  5 10]
     [ 0  0  0  0  0  5 10]
     [ 0  0  0  0  0  5 10]]
    S, -
    S
    U, -
    U
    U, N, -
    U, N, -
    U, N
    U, N, D, -
    U, N, D, A, -
    U, N, D, A, -
    Y, -
    Y, -
    Y, -
    Y, -
    Y, -
    (15, 11)

    sample 1:
    [[ 1  0  0  0  0  0 10]
     [ 0  6  0  0  0  0 10]
     [ 0  6  0  0  0  0 10]
     [ 0  6  7  0  0  0 10]
     [ 0  6  7  8  0  0  0]
     [ 0  6  7  8  0  0 10]
     [ 0  6  7  8  2  0 10]
     [ 0  6  7  8  2  0 10]
     [ 0  0  0  0  0 10 10]
     [ 0  0  0  0  0 10 10]
     [ 0  0  0  0  0 10 10]
     [ 0  0  0  0  0 10 10]
     [ 0  0  0  0  0  0  0]
     [ 0  0  0  0  0  0  0]
     [ 0  0  0  0  0  0  0]]
    S, -
    U, -
    U, -
    U, N, -
    U, N, D
    U, N, D, -
    U, N, D, A, -
    U, N, D, A, -
    -, -
    -, -
    -, -
    -, -

    (15, 11)
    """
    list_vocab = list(' SATRYUNDP-')

    # value_hyp = np.array([[list_vocab.index(s) for s in '-S-A--TR-A-P--Y'],
    #                       [list_vocab.index(s) for s in 'S-AT-R-A-P-    ']],
    #                      dtype=np.int32)
    value_hyp = np.array([[list_vocab.index(s) for s in '---------------'],
                          [list_vocab.index(s) for s in '-----------    ']],
                         dtype=np.int32)

    value_ref = np.array([[list_vocab.index(s) for s in 'SUNDAY'],
                          [list_vocab.index(s) for s in 'SUNDA ']],
                         dtype=np.int32)
    print(optimal_completion_targets_with_blank_v2(value_hyp, value_ref, len(list_vocab)-1))

    # build graph
    hyp = tf.placeholder(tf.int32)
    ref = tf.placeholder(tf.int32)

    optimal_distribution_T, optimal_targets_T = OCD_with_blank_loss(hyp, ref, vocab_size=len(list_vocab))
    print('graph has built...')

    # run graph
    with tf.Session() as sess:
        feed_dict = {hyp: value_hyp,
                     ref: value_ref}
        optimal_distribution, optimal_targets = sess.run([optimal_distribution_T, optimal_targets_T], feed_dict)
        print('optimal_targets: \n', optimal_targets)

        batch_size, time_size, len_target = optimal_targets.shape
        for i in range(batch_size):
            print('\nsample {}:\n{}'.format(i, optimal_targets[i]))
            for t in range(time_size):
                targets = optimal_targets[i][t]
                print(', '.join(list_vocab[token] for token in targets[targets>0]))
            print(optimal_distribution[i].shape)

def test_optimal_completion_targets_tf():
    """
    have not pass the test yet!!
    """
    list_vocab = list('_SATRYUNDP-')

    value_hyp = np.array([[list_vocab.index(s) for s in 'SATURDAY'],
                          [list_vocab.index(s) for s in 'SUNDAY__']],
                         dtype=np.int32)

    value_ref = np.array([[list_vocab.index(s) for s in 'SUNDAY'],
                          [list_vocab.index(s) for s in 'SUNDAY']],
                         dtype=np.int32)

    # build graph
    hyp = tf.placeholder(tf.int32)
    ref = tf.placeholder(tf.int32)

    _, Q_value = optimal_completion_targets_tf(hyp, ref)
    print('graph has built...')

    # run graph
    with tf.Session() as sess:
        feed_dict = {hyp: value_hyp,
                     ref: value_ref}
        Q = sess.run([Q_value], feed_dict)
        print(Q)

if __name__ == '__main__':
    # test_ED_tf()
    # test_ED_batch()
    # test_OCD_loss()
    test_Qvalue()
    # test_OCD_with_blank_loss()
    # test_optimal_completion_targets_tf()
