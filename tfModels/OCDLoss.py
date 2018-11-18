import tensorflow as tf
from tfTools.tfTools import dense_sequence_to_sparse
import numpy as np

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
    this is for no eos
    return the 3d mask tensor of the distance table
    '''
    batch_size, len_hyp = hyp.shape
    _, len_ref = ref.shape

    # initialization
    d = np.zeros((batch_size, len_hyp+1, len_ref+1), dtype=np.int32)
    for i in range(batch_size):
        d[i, 0, :] = np.arange(len_ref+1)
        d[i, :, 0] = np.arange(len_hyp+1)
    # m = np.zeros((batch_size, len_hyp+1), dtype=np.uint8)

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
        # m[:, i] = np.amin(d[:, i, :], 0)
    m = np.amin(d, -1)
    mask_min = np.equal(d, np.stack([m]*d.shape[-1], -1)).astype(np.int32)

    return mask_min


def OCD_tf(hyp, ref, vocab_size):
    """
    make sure the padding id is 0!
    """
    from tfTools.tfTools import batch_pad
    blank_id = vocab_size - 1

    mask_min = tf.py_func(optimal_completion_targets, [hyp, ref], tf.int32)
    # replace the ref's pad with blank since it is the end of the sequence
    ref = tf.to_int32(tf.ones_like(ref)*blank_id)*tf.to_int32(tf.equal(ref, 0)) + ref
    shifted_ref = batch_pad(p=ref, length=1, pad=blank_id, direct='tail')
    tiled_ref = tf.tile(tf.expand_dims(shifted_ref, 1), [1, tf.shape(mask_min)[1], 1])
    optimal_targets = tiled_ref * mask_min

    # labels: batch x time x vocab
    # we do not need the optimal targrts when the prefix is complete.
    labels = tf.one_hot(optimal_targets[:, :-1, :], vocab_size)
    optimal_distribution = tf.reduce_sum(labels, -2)

    # ignore the label 0 in labels
    batch_size = tf.shape(optimal_distribution)[0]
    time_size = tf.shape(optimal_distribution)[1]
    optimal_distribution = tf.concat([tf.zeros(shape=[batch_size, time_size, 1], dtype=tf.float32),
                                      optimal_distribution[:, :, 1:]], -1)
    # average over all the optimal targets
    optimal_distribution = tf.nn.softmax(optimal_distribution*1024)

    return optimal_distribution, optimal_targets


def test_ED_tf():
    list_vocab = list('_SATRYUNDP')
    value_hyp = np.array([list_vocab.index(s) for s in 'SATRAPY'], dtype=np.uint8)
    value_ref = np.array([list_vocab.index(s) for s in 'SUNDAY_'], dtype=np.uint8)

    # build graph
    hpy = tf.placeholder(tf.uint8)
    ref = tf.placeholder(tf.uint8)
    table_distance = ED_tf(hpy, ref)

    # run graph
    with tf.Session() as sess:
        distance = sess.run(table_distance, {hpy: value_hyp,
                                             ref: value_ref})
        print(distance)


def test_ED_batch():
    list_vocab = list('_SATRYUNDP')
    value_hyp = np.array([list_vocab.index(s) for s in 'SATRAPY'], dtype=np.uint8)
    value_ref = np.array([list_vocab.index(s) for s in 'SUNDAY_'], dtype=np.uint8)

    # print(np.array([value_hyp, value_ref]))
    # print(editDistance_batch(np.array([value_hyp, value_ref]), np.array([value_ref, value_hyp])))

    # build graph
    hpy = tf.placeholder(tf.uint8)
    ref = tf.placeholder(tf.uint8)
    table_distance = ED_batch_tf(hpy, ref)

    # run graph
    with tf.Session() as sess:
        distance = sess.run(table_distance, {hpy: np.array([value_hyp, value_ref]),
                                             ref: np.array([value_ref, value_hyp])})
        print(distance)


def test_OCD():
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
     [ 0  0  0  0  0  0 10]]
    S
    U
    U, N
    U, N, D
    U, N, D, A
    Y
    Y, <blk>
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
    <blk>
    <blk>, <blk>
    [[ 0.    1.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
     [ 0.    0.    0.    0.    0.    0.    1.    0.    0.    0.    0.]
     [ 0.    0.    0.    0.    0.    0.    0.5   0.5   0.    0.    0.]
     [ 0.    0.    0.    0.    0.    0.    0.33  0.33  0.33  0.    0.]
     [ 0.    0.    0.25  0.    0.    0.    0.25  0.25  0.25  0.    0.]
     [ 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    1.]
     [ 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    1.]]

    """
    list_vocab = list('_SATRYUNDP')+['<blk>']

    value_hyp = np.array([[list_vocab.index(s) for s in 'SATRAPY'],
                          [list_vocab.index(s) for s in 'SATRAP_']],
                         dtype=np.int32)

    value_ref = np.array([[list_vocab.index(s) for s in 'SUNDAY'],
                          [list_vocab.index(s) for s in 'SUNDA_']],
                         dtype=np.int32)

    # print(optimal_completion_targets(np.array([value_hyp, value_ref]), np.array([value_ref, value_hyp])))

    # build graph
    hpy = tf.placeholder(tf.int32)
    ref = tf.placeholder(tf.int32)

    optimal_distribution_T, optimal_targets_T = OCD_tf(hpy, ref, vocab_size=len(list_vocab))
    print('graph has built...')

    # run graph
    with tf.Session() as sess:
        feed_dict = {hpy: value_hyp,
                     ref: value_ref}
        optimal_distribution, optimal_targets = sess.run([optimal_distribution_T, optimal_targets_T], feed_dict)
        # print(optimal_targets)

        batch_size, time_size, len_target = optimal_targets.shape
        for i in range(batch_size):
            print('\nsample {}:\n{}'.format(i, optimal_targets[i]))
            for t in range(time_size-1):
                targets = optimal_targets[i][t]
                print(', '.join(list_vocab[token] for token in targets[targets>0]))
            print(optimal_distribution[i])


if __name__ == '__main__':
    # test_ED_tf()
    # test_ED_batch()
    test_OCD()
