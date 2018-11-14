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
    assert (len(hyp) < 200) and (len(ref) < 200)
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

def test_ED_tf():
    list_vocab = list('SATRYUNDP')
    value_hyp =  np.array([list_vocab.index(s) for s in 'SATRAPY'], dtype=np.uint8)
    value_ref =  np.array([list_vocab.index(s) for s in 'SUNDAY'], dtype=np.uint8)

    # build graph
    hpy = tf.placeholder(tf.uint8)
    ref = tf.placeholder(tf.uint8)
    table_distance = ED_tf(hpy, ref)

    # run graph
    with tf.Session() as sess:
        distance = sess.run(table_distance, {hpy: value_hyp,
                                             ref: value_ref})
        print(distance)


if __name__ == '__main__':
    test_ED_tf()
