#!/usr/bin/python
# coding=utf-8

import tensorflow as tf
import numpy as np
import sys

# tf fea opr
def tf_kaldi_fea_delt1(features):
    feats_padded = tf.pad(features, [[1, 1], [0, 0]], "SYMMETRIC")
    feats_padded = tf.pad(feats_padded, [[1, 1], [0, 0]], "SYMMETRIC")

    shape = tf.shape(features)
    l2 = tf.slice(feats_padded, [0, 0], shape)
    l1 = tf.slice(feats_padded, [1, 0], shape)
    r1 = tf.slice(feats_padded, [3, 0], shape)
    r2 = tf.slice(feats_padded, [4, 0], shape)

    delt1 = (r1 - l1) * 0.1 + (r2 - l2) * 0.2
    return delt1


def tf_kaldi_fea_delt2(features):
    feats_padded = tf.pad(features, [[1, 1], [0, 0]], "SYMMETRIC")
    feats_padded = tf.pad(feats_padded, [[1, 1], [0, 0]], "SYMMETRIC")
    feats_padded = tf.pad(feats_padded, [[1, 1], [0, 0]], "SYMMETRIC")
    feats_padded = tf.pad(feats_padded, [[1, 1], [0, 0]], "SYMMETRIC")

    shape = tf.shape(features)
    l4 = tf.slice(feats_padded, [0, 0], shape)
    l3 = tf.slice(feats_padded, [1, 0], shape)
    l2 = tf.slice(feats_padded, [2, 0], shape)
    l1 = tf.slice(feats_padded, [3, 0], shape)
    c = tf.slice(feats_padded, [4, 0], shape)
    r1 = tf.slice(feats_padded, [5, 0], shape)
    r2 = tf.slice(feats_padded, [6, 0], shape)
    r3 = tf.slice(feats_padded, [7, 0], shape)
    r4 = tf.slice(feats_padded, [8, 0], shape)

    delt2 = - 0.1 * c - 0.04 * (l1 + r1) + 0.01 * (l2 + r2) + 0.04 * (l3 + l4 + r4 + r3)
    return delt2


def add_delt(feature):
    fb = []
    fb.append(feature)
    delt1 = tf_kaldi_fea_delt1(feature)
    fb.append(delt1)
    delt2 = tf_kaldi_fea_delt2(feature)
    fb.append(delt2)
    return tf.concat(axis=1, values=fb)


def cmvn_global(feature, mean, var):
    fea = (feature - mean) / var
    return fea


def cmvn_utt(feature):
    fea_mean = tf.reduce_mean(feature, 0)
    fea_var = tf.reduce_mean(tf.square(feature), 0)
    fea_var = fea_var - fea_mean * fea_mean
    fea_ivar = tf.rsqrt(fea_var + 1E-12)
    fea = (feature - fea_mean) * fea_ivar
    return fea


def splice(features, left_num, right_num):
    """
    [[1,1,1], [2,2,2], [3,3,3], [4,4,4], [5,5,5], [6,6,6], [7,7,7]]
    left_num=0, right_num=2:
        [[1 1 1 2 2 2 3 3 3]
         [2 2 2 3 3 3 4 4 4]
         [3 3 3 4 4 4 5 5 5]
         [4 4 4 5 5 5 6 6 6]
         [5 5 5 6 6 6 7 7 7]
         [6 6 6 7 7 7 0 0 0]
         [7 7 7 0 0 0 0 0 0]]
    """
    shape = tf.shape(features)
    splices = []
    pp = tf.pad(features, [[left_num, right_num], [0, 0]])
    for i in range(left_num + right_num + 1):
        splices.append(tf.slice(pp, [i, 0], shape))
    splices = tf.concat(axis=1, values=splices)

    return splices


def down_sample(features, rate, axis=1):
    """
    features: batch x time x deep
    Notation: you need to set the shape of the output! tensor.set_shape(None, dim_input)
    """
    len_seq = tf.shape(features)[axis]

    return tf.gather(features, tf.range(len_seq, delta=rate), axis=axis)


def target_delay(features, num_target_delay):
    seq_len = tf.shape(features)[0]
    feats_part1 = tf.slice(features, [num_target_delay, 0], [seq_len-num_target_delay, -1])
    frame_last = tf.slice(features, [seq_len-1, 0], [1, -1])
    feats_part2 = tf.concat([frame_last for _ in range(num_target_delay)], axis=0)
    features = tf.concat([feats_part1, feats_part2], axis=0)

    return features


if __name__ == '__main__':
    x = tf.placeholder(tf.float32)  # 1-D tensor
    i = tf.placeholder(tf.float32)

    # y = splice_features(x,1,1)
    y = add_delt(x)
    # y = tf.slice(x, i, [1,1])
    # y = cmvn_features(x)

    # initialize
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # run
    result = sess.run(y, feed_dict={x: [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]})
    print(result)

    result = sess.run(y, feed_dict={x: [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]})
    print(result)
