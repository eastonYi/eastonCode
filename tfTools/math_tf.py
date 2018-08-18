import tensorflow as tf


def sum_log(*args):
    """
    Stable log sum exp.
    the input number is in log-scale, so as the return
    """
    # if all(a == LOG_ZERO for a in args):
    #     return LOG_ZERO
    a_max = tf.reduce_max(tf.concat([args], 1), 0)
    lsp = tf.log(tf.reduce_sum([tf.exp(a - a_max) for a in args], 0))
    return a_max + lsp


def testSum_log():
    import numpy as np

    a = tf.constant(np.log([[1.4, 0.2, 1e-11],
                            [1.4, 0.2, 1e-11]]))
    b = tf.constant(np.log([[9.6, 0.02, 1e-11],
                            [9.6, 0.02, 1e-11]]))

    ground_truth = np.array(np.log([11, 0.22, 2e-11]))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        results = sess.run(sum_log(a, b))
        print(results)
        print(ground_truth)
        assert np.allclose(results, ground_truth)
        

if __name__ == '__main__':
    testSum_log()
