import tensorflow as tf
import numpy as np

def test_gather():
    # http://www.riptutorial.com/tensorflow/example/29038/numpy-like-indexing-using-tensors
    # data is [[0, 1, 2, 3, 4, 5],
    #          [6, 7, 8, 9, 10, 11],
    #          [12 13 14 15 16 17],
    #          [18 19 20 21 22 23],
    #          [24, 25, 26, 27, 28, 29]]
    data = np.reshape(np.arange(30), [5, 6])
    a = [1, 3]
    b = [2, 2]
    selected = data[a, b]
    print(selected) # [ 8 20]

    result0 = tf.gather(data, [1, 2])
    result1 = tf.gather_nd(data, [1, 2])
    result2 = tf.gather_nd(data, [[1, 2], [4, 3], [2, 5]])

    x = tf.constant(data)
    idx1 = tf.constant(a)
    idx2 = tf.constant(b)
    result = tf.gather_nd(x, tf.stack((idx1, idx2), -1))

    with tf.Session() as sess:
        print(sess.run(result))
        print(sess.run(result0))
        print(sess.run(result1))
        print(sess.run(result2))

def test_scan():
    a = tf.constant([0,0])
    # b = tf.constant([3,3,3,3,3])
    # b = tf.constant([[1,2], [3,4], [5,6], [7,8], [9,10]])

    b = [tf.constant([1,2]), tf.constant([3,4]), tf.constant([5,6]), tf.constant([7,8]), tf.constant([9,10])]

    def step(var, inputs):
        a0 = inputs[0]
        b0 = inputs[1][4]

        return a0+b0

    result = tf.scan(step, (a, b), (0))

    with tf.Session() as sess:
        print(sess.run(result))

def test_map_fn():
    """
    dtype: (optional) The output type(s) of fn.
    If fn returns a structure of Tensors differing from the structure of elems,
    then dtype is not optional and must have the same structure as the output of fn.
    """

    a = tf.constant([1,2,3,4,5])
    b = tf.constant([1,2,3,4,5])
    # b = [(1,2), (3,4), (5,6), (7,8), (9,10)]

    def step(inputs):
        return inputs[0] * inputs[1]

    result = tf.map_fn(step, (a, b), dtype=tf.int32)

    with tf.Session() as sess:
        sess.run(result)


if __name__ == '__main__':
    test_scan()
