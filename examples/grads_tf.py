import tensorflow as tf
import numpy as np

a = tf.Variable([2.0], 'a')
b = tf.Variable([5.0], 'b')
c = tf.Variable([5.0], 'c')

output = a * 3.0 + b

def testA():
    loss = tf.square(output - 10.0)

    g1 = tf.gradients(loss, [output])
    g2 = tf.gradients(output, [a, b])
    g3 = tf.gradients(loss, [a, b])

    optimizer_adam = tf.train.AdamOptimizer(0.008)
    grads_adam = optimizer_adam.compute_gradients(tf.cast(loss, dtype=tf.float32))
    print(grads_adam)
    #
    # optimizer_mon = tf.train.RMSPropOptimizer(0.008)
    # grads_mon = optimizer_mon.compute_gradients(tf.cast(loss, dtype=tf.float32))
    # print(grads_mon)

    with tf.train.MonitoredTrainingSession() as sess:
        print('g1: ', sess.run(g1))
        print('g2: ', sess.run(g2))
        print('g3: ', sess.run(g3))
        # print("grads_adam: ", sess.run(grads_adam[0][0]), sess.run(grads_adam[1][0]))
        # print("grads_mon: ", sess.run(grads_mon[0][0]), sess.run(grads_mon[1][0]))

def testB():
    op_1 = a*3.0 + 7.0*b
    op_2 = 2.0*a + 5.0*c
    output = op_1 * op_2

    g1 = tf.gradients(output, [op_1, op_2])
    g2 = tf.gradients(op_1, [a, b, c])
    g3 = tf.gradients(op_2, [a, b, c])
    g4 = tf.gradients(output, [a, b, c])
    g5 = tf.gradients([op_1, op_2], [a, b, c], grad_ys=g1)

    with tf.train.MonitoredTrainingSession() as sess:
        print('g1: ', sess.run(g1))
        print('g2: ', sess.run(g2[:2]))
        print('g3: ', sess.run([g3[0], g3[2]]))
        print('g4: ', sess.run(g4))
        print('g5: ', sess.run(g5))


def testC():

    a = tf.Variable([[1.2, 3.2],
                     [7.6, 4.2]], 'a')
    b = tf.Variable([[4.0, 6.1],
                     [2.8, 3.9]], 'b')
    c = tf.Variable([[9.3, 3.1],
                     [2.5, 1.2]], 'c')

    op_1 = a*3.0 + 7.0*b
    op_2 = 2.0*a + 5.0*c
    output = op_1 * op_2

    g1 = tf.gradients(output, [op_1, op_2])
    g2 = tf.gradients(op_1, [a, b, c])
    g3 = tf.gradients(op_2, [a, b, c])
    g4 = tf.gradients(output, [a, b, c])

    with tf.train.MonitoredTrainingSession() as sess:
        print('g1: ', sess.run(g1))
        print('g2: ', sess.run(g2[:2]))
        print('g3: ', sess.run([g3[0], g3[2]]))
        print('g4: ', sess.run(g4))


if __name__ == '__main__':
    testA()
