"""
refer to https://r2rt.com/styles-of-truncated-backpropagation.html
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.models.rnn.ptb import reader

#data from http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
raw_data = reader.ptb_raw_data('ptb_data')
train_data, val_data, test_data, num_classes = raw_data

def gen_epochs(n, num_steps, batch_size):
    for i in range(n):
        yield reader.ptb_iterator(train_data, batch_size, num_steps)


def build_graph(num_steps,
                bptt_steps = 4, batch_size = 200, num_classes = num_classes,
                state_size = 4, embed_size = 50, learning_rate = 0.01):
    """
    Builds graph for a simple RNN

    Notable parameters:
    num_steps: sequence length / steps for TF-style truncated backprop
    bptt_steps: number of steps for true truncated backprop
    """

    g = tf.get_default_graph()

    # placeholders
    x = tf.placeholder(tf.int32, [batch_size, None], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, None], name='labels_placeholder')
    default_init_state = tf.zeros([batch_size, state_size])
    init_state = tf.placeholder_with_default(default_init_state,
                                             [batch_size, state_size], name='state_placeholder')
    dropout = tf.placeholder(tf.float32, [], name='dropout_placeholder')

    x_one_hot = tf.one_hot(x, num_classes)
    x_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, num_steps, x_one_hot)]

    with tf.variable_scope('embeddings'):
        embeddings = tf.get_variable('embedding_matrix', [num_classes, embed_size])

    def embedding_lookup(one_hot_input):
        with tf.variable_scope('embeddings', reuse=True):
            embeddings = tf.get_variable('embedding_matrix', [num_classes, embed_size])
            embeddings = tf.identity(embeddings)
            g.add_to_collection('embeddings', embeddings)
            return tf.matmul(one_hot_input, embeddings)

    rnn_inputs = [embedding_lookup(i) for i in x_as_list]

    #apply dropout to inputs
    rnn_inputs = [tf.nn.dropout(x, dropout) for x in rnn_inputs]

    # rnn_cells
    with tf.variable_scope('rnn_cell'):
        W = tf.get_variable('W', [embed_size + state_size, state_size])
        b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))

    def rnn_cell(rnn_input, state):
        with tf.variable_scope('rnn_cell', reuse=True):

            W = tf.get_variable('W', [embed_size + state_size, state_size])
            W = tf.identity(W)
            g.add_to_collection('Ws', W)

            b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
            b = tf.identity(b)
            g.add_to_collection('bs', b)

            return tf.tanh(tf.matmul(tf.concat(1, [rnn_input, state]), W) + b)

    state = init_state
    rnn_outputs = []
    for rnn_input in rnn_inputs:
        state = rnn_cell(rnn_input, state)
        rnn_outputs.append(state)

    #apply dropout to outputs
    rnn_outputs = [tf.nn.dropout(x, dropout) for x in rnn_outputs]

    final_state = rnn_outputs[-1]

    #logits and predictions
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W_softmax', [state_size, num_classes])
        b = tf.get_variable('b_softmax', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
    predictions = [tf.nn.softmax(logit) for logit in logits]

    #losses
    y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, num_steps, y)]
    losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logit,label) \
              for logit, label in zip(logits, y_as_list)]
    total_loss = tf.reduce_mean(losses)

    """
    Implementation of true truncated backprop using TF's high-level gradients function.

    Because I add gradient-ops for each error, this are a number of duplicate operations,
    making this a slow implementation. It would be considerably more effort to write an
    efficient implementation, however, so for testing purposes, it's OK that this goes slow.

    An efficient implementation would still require all of the same operations as the full
    backpropagation through time of errors in a sequence, and so any advantage would not come
    from speed, but from having a better distribution of backpropagated errors.
    """

    embed_by_step = g.get_collection('embeddings')
    Ws_by_step = g.get_collection('Ws')
    bs_by_step = g.get_collection('bs')

    # Collect gradients for each step in a list
    embed_grads = []
    W_grads = []
    b_grads = []

    # Keeping track of vanishing gradients for my own curiousity
    vanishing_grad_list = []

    # Loop through the errors, and backpropagate them to the relevant nodes
    for i in range(num_steps):
        start = max(0,i+1-bptt_steps)
        stop = i+1
        grad_list = tf.gradients(losses[i],
                                 embed_by_step[start:stop] +\
                                 Ws_by_step[start:stop] +\
                                 bs_by_step[start:stop])
        embed_grads += grad_list[0 : stop - start]
        W_grads += grad_list[stop - start : 2 * (stop - start)]
        b_grads += grad_list[2 * (stop - start) : ]

        if i >= bptt_steps:
            vanishing_grad_list.append(grad_list[stop - start : 2 * (stop - start)])

    grad_embed = tf.add_n(embed_grads) / (batch_size * bptt_steps)
    grad_W = tf.add_n(W_grads) / (batch_size * bptt_steps)
    grad_b = tf.add_n(b_grads) / (batch_size * bptt_steps)

    """
    Training steps
    """

    opt = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars_tf_style = opt.compute_gradients(total_loss, tf.trainable_variables())
    grads_and_vars_true_bptt = \
        [(grad_embed, tf.trainable_variables()[0]),
         (grad_W, tf.trainable_variables()[1]),
         (grad_b, tf.trainable_variables()[2])] + \
        opt.compute_gradients(total_loss, tf.trainable_variables()[3:])
    train_tf_style = opt.apply_gradients(grads_and_vars_tf_style)
    train_true_bptt = opt.apply_gradients(grads_and_vars_true_bptt)

    return dict(
        train_tf_style = train_tf_style,
        train_true_bptt = train_true_bptt,
        gvs_tf_style = grads_and_vars_tf_style,
        gvs_true_bptt = grads_and_vars_true_bptt,
        gvs_gradient_check = opt.compute_gradients(losses[-1], tf.trainable_variables()),
        loss = total_loss,
        final_state = final_state,
        x=x,
        y=y,
        init_state=init_state,
        dropout=dropout,
        vanishing_grads=vanishing_grad_list
    )
