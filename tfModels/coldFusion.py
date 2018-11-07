import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import itertools

def cold_fusion(logit_lm, state_decoder, num_cell_units, dim_output):
    """

    """
    # batch_size = tf.shape(logit_lm)[0]

    state_decoder = tf.concat(list(itertools.chain.from_iterable(state_decoder)), 1)

    hidden_output_lm = fully_connected(
        inputs=logit_lm,
        num_outputs=num_cell_units,
        activation_fn=tf.nn.relu,
        biases_initializer=None,
        scope='CF_layer_1')

    gate = fully_connected(
        inputs=tf.concat([state_decoder, hidden_output_lm], 1),
        num_outputs=1,
        activation_fn=tf.nn.relu,
        scope='CF_gate')

    state_coldfusion = tf.concat([state_decoder, gate * hidden_output_lm], 1)

    logits_coldfusion = fully_connected(
        inputs=state_coldfusion,
        num_outputs=dim_output,
        activation_fn=tf.nn.relu,
        biases_initializer=None,
        scope='CF_layer_2')

    return tf.nn.softmax(logits_coldfusion)
