import tensorflow as tf


def smoothing_cross_entropy(logits, labels, vocab_size, confidence):
  """Cross entropy with label smoothing to limit over-confidence."""
  with tf.name_scope("smoothing_cross_entropy", [logits, labels]):
    # Low confidence is given to all non-true labels, uniformly.
    low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 1)
    # Normalizing constant is the best cross-entropy value with soft targets.
    # We subtract it just for readability, makes no difference on learning.
    normalizing = -(confidence * tf.log(confidence) + tf.to_float(
        vocab_size - 1) * low_confidence * tf.log(low_confidence + 1e-20))
    # Soft targets.
    soft_targets = tf.one_hot(
        tf.cast(labels, tf.int32),
        depth=vocab_size,
        on_value=confidence,
        off_value=low_confidence)
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=soft_targets)
    return xentropy - normalizing

#============================================================================
#  building model
#============================================================================
def choose_device(op, device, default_device):
    if op.type.startswith('Variable'):
        device = default_device
    return device

def l2_penalty(iter_variables):
    l2_penalty = 0
    for v in iter_variables:
        if 'biase' not in v.name:
            l2_penalty += tf.nn.l2_loss(v)
    return l2_penalty

#============================================================================
#  learning rate
#============================================================================
def lr_decay_with_warmup(global_step, warmup_steps, hidden_units):
    """Inverse-decay learning rate until warmup_steps, then decay."""
    warmup_steps = tf.to_float(warmup_steps)
    global_step = tf.to_float(global_step)
    return hidden_units ** -0.5 * tf.minimum(
        (global_step + 1.0) * warmup_steps ** -1.5, (global_step + 1.0) ** -0.5)


def warmup_exponential_decay(global_step, warmup_steps, peak, decay_rate, decay_steps):
    warmup_steps = tf.to_float(warmup_steps)
    global_step = tf.to_float(global_step)
    # return peak * global_step / warmup_steps
    return tf.where(global_step <= warmup_steps,
                    peak * global_step / warmup_steps,
                    peak * decay_rate ** ((global_step - warmup_steps) / decay_steps))


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """
    global_step: int64 (scalar) tensor representing global step.
    learning_rate_base: base learning rate.
    total_steps: total number of training steps.
    warmup_learning_rate: initial learning rate for warm up.
    warmup_steps: number of warmup steps.
    hold_base_rate_steps: Optional number of steps to hold base learning rate
      before decaying.
    """
    if learning_rate_base < warmup_learning_rate:
        raise ValueError('learning_rate_base must be larger '
                     'or equal to warmup_learning_rate.')
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                     'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
                          3.1416 *
                          (tf.cast(global_step, tf.float32) - warmup_steps - hold_base_rate_steps)
                          / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = tf.where(global_step > warmup_steps + hold_base_rate_steps,
                             learning_rate, learning_rate_base)
    if warmup_steps > 0:
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * tf.cast(global_step, tf.float32) + warmup_learning_rate
        learning_rate = tf.where(global_step < warmup_steps, warmup_rate, learning_rate)

    return tf.where(global_step > total_steps, 0.0, learning_rate, name='learning_rate')


def create_embedding(size_vocab, size_embedding, name='embedding'):
    with tf.device("/cpu:0"):
        embed_table = tf.get_variable(name, [size_vocab, size_embedding])

    return embed_table


if __name__ == '__main__':
    global_step = tf.train.get_or_create_global_step()
    op_add = tf.assign_add(global_step, 1)
    # lr = cosine_decay_with_warmup(global_step, 0.1, 100000, 0.01, 10, 20)
    lr = lr_decay_with_warmup(
        global_step,
        warmup_steps=10000,
        hidden_units=256)
    # lr = warmup_exponential_decay(global_step,
    #                               warmup_steps=10000,
    #                               peak=0.001,
    #                               decay_rate=0.5,
    #                               decay_steps=1000)

    with tf.train.MonitoredTrainingSession() as sess:
        list_x = []; list_y = []
        for i in range(50000):
            x, y = sess.run([global_step, lr])
            list_x.append(x)
            list_y.append(y)
            sess.run(op_add)

    import matplotlib.pyplot as plt
    plt.plot(list_x, list_y)
    plt.show()
