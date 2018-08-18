import math
import numpy as np


def lr_decay_with_warmup(global_step, warmup_steps, hidden_units=512):
    """Inverse-decay learning rate until warmup_steps, then decay."""
    return hidden_units ** -0.5 * min(
        (global_step + 1.0) * warmup_steps ** -1.5, (global_step + 1.0) ** -0.5)


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
    learning_rate = 0.5 * learning_rate_base * (1 + math.cos(
                          math.pi * (global_step - warmup_steps - hold_base_rate_steps)
                          / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = learning_rate if global_step > warmup_steps + hold_base_rate_steps\
                        else learning_rate_base
    if warmup_steps > 0:
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = warmup_rate if global_step < warmup_steps else learning_rate

    return 0.0 if global_step > total_steps else learning_rate


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.plot([5*lr_decay_with_warmup(x, 10000) for x in range(100000)])
    plt.plot([0.7*lr_decay_with_warmup(x, 8000) for x in range(100000)])
    plt.show()
