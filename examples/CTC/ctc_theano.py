# Author: Ishaan Gulrajani
# License: BSD 3-clause
# https://gist.github.com/igul222/53df6c5dd6c8f3b5280d

import numpy
import theano
import theano.tensor as T

# log(0) = -infinity, but this leads to NaN errors in log_add and elsewhere,
# so we'll just use a large negative value.
_LOG_ZERO = numpy.float32(-100000)
_LOG_ONE = numpy.float32(0)

# Index of the output neuron corresponding to a "blank" prediction
_BLANK = T.cast(numpy.float32(0), 'int32')

# Padding character (for batching together sequences of different lengths)
# We pad with -1 because it's nonzero but still not a valid label
PADDING = -1


def _log_add(log_a, log_b):
    """Theano expression for log(a+b) given log(a) and log(b)."""
    # TODO fix potential axis bug here!!! (it might be subtracting the wrong vals)
    smaller = T.minimum(log_a, log_b)
    larger = T.maximum(log_a, log_b)
    return larger + T.log1p(T.exp(smaller - larger))


def _log_add_3(log_a, log_b, log_c):
    """Theano expression for log(a+b+c) given log(a), log(b), log(c)."""
    smaller = T.minimum(log_a, log_b)
    larger = T.maximum(log_a, log_b)
    largest = T.maximum(larger, log_c)
    larger = T.minimum(larger, log_c)

    return largest + T.log1p(
            T.exp(smaller - largest) +
            T.exp(larger - largest)
            )


def _initial_probabilities(example_count, target_length):
    """The initial value of the forward-backward
       recurrence: a matrix of shape (example_count, target_length) where
       each row is log([1,0,0,...])."""

    row = T.concatenate([
        T.alloc(_LOG_ONE, 1),
        T.alloc(_LOG_ZERO, target_length - 1)
    ])

    return T.shape_padleft(row).repeat(example_count, axis=0)


def _skip_allowed(ttargets):
    """A matrix of shape (example_count, target_length). For each example,
       values are log(1) if a transition is allowed from a given label to
       the next-to-next label (a "skip"), and log(0) otherwise.

       Skip conditions:
           a) next label is blank, and
           b) the next to next label is different from the current

       * note that (b) implies (a), so we really only need to check (b).
       (7.8) to judge which condition
    """

    ttargets = T.concatenate([
            ttargets,
            T.shape_padleft([_BLANK, _BLANK]).repeat(ttargets.shape[0], axis=0)
        ], axis=1)
    skip_allowed = T.neq(ttargets[:, :-2], ttargets[:, 2:])

    # Since the values of skip_allowed are all 1 or 0, we apply a linear
    # function that maps 1 to log(1) and 0 to log(0). This lets us use our
    # "special" val of log(0) and might just be faster than doing
    # T.log(skip_allowed).
    return (skip_allowed * (_LOG_ONE - _LOG_ZERO)) + _LOG_ZERO


def _forward_vars(activations, ttargets):
    """Calculate the CTC forward variables: for each example, a matrix of
       shape (sequence length, target length) where entry (t,u) corresponds
       to the log-probability of the network predicting the target sequence
       prefix [0:u] by time t."""

    ttargets = T.cast(ttargets, 'int32')

    activations = T.log(activations)

    # For each example, a matrix of shape (seq len, target len) with values
    # corresponding to activations at each time and sequence position.
    probs = activations[:, T.shape_padleft(T.arange(activations.shape[1])).T, ttargets]

    initial_probs = _initial_probabilities(
        probs.shape[1],
        ttargets.shape[1])

    skip_allowed = _skip_allowed(ttargets)

    def step(p_curr, p_prev):
        # 7.8 sum
        no_change = p_prev
        next_label = right_shift_rows(p_prev, 1, _LOG_ZERO)
        skip = right_shift_rows(p_prev + skip_allowed, 2, _LOG_ZERO)

        return p_curr + _log_add_3(no_change, next_label, skip)

    probabilities, _ = theano.scan(
        step,
        sequences=[probs],
        outputs_info=[initial_probs]
    )

    return probabilities


def cost(activations, ttargets, target_lengths):
    """Calculate the CTC cost: the mean of the negative log-probabilities of
       the correct labellings for each example.

       The targets passed in should have shape (examples, target length),
       which is the transpose of the usual shape. They should also have blanks
       inserted in between symbols, and at the beginning and end."""

    forward_vars = _forward_vars(activations, ttargets)

    # (7.4)
    target_lengths = T.cast(target_lengths, 'int32')
    lp_1 = forward_vars[-1, T.arange(forward_vars.shape[1]), 2*target_lengths - 1]
    lp_2 = forward_vars[-1, T.arange(forward_vars.shape[1]), 2*target_lengths]
    return -T.mean(_log_add(lp_1, lp_2))


def _best_path_decode(activations):
    """Calculate the CTC best-path decoding for a given activation sequence.
       In the returned matrix, shorter sequences are padded with -1s."""

    # For each timestep, get the highest output
    decoding = T.argmax(activations, axis=2)

    # prev_outputs[time][example] == decoding[time - 1][example]
    prev_outputs = T.concatenate([T.alloc(_BLANK, 1, decoding.shape[1]), decoding], axis=0)[:-1]

    # Filter all repetitions to zero (blanks are already zero)
    decoding = decoding * T.neq(decoding, prev_outputs)

    # Calculate how many blanks each sequence has relative to longest sequence
    blank_counts = T.eq(decoding, 0).sum(axis=0)
    min_blank_count = T.min(blank_counts, axis=0)
    max_seq_length = decoding.shape[0] - min_blank_count # used later
    padding_needed = blank_counts - min_blank_count

    # Generate the padding matrix by ... doing tricky things
    max_padding_needed = T.max(padding_needed, axis=0)
    padding_needed = padding_needed.dimshuffle('x', 0).repeat(max_padding_needed, axis=0)
    padding = T.arange(max_padding_needed).dimshuffle(0, 'x').repeat(decoding.shape[1], axis=1)
    padding = PADDING * T.lt(padding, padding_needed)

    # Apply the padding
    decoding = T.concatenate([decoding, padding], axis=0)

    # Remove zero values
    nonzero_vals = decoding.T.nonzero_values()
    decoding = T.reshape(nonzero_vals, (decoding.shape[1], max_seq_length)).T

    return decoding


def accuracy(activations, targets, target_lengths):
    """Calculate the mean accuracy of a given set of activations w.r.t. given
       (un-transposed) targets."""

    targets = T.cast(targets, 'int32')

    best_paths = _best_path_decode(activations)
    distances = levenshtein_distances(targets, best_paths, PADDING)
    return numpy.float32(1.0) - T.mean(distances / target_lengths)


def transform_targets(targets):
    """Transform targets into a format suitable for passing to cost()."""

    reshaped = T.shape_padleft(targets)
    blanks = T.fill(reshaped, _BLANK)
    result = T.concatenate([blanks, reshaped]).dimshuffle(1, 0, 2).reshape((2*targets.shape[0], targets.shape[1]))
    result = T.concatenate([result, T.shape_padleft(result[0])])
    return result


def levenshtein_distances(a, b, padding_val):
    a = T.as_tensor_variable(a)
    b = T.as_tensor_variable(b)

    n_strings = a.shape[1]
    i_max = b.shape[0] + a.shape[0]
    j_max = a.shape[0]

    def step(i, prev, prev_prev):
        j = T.arange(j_max + 1)

        x = (i - j) % (b.shape[0] + 1)
        y = j

        # The main rule for filling the matrix:
        # If current a == current b, keep the val from (x-1,y-1).
        # Otherwise new val = min(upper, left, upper-left) + 1.
        new_vals = T.switch(
            T.eq(b[x-1], a[y-1]),
            right_shift(prev_prev, 1),
            T.min([
                prev,
                right_shift(prev, 1),
                right_shift(prev_prev, 1),
            ], axis=0) + 1
        )

        # ... but if current a == padding, keep the val from (x, y-1)
        new_vals = T.switch(
            T.eq(a[y-1], padding_val),
            right_shift(prev, 1),
            new_vals
        )
        # ... likewise if current b == padding, keep the val from (x-1, y)
        new_vals = T.switch(
            T.eq(b[x-1], padding_val),
            prev,
            new_vals
        )
        # (If they're both padding, it doesn't matter which val we keep.)

        # Finally, if the value corresponds to an initial value of the matrix,
        # use that instead of anything above.
        is_initial = T.any([T.eq(j, 0), T.eq(j, i)], axis=0)
        is_initial = T.shape_padright(is_initial).repeat(n_strings, axis=1)
        initial_val = T.cast(i, theano.config.floatX)
        new_vals = T.switch(
            is_initial,
            initial_val,
            new_vals
        )

        return new_vals

    results, _ = theano.scan(
        step,
        sequences=[T.arange(i_max + 1)],
        outputs_info=dict(initial=T.zeros((2, j_max + 1, n_strings)), taps=[-1, -2])
    )

    return results[-1, -1, :]#.T


def right_shift(x, shift, pad_val=numpy.float32(0)):
    """Right-shift along the first dimension of a matrix, padding the left with
       a given value."""
    return T.concatenate([
            T.alloc(pad_val, shift, x.shape[1]),
            x[:-shift]
        ], axis=0)


def right_shift_rows(x, shift, pad_val=numpy.float32(0)):
    """Right-shift rows in a matrix, padding the left with a given value."""
    return T.concatenate([
            T.alloc(pad_val, x.shape[0], shift),
            x[:, :-shift]
        ], axis=1)
