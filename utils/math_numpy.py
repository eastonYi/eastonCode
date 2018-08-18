import numpy as np


def sum_log(*args):
    """
    Stable log sum exp.
    the input number is in log-scale, so as the return
    """
    # if all(a == LOG_ZERO for a in args):
    #     return LOG_ZERO
    a_max = np.max(args, 0)
    lsp = np.log(np.sum([np.exp(a - a_max) for a in args], 0))
    return a_max + lsp


def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """
    # make X at least 2d
    y = np.atleast_2d(X)
    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    # multiply y against the theta parameter,
    y = y * float(theta)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def Monte_Carlo_sampling(distribution, size=1):
    if distribution.ndim == 2:
        list_action = []
        for probs in distribution:
            list_action.append(np.random.choice(len(probs), size=size, p=probs))

        return np.asarray(list_action)

    elif distribution.ndim == 3:
        batch_actions = []
        for distrb in distribution:
            batch_actions.append(Monte_Carlo_sampling(distrb, size))

        return np.asarray(batch_actions)


def testSum_log():
    a = np.array(np.log([1.4, 0.2, 1e-11]))
    b = np.array(np.log([9.6, 0.02, 1e-11]))

    ground_truth = np.array(np.log([11, 0.22, 2e-11]))

    print(sum_log(a, b))
    assert np.allclose(sum_log(a, b), ground_truth)


if __name__ == '__main__':
    testSum_log()
