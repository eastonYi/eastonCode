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


def softmax(input):
    """
    Compute the softmax of each element along an axis of X.
    """
    list_value = []
    len_compute = input.shape[-1]
    shape_input = input.shape
    for x in input.reshape(-1, len_compute):
        # print(x)
        e_x = np.exp(x - np.max(x))
        res = e_x / e_x.sum(axis=0)
        list_value.append(res)

    return np.array(list_value).reshape(shape_input)


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
