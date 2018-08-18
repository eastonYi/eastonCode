"""
Author: Awni Hannun
This is an example CTC decoder written in Python. The code is
intended to be a simple example and is not designed to be
especially efficient.
The algorithm is a prefix beam search for a model trained
with the CTC loss function.
For more details checkout either of these references:
  https://distill.pub/2017/ctc/#inference
  https://arxiv.org/abs/1408.2873
"""

import numpy as np
import math
import collections

NEG_INF = -9e99


def make_new_beam():
    fn = lambda: (NEG_INF, NEG_INF)
    return collections.defaultdict(fn)


def logsumexp(*args):
    """
    Stable log sum exp.
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max)
                       for a in args))
    return a_max + lsp


def decode(probs, beam_size=100, blank=0):
    """
    Performs inference for the given output probabilities.
    Arguments:
        probs: The output probabilities (e.g. post-softmax) for each
          time step. Should be an array of shape (time x output dim).
        beam_size (int): Size of the beam to use during inference.
        blank (int): Index of the CTC blank label.
    Returns the output label sequence and the corresponding negative
    log-likelihood estimated by the decoder.
    """
    T, S = probs.shape
    probs = np.log(probs)

    # Elements in the beam are (prefix, (p_blank, p_no_blank))
    # Initialize the beam with the empty sequence, a probability of
    # 1 for ending in blank and zero for ending in non-blank
    # (in log space).
    beam = [(tuple(), (0.0, NEG_INF))]

    for t in range(T):  # Loop over time

        # A default dictionary to store the next step candidates.
        next_beam = make_new_beam()

        for s in range(S):  # Loop over vocab
            p = probs[t, s]

            # The variables p_b and p_nb are respectively the
            # probabilities for the prefix given that it ends in a
            # blank and does not end in a blank at this time step.
            for prefix, (p_b, p_nb) in beam:  # Loop over beam

                # If we propose a blank the prefix doesn't change.
                # Only the probability of ending in blank gets updated.
                if s == blank:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
                    next_beam[prefix] = (n_p_b, n_p_nb)
                    continue

                # Extend the prefix by the new character s and add it to
                # the beam. Only the probability of not ending in blank
                # gets updated.
                end_t = prefix[-1] if prefix else None
                n_prefix = prefix + (s,)
                n_p_b, n_p_nb = next_beam[n_prefix]
                if s != end_t:
                    n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
                else:
                    # We don't include the previous probability of not ending
                    # in blank (p_nb) if s is repeated at the end. The CTC
                    # algorithm merges characters not separated by a blank.
                    n_p_nb = logsumexp(n_p_nb, p_b + p)

                # *NB* this would be a good place to include an LM score.
                next_beam[n_prefix] = (n_p_b, n_p_nb)

                # If s is repeated at the end we also update the unchanged
                # prefix. This is the merging case.
                if s == end_t:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_nb = logsumexp(n_p_nb, p_nb + p)
                    next_beam[prefix] = (n_p_b, n_p_nb)

        # Sort and trim the beam before moving on to the
        # next time-step.

        beam = sorted(next_beam.items(),
                      key=lambda x: logsumexp(*x[1]),
                      reverse=True)
        beam = beam[:beam_size]

    best = beam[0]
    return best[0], -logsumexp(*best[1])


def testdDecode():
    # np.random.seed(3)
    #
    # time = 50
    # output_dim = 20
    #
    # probs = np.random.rand(time, output_dim)
    # probs = probs / np.sum(probs, axis=1, keepdims=True)
    #
    # labels, score = decode(probs)
    # print("Score {:.3f}".format(score))
    vocab_list = ['\'', ' ', 'a', 'b', 'c', 'd', '_']
    beam_size = 20
    activation1 = [[
            0.06390443, 0.21124858, 0.27323887, 0.06870235, 0.0361254,
            0.18184413, 0.16493624
        ], [
            0.03309247, 0.22866108, 0.24390638, 0.09699597, 0.31895462,
            0.0094893, 0.06890021
        ], [
            0.218104, 0.19992557, 0.18245131, 0.08503348, 0.14903535,
            0.08424043, 0.08120984
        ], [
            0.12094152, 0.19162472, 0.01473646, 0.28045061, 0.24246305,
            0.05206269, 0.09772094
        ], [
            0.1333387, 0.00550838, 0.00301669, 0.21745861, 0.20803985,
            0.41317442, 0.01946335
        ], [
            0.16468227, 0.1980699, 0.1906545, 0.18963251, 0.19860937,
            0.04377724, 0.01457421
        ]]
    activation2 = [[
            0.08034842, 0.22671944, 0.05799633, 0.36814645, 0.11307441,
            0.04468023, 0.10903471
        ], [
            0.09742457, 0.12959763, 0.09435383, 0.21889204, 0.15113123,
            0.10219457, 0.20640612
        ], [
            0.45033529, 0.09091417, 0.15333208, 0.07939558, 0.08649316,
            0.12298585, 0.01654384
        ], [
            0.02512238, 0.22079203, 0.19664364, 0.11906379, 0.07816055,
            0.22538587, 0.13483174
        ], [
            0.17928453, 0.06065261, 0.41153005, 0.1172041, 0.11880313,
            0.07113197, 0.04139363
        ], [
            0.15882358, 0.1235788, 0.23376776, 0.20510435, 0.00279306,
            0.05294827, 0.22298418
        ]]
    greedy_result = ["ac'bdc", "b'da"]
    beam_search_result = ['acdc', "b'a"]

    list_activations = np.asarray([activation1, activation2], dtype=np.float32)
    print(list_activations.shape)
    results = [decode(activ, beam_size=20, blank=6) for activ in list_activations]

    for result, score in results:
        result_label = ''.join([vocab_list[idx] for idx in result])
        print(result_label, score)


if __name__ == "__main__":
    # np.random.seed(3)
    #
    # time = 50
    # output_dim = 20
    #
    # probs = np.random.rand(time, output_dim)
    # probs = probs / np.sum(probs, axis=1, keepdims=True)
    #
    # labels, score = decode(probs)
    # print("Score {:.3f}".format(score))
    testdDecode()
