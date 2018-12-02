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

`num_classes` = `num_labels + 1` classes
For example, for a vocabulary containing 3 labels `[a, b, c]`,
`num_classes = 4` and the labels indexing is `{blank: 0, a: 1, b: 2, c: 3}`.
"""

import numpy as np
from tqdm import tqdm
np.set_printoptions(suppress=True)
import collections

from utils.math_numpy import sum_log

ID_BLANK = 0
LOG_ZERO = -1e49
LOG_ONE = 0.0


def ctc_loss(batch_activations, seqs_labels, seq_fea_lens, seq_label_lens, blank=ID_BLANK):
    """
    If using this ctc_loss based on numpy, you unfortunately need to compute the grads by yourself
    y/batch_activations: [num_batch, len_seq_blanked_labels], such as: _c_a_t_, _d_e_e_p_,
    seqs_labels
    id_blank
    """
    def add_epsilon(y, blank):
        list_concat = []
        for i in range(y.shape[1]):
            list_concat.append(np.ones((y.shape[0], 1), dtype=np.int32) * blank)
            list_concat.append(y[:, i].reshape(-1, 1))
        list_concat.append(np.ones((y.shape[0], 1), dtype=np.int32) * blank)

        return np.concatenate(list_concat, 1)

    def right_shift_rows(p, shift, pad):
        assert type(shift) is int
        return np.concatenate([np.ones((p.shape[0], shift))*pad, p[:, :-shift]], axis=1)

    def add3_allowed(y):
        """
        y: seqs_labels
        y1: y skip2
        """
        y1 = np.concatenate([np.ones((y.shape[0], 2), dtype=np.int32) * blank, y], axis=1)[:, :-2]
        skip_allowed = np.not_equal(y1, y) * np.not_equal(y, blank) * np.not_equal(y1, blank)

        # (skip_allowed * (LOG_ONE - LOG_ZERO)) + LOG_ZERO
        return skip_allowed

    def step(probs, forward_vars):
        nonlocal mask_add3
        add_2 = right_shift_rows(forward_vars, 1, LOG_ZERO)
        add_3 = right_shift_rows(forward_vars, 2, LOG_ZERO) + (1-mask_add3) * LOG_ZERO
        # print('(1-mask_add3): ', (1-mask_add3)[1]*LOG_ZERO)
        # print('mask_add3: ', add_3[1])
        #return probs + sum_log(forward_vars, add_2, add_3)
        return np.asarray([a + sum_log(b, c, d) for a, b, c, d in zip(probs, forward_vars, add_2, add_3)])

    size_batch = batch_activations.shape[0]
    seqs_labels = np.asarray(seqs_labels, dtype=np.int32)
    seq_label_lens = np.asarray(seq_label_lens, dtype=np.int32)
    seq_fea_lens = np.asarray(seq_fea_lens, dtype=np.int32)
    assert len(seq_label_lens) == len(seqs_labels) == size_batch
    batch_activations_log = np.log(batch_activations)
    seqs_labels_blanked = add_epsilon(seqs_labels, blank)
    # print('seqs_labels_blanked: ', seqs_labels_blanked[1])

    mask_add3 = add3_allowed(seqs_labels_blanked)
    seqs_probs = activations2seqs_probs(batch_activations_log, seqs_labels_blanked)

    seqs_probs_timeMajor = seqs_probs.transpose((1, 0, 2))

    forward_vars_init = np.ones(seqs_labels_blanked.shape, dtype=np.float32) * LOG_ZERO
    forward_vars_init[:, 0] = LOG_ONE

    forward_vars = [forward_vars_init]
    # forward
    for time, probs in enumerate(seqs_probs_timeMajor):
        # probs: [size_batch, num_labels]
        forward_vars.append(step(probs, forward_vars[-1]))
        print(time, ': ', np.exp(np.around(forward_vars[-1], 3)[1][:5]))

    forward_vars = np.asarray(forward_vars, dtype=np.float32)
    indices_time = seq_fea_lens
    indices_batch = np.arange(size_batch)
    print('\na', forward_vars[indices_time, indices_batch, 2*seq_label_lens-1],
          '\nb', forward_vars[indices_time, indices_batch, 2*seq_label_lens])
    return -sum_log(forward_vars[indices_time, indices_batch, 2*seq_label_lens-1],
                    forward_vars[indices_time, indices_batch, 2*seq_label_lens])
    # return -np.mean(sum_log(forward_vars[:, -1], forward_vars[:, -2]))


def cer_aligns(aligns, ref, f_distance):
    """
    aligns: time x sample
    """
    list_label_samples = ctc_reduce_map(aligns.T) # sample x time
    # from concurrent.futures import ThreadPoolExecutor
    # from itertools import repeat
    # num_samples = len(list_label_samples)
    # with ThreadPoolExecutor(num_samples) as ex:
    #     list_cer = ex.map(cer, repeat(ref, num_samples), list_label_samples)
    # return np.mean([min(d/(len(ref)*1.0), 0.99) for d in list_cer])
    list_cer = []
    for sample in list_label_samples:
        sample = sample[:2*len(ref)]
        d = f_distance(ref, sample)
        list_cer.append(min(d/(len(ref)*1.0), 0.99))

    return np.mean(list_cer)


def sample_aligns(distribution, size=1):
    """
    return:
        aligns: time x size
        or
        batch_aligns: batch x time x size
    """
    if distribution.ndim == 2:
        list_action = []
        for probs in distribution:
            list_action.append(np.random.choice(len(probs), size=size, p=probs))
        return np.asarray(list_action)
    elif distribution.ndim == 3:
        batch_actions = []
        for distrb in distribution:
            batch_actions.append(sample_aligns(distrb, size))
        return np.asarray(batch_actions)


def ctc_reduce_map(batch_samples, blank_id):
    """
    inputs:
        batch_samples: size x time
    return:
        (padded_samples, mask): (size x max_len, size x max_len)
                                 max_len <= time
    """
    from utils.tools import padding_list_seqs

    sents = []
    for align in batch_samples:
        sent = []
        tmp = None
        for token in align:
            if token != blank_id and token != tmp:
                sent.append(token)
            tmp = token
        sents.append(sent)

    return padding_list_seqs(sents, dtype=np.int32, pad=0)


def rna_reduce_map(batch_samples, blank_id):
    """
    inputs:
        batch_samples: size x time
    return:
        (padded_samples, mask): (size x max_len, size x max_len)
                                 max_len <= time
    """
    from utils.tools import padding_list_seqs

    sents = []
    for align in batch_samples:
        sent = []
        for token in align:
            if token == 0:
                break
            if token != blank_id:
                sent.append(token)
        sents.append(sent)

    return padding_list_seqs(sents, dtype=np.int32, pad=0)


def ctc_decode(activaties, beam_size=10, blank=None):
    """
    Performs inference for the given output probabilities.
    decode does not need to be batching up

    Arguments:
            activaties: The output probabilities (e.g. post-softmax) for each
                time step. Should be an array of shape (time x output dim).
            beam_size (int): Size of the beam to use during inference.
            blank (int): Index of the CTC blank label.

    Returns the output label sequence and the corresponding negative
    log-likelihood estimated by the decoder.

    When we extend [a] to produce [a,a], we only want include the part of the
    previous score for alignments which end in \epsilon.
    Similarly, when we don’t extend the preﬁx and produce [a], we should only
    include the part of the previous score for alignments which don’t end in \epsilon.
    Given this, we have to keep track of two probabilities for each preﬁx in the beam.
    The probability of all alignments which end in \epsilonϵ and the probability
    of all alignments which don’t end in \epsilon.ϵ. When we rank the hypotheses
    at each step before pruning the beam, we’ll use their combined scores.

    The variables p_b and p_nb are respectively the probabilities for the prefix
    given that it ends in a blank and does not end in a blank at this time step.
    """
    T, S = activaties.shape
    if not blank:
        blank = S-1
    activaties_log = np.log(activaties)
    # Elements in the beam are (prefix, (p_blank, p_no_blank))
    # Initialize the beam with the empty sequence, a probability of
    # 1 for ending in blank and zero for ending in non-blank (in log space).
    beam = [(tuple(), (LOG_ONE, LOG_ZERO))]

    for t in range(T): # Loop over time
        # A default dictionary to store the next step candidates.
        beam_extend = collections.defaultdict(lambda: [LOG_ZERO, LOG_ZERO])
        for s in range(S): # Loop over vocab
            p = activaties_log[t, s]
            for prefix, (p_b, p_nb) in beam: # Loop over beam
                # If we propose a blank the prefix doesn't change.
                # Only the probability of ending in blank gets updated.
                if s == blank:
                    beam_extend[prefix][0] = sum_log(beam_extend[prefix][0], p_b + p, p_nb + p)
                    continue
                # Extend the prefix by the new character s and add it to the beam.
                # Only the probability of not ending in blank gets updated.
                end_t = prefix[-1] if prefix else None
                prefix_extend = prefix + (s,)

                if s == end_t:
                    # When we extend [a] to produce [a,a], we only want include the part of the
                    # previous score for alignments which end in \epsilon.
                    beam_extend[prefix_extend][1] = sum_log(beam_extend[prefix_extend][1], p_b + p)
                    # If s is repeated at the end we also update the unchanged prefix.
                    # This is the merging case.
                    beam_extend[prefix][1] = sum_log(beam_extend[prefix][1], p_nb + p)
                else:
                    beam_extend[prefix_extend][1] = sum_log(beam_extend[prefix_extend][1], p_b + p, p_nb + p)

        # Sort and trim the beam before moving on to the
        # next time-step.

        beam = sorted(beam_extend.items(), key=lambda x: sum_log(*x[1]), reverse=True)[:beam_size]

    best = beam[0]

    return best[0], -sum_log(*best[1])


def rna_decode(activaties, beam_size=10, blank=None, prune=None, lm=None, alpha=0.30, beta=5):
    """
    """
    T, S = activaties.shape
    activaties_log = np.log(activaties)
    lm = (lambda l: 1) if lm is None else lm
    if not blank:
        blank = S-1
    # Elements in the beam are (prefix, (p_blank, p_no_blank))
    # Initialize the beam with the empty sequence, a probability of
    # 1 for ending in blank and zero for ending in non-blank (in log space).
    beam = [(tuple(), (LOG_ONE, LOG_ZERO))]

    for t in range(T): # Loop over time
        # A default dictionary to store the next step candidates.
        beam_extend = collections.defaultdict(lambda: [LOG_ZERO, LOG_ZERO])
        prune_log = np.log(prune) if prune else LOG_ZERO
        pruned_vocab = [i for i in np.where(activaties_log[t] > prune_log)[0]]
        # print(len(pruned_vocab))
        for s in pruned_vocab: # Loop over vocab
            p = activaties_log[t, s]
            for prefix, (p_b, p_nb) in beam: # Loop over beam
                if s == blank:
                    # If we propose a blank the prefix doesn't change.
                    # Only the probability of ending in blank gets updated.
                    beam_extend[prefix][0] = sum_log(p_b + p, p_nb + p)
                else:
                    # Extend the prefix by the new character s and add it to the beam.
                    # Only the probability of not ending in blank gets updated.
                    prefix_extend = prefix + (s,)
                    lm_prob_log = np.log(lm(prefix_extend) ** alpha)
                    beam_extend[prefix_extend][1] = sum_log(p_b + p + lm_prob_log, p_nb + p + lm_prob_log)

        # Sort and trim the beam before moving on to the next time-step.
        sorter = lambda l:sum_log(*l[1]) * (len(l) + 1) ** beta
        beam = sorted(beam_extend.items(), key=sorter, reverse=True)[:beam_size]

    if len(beam)>0:
        best = beam[0]
        return best[0], -sum_log(*best[1])
    else:
        return [], None


def activations2seqs_probs(batch_activations, seqs_labels):
    """
    switch activation to seqs probs
    prob: [size_batch, len_seq, len_labels_blanked]

    TODO: not using large dict to store the prob, using a lookup mechanism
    """
    size_batch, len_seq, num_labels = batch_activations.shape
    # seqs_labels = np.expand_dims(seqs_labels, 1)
    index_seq = np.arange(len_seq).reshape(-1, 1)
    seqs_probs = [batch_activations[i, index_seq, seqs_labels[i]] for i in range(size_batch)]

    return np.asarray(seqs_probs)


def batch_ctc_decode(distribution, beam_size=10, blank=0):
    list_res = []
    for activaties in distribution:
        res, _ = ctc_decode(activaties, beam_size=beam_size, blank=blank)
        list_res.append(res)
    return list_res


def testActivations2seqs_probs():
    size_batch = 2
    len_seq = 8
    len_labels = 4
    seqs_labels = [[0, 1, 1, 0], [0, 2, 1, 3]]
    activations = np.arange(size_batch*len_seq*len_labels).reshape(size_batch, len_seq, len_labels)
    probs = activations2seqs_probs(activations, seqs_labels)
    return activations, seqs_labels, probs


def testCost():
    num_class = 5
    num_lables = 4
    # inputs = np.array(
    #     [[[0.633766, 0.221185, 0.0917319],
    #       [0.111121, 0.588392, 0.278779],
    #       [0.0357786, 0.633813, 0.321418],
    #       [0.0663296, 0.643849, 0.280111],
    #       [0.458235, 0.396634, 0.123377],
    #       [0.633766, 0.221185, 0.0917319],
    #       [0.111121, 0.588392, 0.278779],
    #       [0.0357786, 0.633813, 0.321418],
    #       [0.0663296, 0.643849, 0.280111],
    #       [0.458235, 0.396634, 0.123377]],
    #      [[0.30176, 0.28562, 0.0831517],
    #       [0.24082, 0.397533, 0.0557226],
    #       [0.230246, 0.450868, 0.0389607],
    #       [0.280884, 0.429522, 0.0326593],
    #       [0.423286, 0.315517, 0.0338439],
    #       [0.30176, 0.28562, 0.0831517],
    #       [0.24082, 0.397533, 0.0557226],
    #       [0.230246, 0.450868, 0.0389607],
    #       [0, 0, 0],
    #       [0, 0, 0]]],
    #     dtype=np.float32)
    inputs = np.asarray(
        [[[0.8, 0.1, 0.1],
          [0.8, 0.1, 0.1],
          [0.8, 0.1, 0.1],
          # [0.05, 0.9, 0.05],
          [0.8, 0.1, 0.1],
          [0.8, 0.1, 0.1],
          [0.8, 0.1, 0.1],
          [0.8, 0.1, 0.1],
          [0.8, 0.1, 0.1],
          # [0.1, 0.1, 0.8],
          [0.8, 0.1, 0.1],
          [0.8, 0.1, 0.1]],
         [[0.98, 0.01, 0.01],
          # [0.98, 0.01, 0.01],
          [0.98, 0.01, 0.01],
          [0.01, 0.98, 0.01],
          [0.01, 0.98, 0.01],
          # [0.98, 0.01, 0.01],
          [0.98, 0.01, 0.01],
          # [0.98, 0.01, 0.01],
          [0.98, 0.01, 0.01],
          [0.01, 0.01, 0.98],
          [0.98, 0.01, 0.01],
          [0, 0, 0],
          [0, 0, 0]]],
        dtype=np.float32)

    seqs_labels = np.array([[1, 1, 2, 1],
                            [1, 2, 0, 0]])
    seq_label_lens = np.sum(np.not_equal(seqs_labels, 0), -1)
    seq_fea_lens = [10, 8]
    # assert np.allclose(np.sum(inputs, -1), 1)


    print(ctc_loss(inputs, seqs_labels, seq_fea_lens, seq_label_lens))


def test_reduce_map():
    """
    """
    batch_samples = np.array([[3,3,4,4,0,0,3,3,0,0,3,5,5,0],
                              [2,2,0,0,4,0,6,6,0,0,6,6,6,0]],
                             dtype=np.int32)
    ctc_reduce_map(batch_samples)


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
    vocab_list = ['<blank>', 'a', 'b']
    beam_size = 20
    inputs = np.asarray(
        [[[0.8, 0.1, 0.1],
          [0.8, 0.1, 0.1],
          [0.8, 0.1, 0.1],
          # [0.05, 0.9, 0.05],
          [0.8, 0.1, 0.1],
          [0.8, 0.1, 0.1],
          [0.8, 0.1, 0.1],
          [0.8, 0.1, 0.1],
          [0.8, 0.1, 0.1],
          # [0.1, 0.1, 0.8],
          [0.8, 0.1, 0.1],
          [0.8, 0.1, 0.1]],
         [[0.98, 0.01, 0.01],
          # [0.98, 0.01, 0.01],
          [0.98, 0.01, 0.01],
          [0.01, 0.98, 0.01],
          [0.01, 0.98, 0.01],
          # [0.98, 0.01, 0.01],
          [0.98, 0.01, 0.01],
          # [0.98, 0.01, 0.01],
          [0.98, 0.01, 0.01],
          [0.01, 0.01, 0.98],
          [0.98, 0.01, 0.01],
          [0, 0, 0],
          [0, 0, 0]]],
        dtype=np.float32)
    # greedy_result = ["ac'bdc", "b'da"]
    # beam_search_result = ['acdc', "b'a"]

    # list_activations = np.asarray([activation1, activation2], dtype=np.float32)
    # print(list_activations.shape)
    results = [ctc_decode(activ, beam_size=1, blank=0) for activ in inputs]

    for result, score in results:
        result_label = ''.join([vocab_list[idx] for idx in result])
        print(result_label, score)


def testRNADecode():
    import pickle
    from utils.vocab import load_vocab

    _, idx2token = load_vocab('/Users/easton/Projects/eastonCode/examples/decode/vocab_3673+1.txt')
    with open('/Users/easton/Projects/eastonCode/examples/decode/distribution.txt', 'rb') as f,\
        open('/Users/easton/Projects/eastonCode/examples/decode/dev_rna_res.txt', 'w') as fw:
        while True:
            try:
                res, _ = rna_decode(pickle.load(f), beam_size=10, prune=0.0002, alpha=0.30, beta=5)
                # res, _ = ctc_decode(pickle.load(f), beam_size=1)
                res = ' '.join(idx2token[id] for id in res)
                print(res)
                fw.write(res+'\n')
            except EOFError:
                break

if __name__ == "__main__":
    # testCost()
    # testdDecode()
    testRNADecode()
