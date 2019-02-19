import tensorflow as tf
import numpy as np


def pad_to_same(list_array, max_len, size_pad):
    pad = np.zeros((max_len-len(list_array), size_pad), np.float32)
    if list_array:
        res = np.concatenate([np.array(list_array), pad], 0)
    else:
        res = pad

    return res


def add_avg_hidden(list_hidden, list_one_no_blank):
    if list_one_no_blank:
        h = np.mean(np.array(list_one_no_blank), 0)
        del list_one_no_blank[:]
        list_hidden.append(h)


def acoustic_hidden_shrink(distribution_acoustic, hidden, len_acoustic, blank_id, num_post=0):
    """
    distribution_acoustic: [b x t x v]
    hidden: [b x t x h]
    num_post: the number of posterior hidden frame that add to the current one
    """
    alignments = np.argmax(distribution_acoustic, -1) # alignments: [b x t]
    list_batch = []
    list_len = []
    # batch loop
    for i, _hidden in enumerate(hidden):
        list_hidden = []
        token_pre = None
        list_one_no_blank = []
        # time loop
        for t, h in zip(range(len_acoustic[i]), _hidden):
            if alignments[i][t] == token_pre:
                # a repeated no blank
                list_one_no_blank.append(h)

            elif alignments[i][t] == blank_id:
                # a blank
                token_pre = None
                if num_post and list_one_no_blank:
                    list_one_no_blank.extend([*_hidden[t+1: min(t+num_post+1, len_acoustic[i]-1)]])
                add_avg_hidden(list_hidden, list_one_no_blank)
            else:
                # a new no blank
                add_avg_hidden(list_hidden, list_one_no_blank)
                list_one_no_blank = [h]
                token_pre = alignments[i][t]

        add_avg_hidden(list_hidden, list_one_no_blank)

        list_batch.append(list_hidden)
        list_len.append(len(list_hidden))

    list_hidden_padded = []
    for hidden_shrinked in list_batch:
        # the hidden is at least with length of 1
        hidden_padded = pad_to_same(hidden_shrinked, max(list_len+[1]), hidden.shape[-1])
        list_hidden_padded.append(hidden_padded)
    acoustic_shrinked = np.stack(list_hidden_padded, 0)
    # the hidden is at least with length of 1
    list_len = [x if x != 0 else 1 for x in list_len]

    return acoustic_shrinked, np.array(list_len, np.int32)


def acoustic_hidden_shrink_tf(distribution_acoustic, hidden, len_acoustic, blank_id, num_post):
    hidden_shrinked, len_no_blank = tf.py_func(acoustic_hidden_shrink, [distribution_acoustic, hidden, len_acoustic, blank_id, num_post],
                      (tf.float32, tf.int32))
    hidden_shrinked.set_shape([None, None, hidden.get_shape()[-1]])
    len_no_blank.set_shape([None])

    return hidden_shrinked, len_no_blank


def test_acoustic_hidden_shrink():
    distribution_acoustic = np.array(
        [[[0.04,0.01, 0.05, 0.9],
          [0.3, 0.8, 0.05, 0.05],
          [0.1, 0.8, 0.05, 0.05],
          [0.05,0.01, 0.9, 0.04]],
         [[0.9, 0.01, 0.05, 0.04],
          [0.1, 0.05, 0.05, 0.9],
          [0.1, 0.05, 0.05, 0.9],
          [0.05,0.01, 0.9, 0.04]]])
    len_acoustic = [2, 4]
    res = acoustic_hidden_shrink(distribution_acoustic, distribution_acoustic, len_acoustic, 3)
    print(res)


if __name__ == '__main__':
    test_acoustic_hidden_shrink()
