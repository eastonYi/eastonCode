import numpy as np
import os
from random import random


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    demo:
    a ={'length': 10, 'shape': (2,3)}
    config = AttrDict(a)
    config.length #10

    for i in read_tfdata_info(args.dirs.train.tfdata).items():
        args.data.train.i[0] = i[1]
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]


class check_to_stop(object):
    def __init__(self):
        self.value_1 = 999
        self.value_2 = 999
        self.value_3 = 999

    def __call__(self, new_value):
        import sys

        self.value_1 = self.value_2
        self.value_2 = self.value_3
        self.value_3 = new_value

        if self.value_1 < self.value_2 < self.value_3 and new_value > self.value_2:
            print('force exit!')
            sys.exit()


def padding_list_seqs(sequences, maxlen=None, dtype=np.float32, value=0., masking=True):
    '''
    Pads each sequence to the same length of the longest sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens off either the beginning or the end
        (default) of the sequence. Supports post-padding (default) and
        pre-padding.

        Args:
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            value: float, value to pad the sequences to the desired value.
            masking: return mask that has the same shape as x; or return length

        Returns:
            numpy.ndarray: Padded sequences shape = (number_of_sequences, maxlen)
            list: original sequence lengths
    '''
    list_lengths = [len(s) for s in sequences]

    size_batch = len(sequences)
    if maxlen is None:
        maxlen = max(list_lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    shape_feature = tuple()
    for s in sequences:
        if len(s) > 0:
            shape_feature = np.asarray(s).shape[1:]
            break

    # a tensor filled with padding value
    x = (np.ones((size_batch, maxlen) + shape_feature) * value).astype(dtype)
    if masking:
        mask = np.zeros(x.shape)
    else:
        sen_len = []
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        trunc = s[:maxlen]

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != shape_feature:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, shape_feature))

        x[idx, :len(trunc)] = trunc
        if masking:
            mask[idx, :len(trunc)] = 1
        else:
            sen_len.append(len(trunc))

    return x, mask if masking else np.asarray(sen_len, dtype=np.uint8)


def pad_to_split(batch, num_split):
    num_pad = num_split - len(batch) % num_split
    if num_pad != 0:
        if batch.ndim > 1:
            pad = np.tile(np.expand_dims(batch[0,:], 0), [num_pad]+[1]*(batch.ndim-1))
        elif batch.ndim == 1:
            pad = np.asarray([batch[0]] * num_pad, dtype=batch[0].dtype)
        batch = np.concatenate([batch, pad], 0)

    return batch


def size_bucket_to_put(l, buckets):
    for i, l1 in enumerate(buckets):
        if l < l1: return i, l1
    # logging.info("The sequence is too long: {}".format(l))
    return -1, 9999


def iter_filename(dataset_dir, suffix='*', sort=None):
    if not os.path.exists(dataset_dir):
        raise IOError("'%s' does not exist" % dataset_dir)
        exit()

    import glob
    iter_filename = glob.iglob(os.path.join(dataset_dir, suffix))

    if sort:
        SORTS = ['filesize_low_high', 'filesize_high_low', 'alpha', 'random']
        if sort not in SORTS:
            raise ValueError('sort must be one of [%s]', SORTS)
        reverse = False
        key = None
        if 'filesize' in sort:
            key = os.path.getsize
        if sort == 'filesize_high_low':
            reverse = True
        elif sort == 'random':
            key = lambda *args: random()

        iter_filename = iter(sorted(list(iter_filename), key=key, reverse=reverse))

    return iter_filename


class Sentence_iter(object):
    '''
    文件夹中文本文件遍历
    sentence_iter = MySentences('/some/directory')
    '''
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.strip().split()


def sparse_tuple_from(sequences):
    """
    Create a sparse representention of ``sequences``.

    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int32)
    shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype=np.int64)

    # return tf.SparseTensor(indices=indices, values=values, shape=shape)
    return indices, values, shape


# def sparse_tuple_to_texts(tuple):
#     indices = tuple[0]
#     values = tuple[1]
#     results = [''] * tuple[2][0]
#     for i in range(len(indices)):
#         index = indices[i][0]
#         c = values[i]
#         c = ' ' if c == SPACE_INDEX else chr(c + FIRST_INDEX)
#         results[index] = results[index] + c
#     # List of strings
#     return results

if __name__ == '__main__':
    checker = check_to_stop()
    for i in [5,4,3,3,2,1,1,2,2]:
        checker(i)
