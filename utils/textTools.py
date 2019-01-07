import re
import codecs
import unicodedata
import numpy as np
import editdistance as ed


# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space


def unpadding(list_idx, eos_idx=None, min_idx=0, max_idx=None):
    """
    for the 1d array
    Demo:
        a = np.array([2,2,3,4,5,1,0,0,0])
        unpadding(a, 1)
        # array([2, 2, 3, 4, 5])
    """
    if eos_idx is not None:
        end_idx = np.where(list_idx==eos_idx)[0]
        end_idx = end_idx[0] if len(end_idx)>0 else None
        list_idx = list_idx[:end_idx]

    list_idx = list_idx[np.where(list_idx>min_idx)]
    if max_idx is not None:
        list_idx = list_idx[np.where(list_idx<max_idx)]

    return list_idx


def batch_cer(result, reference, eos_idx=None, min_idx=0, max_idx=None):
    """
    result and reference are lists of tokens
    eos_idx is the padding token or eos token
    """
    batch_dist = 0
    batch_len = 0
    for res, ref in zip(result, reference):
        res = unpadding(res, eos_idx, min_idx, max_idx)
        ref = unpadding(ref, eos_idx, min_idx, max_idx)
        batch_dist += ed.eval(res, ref)
        batch_len += len(ref)

    return batch_dist, batch_len


def batch_wer(result, reference, idx2token, unit, eos_idx=None, min_idx=0, max_idx=None):
    """
    Args:
        result and reference are lists of tokens idx
        eos_idx is the padding token or eos token idx
        idx2token is a dict form idx to token
        seperator is what to join the tokens. If token is char, seperator is '';
            if token is word, seperator is ' '.
        eos_idx is the padding token idx or the eos token idx
    """
    batch_dist = 0
    batch_len = 0
    for res, ref in zip(result, reference):
        list_res_txt = array2text(res, unit, idx2token, eos_idx, min_idx, max_idx).split()
        # print(list_res_txt)
        list_ref_txt = array2text(ref, unit, idx2token, eos_idx, min_idx, max_idx).split()
        batch_dist += ed.eval(list_res_txt, list_ref_txt)
        batch_len += len(list_ref_txt)

    return batch_dist, batch_len


def array2text(res, unit, idx2token, eos_idx=None, min_idx=0, max_idx=None):
    """
    char: the english characters including blank. The Chinese characters belongs to the word
    for the 1d array
    """
    res = unpadding(res, eos_idx, min_idx, max_idx)
    if unit == 'char':
        list_res_txt = array_idx2char(res, idx2token, seperator='')
    elif unit == 'word':
        list_res_txt = array_idx2char(res, idx2token, seperator=' ')
    elif unit == 'subword':
        list_res_txt = array_idx2char(res, idx2token, seperator=' ').replace('@@ ', '')
    else:
        raise NotImplementedError('not know unit!')

    return list_res_txt


def normalize_text(original, remove_apostrophe=True):
    """
    Given a Python string ``original``, remove unsupported characters.
    The only supported characters are letters and apostrophes"'".
    """
    # convert any unicode characters to ASCII equivalent
    # then ignore anything else and decode to a string
    result = unicodedata.normalize("NFKD", original).encode("ascii", "ignore").decode()
    if remove_apostrophe:
        # remove apostrophes to keep contractions together
        result = result.replace("'", "")
    # return lowercase alphabetic characters and apostrophes (if still present)
    return re.sub("[^a-zA-Z']+", ' ', result).strip().lower()


def normalize_txt_file(txt_file, remove_apostrophe=True):
    """
    Given a path to a text file, return contents with unsupported characters removed.
    """
    with codecs.open(txt_file, encoding="utf-8") as f:
        return normalize_text(f.read(), remove_apostrophe)


def array_idx2char(array_idx, idx2token, seperator=''):
    # array_idx = np.asarray(array_idx, dtype=np.int32)
    if len(array_idx)==0 or np.isscalar(array_idx[0]):
        return seperator.join(idx2token[i] for i in array_idx)
    else:
        return [array_idx2char(i, idx2token, seperator=seperator) for i in array_idx]


def array_char2idx(list_idx, token2idx, seperator=''):
    """
    list of chars to the idx array and length
    """
    from utils.tools import padding_list_seqs
    sents = []
    if seperator:
        for sent in list_idx:
            sents.append([token2idx[token] for token in sent.split(seperator)])
    else:
        for sent in list_idx:
            sents.append([token2idx[token] for token in list(sent)])
    padded, len_seqs = padding_list_seqs(sents, dtype=np.int32)

    return padded, len_seqs


def text_to_char_array(original):
    """
    Given a Python string ``original``, map characters
    to integers and return a np array representing the processed string.
    """
    # Create list of sentence's words w/spaces replaced by ''
    result = original.replace(' ', '  ')
    result = result.split(' ')

    # Tokenize words into letters adding in SPACE_TOKEN where required
    result = np.hstack([SPACE_TOKEN if xt == '' else list(xt) for xt in result])

    # Return characters mapped into indicies
    return np.asarray([SPACE_INDEX if xt == SPACE_TOKEN else ord(xt) - FIRST_INDEX for xt in result])


def sparse_tuple_from(sequences, dtype=np.int32):
    """
    a =[0,1]
    a[0] =text_to_char_array(" look at ___")
    a[1] =text_to_char_array("a son shane __")
    (array([[ 0,  0],
            [ 0,  1],
            [ 0,  2],
            [ 0,  3],
            [ 0,  4],
            [ 0,  5],
            [ 0,  6],
            [ 0,  7],
            [ 0,  8],
            [ 0,  9],
            [ 0, 10],
            [ 0, 11],
            [ 0, 12],
            [ 1,  0],
            [ 1,  1],
            [ 1,  2],
            [ 1,  3],
            [ 1,  4],
            [ 1,  5],
            [ 1,  6],
            [ 1,  7],
            [ 1,  8],
            [ 1,  9],
            [ 1, 10],
            [ 1, 11],
            [ 1, 12],
            [ 1, 13]]),
     array([ 0,  0, 12, 15, 15, 11,  0,  1, 20,  0, -1, -1, -1,  1,  0, 19, 15,
            14,  0, 19,  8,  1, 14,  5,  0, -1, -1], dtype=int32),
     array([ 2, 14]))
    """

    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype=np.int64)

    # return tf.SparseTensor(indices=indices, values=values, shape=shape)
    return indices, values, shape
