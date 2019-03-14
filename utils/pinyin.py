from pypinyin import lazy_pinyin, Style
import numpy as np
import logging
import sys
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


def homophone_table(list_vocab):
    len_vocab = len(list_vocab)
    list_pinyin = [lazy_pinyin(token, style=Style.NORMAL) for token in list_vocab]
    table_value = np.ones([len_vocab, len_vocab], dtype=np.float32) * -1e3

    for i in range(len_vocab):
        for j in range(len_vocab):
            if list_pinyin[i] == list_pinyin[j]:
                table_value[i][j] = 1.0
        table_value[i][i] = 2.0
    logging.info('loading homophone_table success!')

    return table_value


def main():
    from utils.vocab import load_vocab
    from utils.math_numpy import softmax

    token2idx, idx2token = load_vocab('/Users/easton/Documents/vocab_3673+1.txt')
    vocab = token2idx.keys()
    table_value = homophone_table(vocab)
    # print(table_value, table_value.shape)
    target_distributions = softmax(table_value)

    m = target_distributions[10]
    print(m[m>0])
    # import pdb; pdb.set_trace()
    print('asd')

if __name__ == '__main__':
    main()
