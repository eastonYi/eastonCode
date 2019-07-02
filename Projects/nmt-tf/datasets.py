import logging
import sys
import time
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')
import numpy as np
from pathlib import Path
from random import shuffle, random
import tensorflow as tf
from tqdm import tqdm

from utils.dataset_info import get_tfdata_info, histogram
from arguments import args

if __name__ == '__main__':
    """
    noitation: the sample fentch from the dataset can be none!
    """
    # list_len = []
    # print(len(args.dataset_train))
    # for i, sample in enumerate(args.dataset_train):
    #     x = sample['feature']
    #     list_len.append(len(x))
    #
    # histogram(list_len, 'data')

    idx_init=9
    list_num = []
    list_length = []
    f_len_hist = 'data/dataset_len_hist.txt'
    with open(f_len_hist) as f:
        for line in f:
            num, length = line.strip().split(':')
            list_num.append(int(num))
            list_length.append(int(length))

    def next_idx(idx, energy):
        for i in range(idx, len(list_num), 1):
            if list_length[i]*sum(list_num[idx+1:i+1]) > energy:
                return i-1
        return

    M = args.num_batch_tokens
    b0 = int(M / list_length[idx_init])
    k = b0/sum(list_num[:idx_init+1])
    energy = M/k

    list_batchsize = [b0]
    list_boundary = [list_length[idx_init]]

    idx = idx_init
    while idx < len(list_num):
        idx = next_idx(idx, energy)
        if not idx:
            break
        if idx == idx_init:
            print('enlarge the idx_init!')
            sys.exit()
        list_boundary.append(list_length[idx])
        list_batchsize.append(int(M / list_length[idx]))

    list_boundary.append(list_length[-1])
    list_batchsize.append(int(M/list_length[-1]))

    print('suggest boundaries: \n{}'.format(','.join(map(str, list_boundary))))
    print('corresponding batch size: \n{}'.format(','.join(map(str, list_batchsize))))
