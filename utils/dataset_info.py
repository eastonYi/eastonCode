#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np


def get_tfdata_info(dir_tfdata, len_dataset, args, idx_init=150, dir_save_info='data'):
    """
    enlarge idx_init can shrink the num of buckets
    """
    print('get the dataset info')
    import tensorflow as tf
    from tqdm import tqdm
    from tfTools.tfRecord import readTFRecord

    feat, label = readTFRecord(dir_tfdata, args, transform=True)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    with tf.train.MonitoredTrainingSession(config=config) as sess:
        list_len = []
        for _ in tqdm(range(len_dataset)):
            feature = sess.run(feat)
            list_len.append(len(feature))

    histogram(list_len, dir_save_info)

    list_num = []
    list_length = []
    f_len_hist = dir_save_info + '/dataset_len_hist.txt'
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

    M = args.num_batch_token
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
        list_boundary.append(list_length[idx])
        list_batchsize.append(int(M / list_length[idx]))

    list_boundary.append(list_length[-1])
    list_batchsize.append(int(M/list_length[-1]))

    print('suggest boundaries: \n{}'.format(','.join(map(str, list_boundary))))
    print('corresponding batch size: \n{}'.format(','.join(map(str, list_batchsize))))


def histogram(list_len, dir_save='.'):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('Smarts')
    ax.set_ylabel('Probability')
    ax.set_title('Histogram dataset lenght')
    n, bins, patches = ax.hist(list_len, facecolor='g', alpha=0.75)
    # sort -n *.txt | uniq -c
    fig.savefig(dir_save + '/' + 'hist_dataset_len')
    # print('info: ', ' '.join(map(str, list_len)))

    hist, edges = np.histogram(list_len, bins=(max(list_len)-min(list_len)+1))

    # save hist
    info_file = dir_save + '/' + 'dataset_len_hist.txt'
    with open(info_file, 'w') as fw:
        for num, edge in zip(hist, edges):
            fw.write('{}: {}\n'.format(str(num), str(int(np.ceil(edge)))))


if __name__ == '__main__':
    dataset = [[1,2,3],
               [1,2,3,4,5]
               [1,2,3,4,5,6,7]]
    histogram(dataset)
