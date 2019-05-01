import logging
import sys
import time
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')
import numpy as np
from pathlib import Path
from random import shuffle, random
import tensorflow as tf
from tqdm import tqdm

from dataProcessing.audio import audio2vector, process_raw_feature
from dataProcessing.dataHelper import DataSet, ASRDataLoader
from tfTools.tfRecord import save2tfrecord, readTFRecord, TFReader, save2tfrecord_multilabel
from utils.textTools import array_idx2char, unpadding, array2text
from utils.dataset_info import get_tfdata_info
from arguments import args


def showing_scp_data():
    from dataProcessing.dataHelper import ASR_scp_DataSet
    dataset_train = ASR_scp_DataSet(
        f_scp=args.dirs.train.data,
        f_trans=args.dirs.train.label,
        args=args,
        _shuffle=False,
        transform=False)
    ref_txt = array2text(dataset_train[0]['label'], args.data.unit, args.idx2token)
    print(ref_txt)


def showing_csv_data():
    from dataProcessing.dataHelper import ASR_csv_DataSet
    dataset_train = ASR_csv_DataSet(
        list_files=[args.dirs.train.data],
        args=args,
        _shuffle=False,
        transform=False)
    ref_txt = array2text(dataset_train[0]['label'], args.data.unit, args.idx2token)
    print(ref_txt)


if __name__ == '__main__':
    """
    noitation: the sample fentch from the dataset can be none!
    the bucket size is related to the **transformed** feature length
    each time you change dataset or frame skipping strategy, you need to rerun this script
    """

    confirm = input("You are going to generate new tfdata, may covering the existing one.\n press ENTER to continue. ")
    if confirm == "":
        print('will generate tfdata in 5 secs!')
        time.sleep(5)
    save2tfrecord(args.dataset_dev, args.dirs.dev.tfdata)
    save2tfrecord(args.dataset_train, args.dirs.train.tfdata)

    len_dataset = args.data.train.size_dataset or len(args.dataset_train)
    get_tfdata_info(args.dirs.train.tfdata, len_dataset, args, idx_init=290, dir_save_info='data', rerun=False)
