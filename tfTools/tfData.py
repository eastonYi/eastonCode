#!/usr/bin/env
# coding=utf-8

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    import os

import tensorflow as tf
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path
from random import shuffle

from .tfTools import sequence_mask
from .tfAudioTools import splice, down_sample, add_delt


def save2tfrecord(dataset, dir_save, size_file=5000000):
    """
    Args:
        dataset = ASRdataSet(data_file, args)
        dir_save: the dir to save the tfdata files
    Return:
        Nothing but a folder consist of `tfdata.info`, `*.recode`

    Notice: the feats.scp file is better to cluster by ark file and sort by the index in the ark files
    For example, '...split16/1/final_feats.ark:143468' the paths share the same arkfile '1/final_feats.ark' need to close with each other,
    Meanwhile, these files need to be sorted by the index ':143468'
    ther sorted scp file will be 10x faster than the unsorted one.
    """

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    num_token = 0
    idx_file = -1
    num_damaged_sample = 0

    assert dataset.transform == False
    for i, sample in enumerate(tqdm(dataset)):
        if not sample:
            num_damaged_sample += 1
            continue
        dim_feature = sample['feature'].shape[-1]
        if (num_token // size_file) > idx_file:
            idx_file = num_token // size_file
            print('saving to file {}/{}.recode'.format(dir_save, idx_file))
            writer = tf.io.TFRecordWriter(str(dir_save/'{}.recode'.format(idx_file)))

        example = tf.train.Example(
            features=tf.train.Features(
                feature={'feature': _bytes_feature(sample['feature'].tostring()),
                         'label': _bytes_feature(sample['label'].tostring())}
            )
        )
        writer.write(example.SerializeToString())
        num_token += len(sample['feature'])

    with open(dir_save/'tfdata.info', 'w') as fw:
        print('data_file {}'.format(dataset.list_files), file=fw)
        print('dim_feature {}'.format(dim_feature), file=fw)
        print('num_tokens {}'.format(num_token), file=fw)
        print('size_dataset {}'.format(i-num_damaged_sample), file=fw)
        print('damaged samples: {}'.format(num_damaged_sample), file=fw)

    return

def save2tfrecord_multilabel(dataset, dir_save, size_file=5000000):
    """
    use for multi-label
    """

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    num_token = 0
    idx_file = -1
    num_damaged_sample = 0

    assert dataset.transform == False
    for i, sample in enumerate(tqdm(dataset)):
        if not sample:
            num_damaged_sample += 1
            continue
        dim_feature = sample['feature'].shape[-1]
        if (num_token // size_file) > idx_file:
            idx_file = num_token // size_file
            print('saving to file {}/{}.recode'.format(dir_save, idx_file))
            writer = tf.io.TFRecordWriter(str(dir_save/'{}.recode'.format(idx_file)))

        example = tf.train.Example(
            features=tf.train.Features(
                feature={'feature': _bytes_feature(sample['feature'].tostring()),
                         'label': _bytes_feature(sample['label'].tostring()),
                         'phone': _bytes_feature(sample['phone'].tostring())}
            )
        )
        writer.write(example.SerializeToString())
        num_token += len(sample['feature'])

    with open(dir_save/'tfdata.info', 'w') as fw:
        print('data_file {}'.format(dataset.list_files), file=fw)
        print('dim_feature {}'.format(dim_feature), file=fw)
        print('num_tokens {}'.format(num_token), file=fw)
        print('size_dataset {}'.format(i-num_damaged_sample), file=fw)
        print('damaged samples: {}'.format(num_damaged_sample), file=fw)

    return


def readTFRecord(dir_data, args, _shuffle=False, transform=False):
    """
    the tensor could run unlimitatly
    """
    list_filenames = fentch_filelist(dir_data)
    if _shuffle:
        shuffle(list_filenames)
    else:
        list_filenames.sort()

    filename_queue = tf.train.string_input_producer(
        list_filenames, num_epochs=None, shuffle=shuffle)

    reader_tfRecord = tf.TFRecordReader()
    _, serialized_example = reader_tfRecord.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={'feature': tf.FixedLenFeature([], tf.string),
                  # 'id': tf.FixedLenFeature([], tf.string)}
                  'label': tf.FixedLenFeature([], tf.string)}
    )

    feature = tf.reshape(tf.decode_raw(features['feature'], tf.float32),
                         [-1, args.data.dim_feature])[:3000, :]
    # id = tf.decode_raw(features['id'], tf.string)
    label = tf.decode_raw(features['label'], tf.int32)

    if transform:
        feature = process_raw_feature(feature, args)

    return feature, label

def readTFRecord_multilabel(dir_data, args, _shuffle=False, transform=False):
    """
    use for multi-label
    """
    list_filenames = fentch_filelist(dir_data)
    if _shuffle:
        shuffle(list_filenames)
    else:
        list_filenames.sort()

    filename_queue = tf.train.string_input_producer(
        list_filenames, num_epochs=None, shuffle=shuffle)

    reader_tfRecord = tf.TFRecordReader()
    _, serialized_example = reader_tfRecord.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={'feature': tf.FixedLenFeature([], tf.string),
                  # 'id': tf.FixedLenFeature([], tf.string)}
                  'label': tf.FixedLenFeature([], tf.string),
                  'phone': tf.FixedLenFeature([], tf.string)}
    )

    feature = tf.reshape(tf.decode_raw(features['feature'], tf.float32),
        [-1, args.data.dim_feature])[:2000, :]
    # id = tf.decode_raw(features['id'], tf.string)
    label = tf.decode_raw(features['label'], tf.int32)
    phone = tf.decode_raw(features['phone'], tf.int32)

    if transform:
        feature = process_raw_feature(feature, args)

    return feature, label, phone


def process_raw_feature(seq_raw_features, args):
    # 1-D, 2-D
    if args.data.add_delta:
        seq_raw_features = add_delt(seq_raw_features)

    # Splice
    fea = splice(
        seq_raw_features,
        left_num=0,
        right_num=args.data.num_context)

    # downsample
    fea = down_sample(
        fea,
        rate=args.data.downsample,
        axis=0)
    fea.set_shape([None, args.data.dim_input])

    return fea


def fentch_filelist(dir_data):
    p = Path(dir_data)
    assert p.is_dir()

    return [str(i) for i in p.glob('*.recode')]


class TFReader:
    def __init__(self, dir_tfdata, args, is_train=True):
        self.is_train = is_train
        self.args = args
        self.sess = None
        self.list_batch_size = self.args.list_batch_size
        self.list_bucket_boundaries = self.args.list_bucket_boundaries
        if args.phone:
            self.feat, self.label, self.phone = readTFRecord_multilabel(
                dir_tfdata,
                args,
                _shuffle=is_train,
                transform=True)
        else:
            self.feat, self.label = readTFRecord(
                dir_tfdata,
                args,
                _shuffle=is_train,
                transform=True)

    def __iter__(self):
        """It is only a demo! Using `fentch_batch_with_TFbuckets` in practice."""
        if not self.sess:
            raise NotImplementedError('please assign sess to the TFReader! ')

        for i in range(len(self.args.data.size_dev)):
            yield self.sess.run([self.feat, self.label])

    def fentch_batch(self):
        list_inputs = [self.feat, self.label, tf.shape(self.feat)[0], tf.shape(self.label)[0]]
        list_outputs = tf.train.batch(
            tensors=list_inputs,
            batch_size=16,
            num_threads=8,
            capacity=500,
            dynamic_pad=True,
            allow_smaller_final_batch=True
        )
        seq_len_feats = tf.reshape(list_outputs[2], [-1])
        seq_len_label = tf.reshape(list_outputs[3], [-1])

        return list_outputs[0], list_outputs[1], seq_len_feats, seq_len_label


    def fentch_batch_bucket(self):
        """
        the input tensor length is not equal,
        so will add the len as a input tensor
        list_inputs: [tensor1, tensor2]
        added_list_inputs: [tensor1, tensor2, len_tensor1, len_tensor2]
        """
        list_inputs = [self.feat, self.label, tf.shape(self.feat)[0], tf.shape(self.label)[0]]
        _, list_outputs = tf.contrib.training.bucket_by_sequence_length(
            input_length=list_inputs[2],
            tensors=list_inputs,
            batch_size=self.list_batch_size,
            bucket_boundaries=self.list_bucket_boundaries,
            num_threads=8,
            bucket_capacities=[i*2 for i in self.list_batch_size],
            capacity=30,
            dynamic_pad=True,
            allow_smaller_final_batch=True)
        seq_len_feats = tf.reshape(list_outputs[2], [-1])
        seq_len_label = tf.reshape(list_outputs[3], [-1])

        return list_outputs[0], list_outputs[1], seq_len_feats, seq_len_label

    def fentch_multi_label_batch_bucket(self):
        """
        the input tensor length is not equal,
        so will add the len as a input tensor
        list_inputs: [tensor1, tensor2]
        added_list_inputs: [tensor1, tensor2, len_tensor1, len_tensor2]
        """
        list_inputs = [self.feat, self.label, self.phone,
                       tf.shape(self.feat)[0], tf.shape(self.label)[0], tf.shape(self.phone)[0]]
        _, list_outputs = tf.contrib.training.bucket_by_sequence_length(
            input_length=tf.shape(self.feat)[0],
            tensors=list_inputs,
            batch_size=self.args.list_batch_size,
            bucket_boundaries=self.args.list_bucket_boundaries,
            num_threads=8,
            bucket_capacities=[i*3 for i in self.args.list_batch_size],
            capacity=2000,
            dynamic_pad=True,
            allow_smaller_final_batch=True)
        seq_len_feats = tf.reshape(list_outputs[3], [-1])
        seq_len_label = tf.reshape(list_outputs[4], [-1])
        seq_len_phone = tf.reshape(list_outputs[5], [-1])

        return list_outputs[0], list_outputs[1], list_outputs[2],\
                seq_len_feats, seq_len_label, seq_len_phone


class TFData:
    """
    test on TF2.0-alpha
    """
    def __init__(self, dataset, dataAttr, dir_save, args, size_file=5000000, max_feat_len=3000):
        self.dataset = dataset
        self.dataAttr =  dataAttr # ['feature', 'label', 'align']
        self.max_feat_len = max_feat_len
        self.dir_save = dir_save
        self.args = args
        self.size_file = size_file
        self.dim_feature = dataset[0]['feature'].shape[-1] \
            if dataset else self.read_tfdata_info(dir_save)['dim_feature']

    def save(self, name):
        num_token = 0
        num_damaged_sample = 0

        def serialize_example(feature, label, align):
            atts = {
                'feature': self._bytes_feature(feature.tostring()),
                'label': self._bytes_feature(label.tostring()),
                'align': self._bytes_feature(align.tostring()),
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=atts))

            return example_proto.SerializeToString()

        def generator():
            for features, _ in zip(self.dataset, tqdm(range(len(self.dataset)))):
                # print(features['feature'].shape)
                yield serialize_example(features['feature'], features['label'], features['align'])

        dataset_tf = tf.data.Dataset.from_generator(
            generator=generator,
            output_types=tf.string,
            output_shapes=())

        writer = tf.data.experimental.TFRecordWriter(str(self.dir_save/'{}.recode'.format(name)))
        writer.write(dataset_tf)

        with open(str(self.dir_save/'tfdata.info'), 'w') as fw:
            fw.write('data_file {}\n'.format(self.dataset.file))
            fw.write('dim_feature {}\n'.format(self.dim_feature))
            fw.write('num_tokens {}\n'.format(num_token))
            fw.write('size_dataset {}\n'.format(len(self.dataset)-num_damaged_sample))
            fw.write('damaged samples: {}\n'.format(num_damaged_sample))

        return

    def read(self, _shuffle=False):
        """
        the tensor could run unlimitatly
        """
        list_filenames = self.fentch_filelist(self.dir_save)
        if _shuffle:
            shuffle(list_filenames)
        else:
            list_filenames.sort()

        raw_dataset = tf.data.TFRecordDataset(list_filenames)

        def _parse_function(example_proto):
            features = tf.io.parse_single_example(
                example_proto,
                # features={attr: tf.io.FixedLenFeature([], tf.string) for attr in self.dataAttr}
                features={
                    'feature': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([], tf.string),
                    'align': tf.io.FixedLenFeature([], tf.string)
                }
            )
            feature = tf.reshape(tf.io.decode_raw(features['feature'], tf.float32),
                                 [-1, self.dim_feature])[:self.max_feat_len, :]
            label = tf.io.decode_raw(features['label'], tf.int32)
            align = tf.io.decode_raw(features['align'], tf.int32)

            return feature, label, align

        features = raw_dataset.map(_parse_function)

        return features

    def get_bucket_size(self, idx_init, reuse=False):

        f_len_hist = './dataset_len_hist.txt'
        list_len = []

        if not reuse:
            dataset = self.read(_shuffle=False)
            for sample in tqdm(dataset):
                feature, *_ = sample
                list_len.append(len(feature))

            hist, edges = np.histogram(list_len, bins=(max(list_len)-min(list_len)+1))

            # save hist
            with open(f_len_hist, 'w') as fw:
                for num, edge in zip(hist, edges):
                    fw.write('{}: {}\n'.format(str(num), str(int(np.ceil(edge)))))
                    print(str(num), str(int(np.ceil(edge))))

            list_num = []
            list_length = []
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

        M = self.args.num_batch_tokens
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

    def __len__(self):
        return self.read_tfdata_info(self.dir_save)['size_dataset']

    @staticmethod
    def fentch_filelist(dir_data):
        p = Path(dir_data)
        assert p.is_dir()

        return [str(i) for i in p.glob('*.recode')]

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a list of string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def read_tfdata_info(dir_save):
        data_info = {}
        with open(dir_save/'tfdata.info') as f:
            for line in f:
                if 'dim_feature' in line or \
                    'num_tokens' in line or \
                    'size_dataset' in line:
                    line = line.strip().split(' ')
                    data_info[line[0]] = int(line[1])

        return data_info


if __name__ == '__main__':
    # from configs.arguments import args
    from tqdm import tqdm
    import sys

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')
    # test_bucket_boundaries(args=args)
    # test_tfdata_bucket(args=args, num_threads=args.num_parallel)
    # test_queue()
