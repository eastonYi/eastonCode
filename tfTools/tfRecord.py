#!/usr/bin/env
# coding=utf-8

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    import os

import tensorflow as tf
import logging
from tqdm import tqdm
from pathlib import Path

from tfTools import tfAudioTools as tfAudio
from tfTools.tfTools import sequence_mask
from tfTools.tfAudioTools import splice, down_sample


def save2tfrecord(dataset, dir_save, size_file=5000000):
    """
    Args:
        dataset = ASRdataSet(data_file, args)
        dir_save: the dir to save the tfdata files
    Return:
        Nothing but a folder consist of `tfdata.info`, `*.recode`
    """

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    num_token = 0
    idx_file = -1
    num_damaged_sample = 0

    for i, sample in enumerate(tqdm(dataset)):
        if not sample:
            num_damaged_sample += 1
            continue
        dim_feature = sample['feature'].shape[-1]
        if (num_token // size_file) > idx_file:
            idx_file = num_token // size_file
            print('saving to file {}/{}.recode'.format(dir_save, idx_file))
            writer = tf.python_io.TFRecordWriter(str(dir_save/'{}.recode'.format(idx_file)))

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
        print('size_dataset {}'.format(i-num_damaged_sample+1), file=fw)
        print('damaged samples: {}'.format(num_damaged_sample), file=fw)

    return


def readTFRecord(dir_data, args, shuffle=False, transform=False):
    """
    the tensor could run unlimitatly
    """
    list_filenames = fentch_filelist(dir_data)
    if not shuffle:
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
        [-1, args.data.dim_feature])
    # id = tf.decode_raw(features['id'], tf.string)
    label = tf.decode_raw(features['label'], tf.int32)

    if transform:
        feature = process_raw_feature(feature, args)

    return feature, label


def process_raw_feature(seq_raw_features, args):
    # 1-D, 2-D
    if args.data.add_delta:
        seq_raw_features = tfAudio.add_delt(seq_raw_features)

    # Splice
    fea = tfAudio.splice(
        seq_raw_features,
        left_num=0,
        right_num=args.data.num_context)

    # downsample
    fea = tfAudio.down_sample(
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

        self.feat, self.label = readTFRecord(
            dir_tfdata,
            args,
            shuffle=is_train,
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
            capacity=2000,
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
            batch_size=self.args.list_batch_size,
            bucket_boundaries=self.args.list_bucket_boundaries,
            num_threads=8,
            bucket_capacities=[i*3 for i in self.args.list_batch_size],
            capacity=2000,
            dynamic_pad=True,
            allow_smaller_final_batch=True)
        seq_len_feats = tf.reshape(list_outputs[2], [-1])
        seq_len_label = tf.reshape(list_outputs[3], [-1])

        return list_outputs[0], list_outputs[1], seq_len_feats, seq_len_label

# class TFReader:
#     def __init__(self, dir_tfdata, args):
#         self.filename_config = os.path.join(dir_tfdata, 'fea.cfg')
#         self.filename_tflist = os.path.join(dir_tfdata, 'tf.lst')
#         assert self.check_dir()
#         self.list_tfdata_filenames = self.fentch_filelist(self.filename_tflist)
#         self.dict_raw_feature_properties = self.fentch_dict_raw_feature_properties(self.filename_config)
#         self.args = args
#         self.sess = None
#
#     def __iter__(self):
#         """It is only a demo! Using `fentch_batch_with_TFbuckets` in practice."""
#         if not self.sess:
#             raise NotImplementedError('please assign sess to the TFReader! ')
#
#         tfqueue_filename = tf.train.string_input_producer(self.list_tfdata_filenames,
#                                                           num_epochs=None,
#                                                           shuffle=True)
#
#         tensor_seq_features, tensor_seq_labels, reader_tfRecord = filename_queue2iter_tensor(tfqueue_filename, self.args)
#
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
#
#         for i in range(10000):
#             yield self.sess.run([tensor_seq_features, tensor_seq_labels])
#
#         coord.request_stop()
#         coord.join(threads)
#
#     def check_dir(self):
#         return os.path.exists(self.filename_config) and os.path.exists(self.filename_tflist)
#
#     @staticmethod
#     def fentch_filelist(path_scpfile):
#         with open(path_scpfile) as f:
#             return [line.strip() for line in f]
#
#     @staticmethod
#     def fentch_dict_raw_feature_properties(filename):
#         dict_feature_properties = {}
#         with open(filename) as f:
#             for line in f:
#                 name = line.strip().split()[0]
#                 properties = line.strip().split()[1:]
#                 if name in ('Fea_Dim', 'State_Num', 'Utt_Num', 'Fea_Num_Total'):
#                     dict_feature_properties[name] = int(properties[0])
#                 elif name in ('Fea_Mean', 'Fea_Var'):
#                     dict_feature_properties[name] = [float(i) for i in properties]
#                 else:
#                     raise FileNotFoundError
#
#         return dict_feature_properties
#
#     def fentch_batch_with_map(self, args, is_train=True):
#         def _parse_function(example_proto):
#             nonlocal args
#             features = {'feat': tf.FixedLenFeature([], tf.string),
#                         'label': tf.FixedLenFeature([], tf.string)}
#             raw_example = tf.parse_single_example(example_proto, features)
#
#             seq_raw_features = tf.reshape(tf.decode_raw(raw_example['feat'], tf.float32),
#                                  [-1, self.dict_raw_feature_properties['Fea_Dim']])
#             seq_labels = tf.decode_raw(raw_example['label'], tf.int32)
#
#             seq_features = process_raw_feature(seq_raw_features, args)
#
#             return seq_features, seq_labels
#
#         dataset = tf.data.TFRecordDataset(self.list_tfdata_filenames)
#         dataset = dataset.map(_parse_function, args.num_parallel)
#         dataset = dataset.repeat()
#         iterator = dataset.make_initializable_iterator()
#         seq_features, seq_labels = iterator.get_next()
#
#         return (seq_features, seq_labels), iterator.initializer
#
#     @staticmethod
#     def fentch_batch_with_TFbuckets(list_inputs, args, is_train):
#         logging.info("list_batch_size: {}, \t{}".format(len(args.list_batch_size), args.list_batch_size))
#         logging.info("list_bucket_boundaries: {}, \t{}".format(len(args.list_bucket_boundaries), args.list_bucket_boundaries))
#         seq_len, list_outputs, *_ = bucket_by_sequence_length(
#             input_length=tf.shape(list_inputs[1]),
#             tensors=list_inputs,
#             batch_size=args.list_batch_size,
#             bucket_boundaries=args.list_bucket_boundaries,
#             num_threads=10,
#             bucket_capacities=[i*3 for i in args.list_batch_size],
#             capacity=1000,
#             dynamic_pad=True,
#             allow_smaller_final_batch=True)
#         seq_len = tf.squeeze(seq_len, [1])
#         label_mask = sequence_mask(seq_len, dtype=tf.float32)
#         # label_mask = tf.Print(label_mask, [tf.shape(list_outputs[0]), tf.shape(list_outputs[1]), tf.shape(label_mask), tf.shape(seq_len)], message='batch shape: ')
#
#         return list_outputs[0], list_outputs[1], label_mask, seq_len
#
#     @staticmethod
#     def fentch_batch_with_TFbuckets_NE(list_inputs, args, is_train):
#         """
#         the input tensor length is not equal,
#         so will add the len as a input tensor
#         list_inputs: [tensor1, tensor2]
#         added_list_inputs: [tensor1, tensor2, len_tensor1, len_tensor2]
#         """
#         logging.info("list_batch_size: {}, \t{}".format(len(args.list_batch_size), args.list_batch_size))
#         logging.info("list_bucket_boundaries: {}, \t{}".format(len(args.list_bucket_boundaries), args.list_bucket_boundaries))
#         list_inputs.extend([list_inputs[0], list_inputs[1]])
#         _, list_outputs, *_ = bucket_by_sequence_length(
#             input_length=tf.shape(list_inputs[0]),
#             tensors=list_inputs,
#             batch_size=args.list_batch_size,
#             bucket_boundaries=args.list_bucket_boundaries,
#             num_threads=10,
#             bucket_capacities=[i*3 for i in args.list_batch_size],
#             capacity=2000,
#             dynamic_pad=True,
#             allow_smaller_final_batch=True)
#         # label_mask = tf.Print(label_mask, [tf.shape(list_outputs[0]), tf.shape(list_outputs[1]), tf.shape(label_mask), tf.shape(seq_len)], message='batch shape: ')
#         seq_len_feats = tf.reshape(list_outputs[2], [-1])
#         seq_len_label = tf.reshape(list_outputs[3], [-1])
#
#         return list_outputs[0], list_outputs[1], seq_len_feats, seq_len_label


# def test_numpy_batch_iter():
#     # placeholder
#     dataReader = TFReader(args.dir_tfdata, args=args)
#
#     with tf.train.MonitoredTrainingSession() as sess:
#         dataReader.sess = sess
#         for batch in dataReader.fentch_batch_with_buckets(dataReader):
#             import pdb; pdb.set_trace()


# def test_bucket_boundaries(args, is_train=False):
#     import numpy as np
#
#     dataReader = TFReader(args.dir_dev_data, args=args)
#
#     tfqueue_filename = tf.train.string_input_producer(dataReader.list_tfdata_filenames,
#                                                       num_epochs=1,
#                                                       shuffle=True if is_train else False)
#
#     tensor_seq_features, tensor_seq_labels, reader_tfRecord = filename_queue2iter_tensor(tfqueue_filename, args)
#
#     with tf.train.MonitoredTrainingSession() as sess:
#         list_length = []
#         for _ in tqdm(range(10000)):
#             try:
#                 batch = sess.run(tensor_seq_labels)
#                 list_length.append(len(batch))
#             except tf.errors.OutOfRangeError:
#                 logging.info("out of range !{}")
#                 break
#
#         logging.info("list sample lengths: {}".format(list_length))
#         logging.info("total number of samples: {}".format(len(list_length)))
#         list_length.sort()
#         list_boundaries = [list_length[int(i)-1] for i in np.linspace(0, len(list_length), 20)
#                            if i != 0]
#         list_batch_size = [int(args.num_batch_token/boundary) for boundary in list_boundaries]+[1]
#         logging.info("list_boundaries: {}".format(list_boundaries))
#         logging.info("list_batch_size: {}".format(list_batch_size))


# def test_tfdata_bucket(args, num_threads, capacity=100000, is_train=False):
#     import time
#     logging.info("list_batch_size:{}".format(args.list_batch_size))
#     logging.info("list_bucket_boundaries:{}".format(args.list_bucket_boundaries))
#
#     dataReader = TFReader(args.dir_dev_data, args=args)
#     tfqueue_filename = tf.train.string_input_producer(dataReader.list_tfdata_filenames,
#                                                       num_epochs=1,
#                                                       shuffle=True if is_train else False)
#
#     tensor_seq_features_fromReader, tensor_seq_labels_fromReader, reader_tfRecord = filename_queue2iter_tensor(tfqueue_filename, args)
#
#     queue_tuple_data = tf.RandomShuffleQueue(capacity=capacity,
#                                              min_after_dequeue=int(0.9*capacity),
#                                              dtypes=(tf.float32, tf.int32))
#     op_enqueue = queue_tuple_data.enqueue([tensor_seq_features_fromReader, tensor_seq_labels_fromReader])
#     tf.train.add_queue_runner(tf.train.QueueRunner(queue_tuple_data, [op_enqueue] * num_threads))
#     op_close = queue_tuple_data.close()
#
#     tensor_seq_features, tensor_seq_labels = queue_tuple_data.dequeue()
#     tensor_seq_features.set_shape(tensor_seq_features_fromReader.get_shape())
#     tensor_seq_labels.set_shape(tensor_seq_labels_fromReader.get_shape())
#
#     tensor_batch = dataReader.fentch_batch_with_TFbuckets([tensor_seq_features, tensor_seq_labels],
#                                                           args=args,
#                                                           is_train=True)
#     tensor_num_records_produced = reader_tfRecord.num_records_produced()
#
#     # with tf.train.MonitoredTrainingSession() as sess:
#     #     # sess.run(op_close_tfqueue_filename)
#     #     list_batch_length = []
#     #     for i in range(100):
#     #         batch = sess.run(tensor_batch)
#     #         list_batch_length.append(len(batch[3]))
#     #         logging.info("{}th total number of samples:{}".format(i, sum(list_batch_length)))
#     #         logging.info("num_records_produced:{}".format(sess.run(tensor_num_records_produced)))
#     #     # import pdb; pdb.set_trace()
#
#     with tf.Session() as sess:
#         sess.run(tf.local_variables_initializer())
#         sess.run(tf.global_variables_initializer())
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#         # for i in range(10):
#         #     try:
#         #         sess.run(op_enqueue)
#         #     except tf.errors.OutOfRangeError:
#         #         # sess.run(op_close)
#         #         break
#
#         time.sleep(2)
#         logging.info("num_records_produced:{}".format(sess.run(tensor_num_records_produced)))
#         list_batch_length = []
#
#         for i in range(100):
#             try:
#                 sess.run(op_close)
#                 batch = sess.run(tensor_batch)
#                 list_batch_length.append(len(batch[3]))
#                 logging.info("{}th total number of samples:{}".format(i, sum(list_batch_length)))
#                 logging.info("num_records_produced:{}".format(sess.run(tensor_num_records_produced)))
#                 time.sleep(0.1)
#             except tf.errors.OutOfRangeError:
#                 logging.info("out of range!!!!!")
#                 list_batch_length.append(len(batch[3]))
#                 logging.info("{}th total number of samples:{}".format(i, sum(list_batch_length)))
#                 logging.info("num_records_produced:{}".format(sess.run(tensor_num_records_produced)))
#                 break
#
#         coord.request_stop()
#         coord.join(threads)


if __name__ == '__main__':
    from configs.arguments import args
    from tqdm import tqdm
    import sys

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')
    # test_bucket_boundaries(args=args)
    # test_tfdata_bucket(args=args, num_threads=args.num_parallel)
    # test_queue()
