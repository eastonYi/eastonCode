import os
import numpy as np
import logging
from random import random, randint
from queue import Queue
import threading
import time
import collections
from random import shuffle, random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataProcessing.audio import audio2vector, process_raw_feature
from abc import ABCMeta, abstractmethod

from utils.tools import size_bucket_to_put

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class DataSet:
    __metaclass__ = ABCMeta

    def __iter__(self):
        """
        utility the __getitem__ to impliment the __iter__
        """
        for idx in range(len(self)):
            yield self[idx]

    @abstractmethod
    def __getitem__(self, idx):
        """
        sample = dataset_obj[idx]
        sample = dataset_obj(idx)
        """

    @abstractmethod
    def __len__(self):
        """
        the length of the dataset
        """

    def __call__(self, idx):

        return self.__getitem__(idx)


class ASRDataSet(DataSet):
    def __init__(self, list_files, args, shuffle, transform):
        self.list_files = list_files
        self.transform = transform
        self.args = args
        self.shuffle = shuffle
        self.token2idx, self.idx2token = args.token2idx, args.idx2token
        self.end_id = self.gen_end_id(self.token2idx)

    def gen_end_id(self, token2idx):
        if '<eos>' in token2idx.keys():
            eos_id = [token2idx['<eos>']]
        else:
            eos_id = []

        return eos_id


class ASR_csv_DataSet(ASRDataSet):
    def __init__(self, list_files, args, shuffle, transform):
        super().__init__(list_files, args, shuffle, transform)
        self.list_utterances = self.gen_utter_list(list_files)
        if shuffle:
            self.shuffle_list_files()

    def __getitem__(self, idx):
        utterance = self.list_utterances[idx]
        wav, seq_label = utterance.strip().split(',')
        fea = audio2vector(wav, self.args.data.dim_raw_input)
        if self.transform:
            fea = process_raw_feature(fea, self.args)

        seq_label = np.array(
            [self.token2idx.get(word, self.token2idx['<unk>'])
            for word in seq_label.split(' ')] + self.end_id,
            dtype=np.int32)

        sample = {'id': wav, 'feature': fea, 'label': seq_label}

        return sample

    @staticmethod
    def gen_utter_list(list_files):
        list_utter = []
        for file in list_files:
            with open(file) as f:
                list_utter.extend(f.readlines())
        return list_utter

    def shuffle_list_files(self):
        shuffle(self.list_utterances)

    def __len__(self):
        return len(self.list_utterances)


class ASR_scp_DataSet(ASRDataSet):
    def __init__(self, f_scp, f_trans, f_vocab, args, shuffle, transform):
        """
        Args:
            f_scp: the scp file consists of paths to feature data
            f_trans: the scp file consists of id and trans
            f_id2label: the normalized transcripts
            f_vocab: the vocab of the transcripts
        """
        from dataProcessing.ark import ArkReader
        self.list_files = [f_scp]
        super().__init__(self.list_files, args, shuffle, transform)
        self.reader = ArkReader(f_scp)
        self.id2trans = self.gen_id2trans(f_trans)

    def __getitem__(self, idx):
        sample = {}

        try:
            sample['feature'] = self.reader.read_utt_data(idx)
            if self.transform:
                sample['feature'] = process_raw_feature(sample['feature'], self.args)

            trans = self.id2trans[self.reader.utt_ids[idx]]
            sample['label'] = np.array(
                [self.token2idx.get(token, self.token2idx['<unk>'])
                for token in trans] + self.end_id,
                dtype=np.int32)
            sample['id'] = self.reader.utt_ids[idx]
        except:
            print('Not fond {}!'.format(self.reader.utt_ids[idx]))
            sample = None

        return sample

    def __len__(self):
        return len(self.reader.utt_ids)

    def gen_id2trans(self, f_trans):
        id2trans = {}
        with open(f_trans) as f:
            for line in f:
                line = line.strip().split()
                id = line[0]
                trans = line[1:]
                id2trans[id] = trans

        return id2trans


class FakeDataSet(DataSet):
    def __init__(self):
        self.dim_feature = 3

    def __getitem__(self, idx):
        sample = {}
        sample['label'] = np.random.randint(self.dim_feature, size=randint(5, 10), dtype=np.int32)
        sample['feature'] = self.embedding(sample['label'])

        return sample

    def __len__(self):

        return 100

    def embedding(self, list_idx):
        list_embeded = []
        for idx in list_idx:
            embeded = np.zeros([self.dim_feature], dtype=np.float32)
            embeded[idx] = 1
            list_embeded.append(embeded)

        return list_embeded


class SimpleDataLoader:
    def __init__(self, dataset, num_loops=1, batch_size=10):
        self.dataset = dataset
        self.num_loops = num_loops
        self.batch_size = batch_size
        self.list_seq_features = []
        self.list_seq_labels = []

    def __iter__(self):
        return self.next_batch(self.batch_size)

    def next_batch(self, size_batch):
        for _ in range(self.num_loops):
            for sample in self.dataset:
                seq_features, seq_labels = sample['feature'], sample['label']

                self.list_seq_features.append(seq_features)
                self.list_seq_labels.append(seq_labels)

                if len(self.list_seq_features) >= size_batch:
                    yield self.padding_list_seq_with_labels(self.list_seq_features, self.list_seq_labels)
                    self.list_seq_features = []
                    self.list_seq_labels = []

    @staticmethod
    def padding_list_seqs(list_seqs, dtype=np.float32, pad=0.):
        '''
        Pads each sequence to the same length of the longest sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens.

        Args:
            list_seqs: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            pad: float, value to pad the list_seqs to the desired value.

        Returns:
            numpy.ndarray: Padded list_seqs shape = (number_of_list_seqs, maxlen)
            list: original sequence lengths
        DEmo:
            >> padding_list_seqs([[21, 11, 3], [31,1]])
            >> (array([[ 21.,  11.,   3.],
                [ 31.,   1.,   0.]], dtype=float32), [3, 2])
        '''
        len_x = [len(s) for s in list_seqs]

        size_batch = len(list_seqs)
        maxlen = max(len_x)

        # take the sample shape from the first non empty sequence
        # checking for consistency in the main loop below.
        shape_feature = tuple()
        for s in list_seqs:
            if len(s) > 0:
                shape_feature = np.asarray(s).shape[1:]
                break

        # a tensor filled with padding value
        x = (np.ones((size_batch, maxlen) + shape_feature) * pad).astype(dtype)
        for idx, s in enumerate(list_seqs):
            x[idx, :len(s)] = s

        return x, len_x

    @staticmethod
    def padding_list_seq_with_labels(list_seqs_features,
                                     list_seqs_labels,
                                     dtype=np.float32,
                                     value1=0.,
                                     value2=0):
        x, len_x = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_features,
            dtype=dtype,
            pad=value1)
        y, len_y = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_labels,
            dtype=np.int32,
            pad=value2)

        return [x, y, len_x, len_y]


class DataLoader(SimpleDataLoader):
    __metaclass__ = ABCMeta

    '''
    Train/test/dev dataset API for loading via threads and delivering batches.
    '''
    def __init__(self, dataset, args, num_loops=1, num_thread=4, size_queue=2000):
        super().__init__(dataset, num_loops)
        self.args = args
        self.num_thread = num_thread
        self.queue_sample = Queue(maxsize=size_queue)

        self.thread_queue_put = threading.Thread(target=self.feed_queue)
        self.thread_queue_put.daemon = True #http://www.xavierdupre.fr/blog/2013-11-02_nojs.html

        self.num_batch_token = args.num_batch_token
        self.bucket_boundaries = args.bucket_boundaries

    @abstractmethod
    def __iter__(self):
        '''
        return a iterator of seq, which is used to fentch a batch(with or without bucket)
        yield (seq_features, seq_labels)
        '''

    def batch_with_buckets(self):
        '''
        use the iter_seq:
        ```python
        args.size_bucket_start =
        args.size_bucket_end =
        args.size_bucket_gap =
        args.self.num_batch_token =
        dataReader = DataLoader(...)
        for batch in DataLoader.fentch_batch_with_buckets(dataReader)
            ...
        ```
        caches:
            {5: [[], [], 0],
             8: [[], [], 0],
            11: [[], [], 0],
            14: [[], [], 0],
            17: [[], [], 0]}
            id: [list_feats, list_labels, num_frame]
        len(caches[bucket][0]) is the batch length, i.e. the num of sents in a batch,
        while caches[bucket][1] is the num of tokens in a batch
        '''
        buckets = self.args.list_bucket_boundaries
        # max_length = buckets[-1]
        caches = collections.defaultdict(lambda: [[], [], 0])

        for sample in self.dataset:
            if not sample: continue
            seq_features, seq_labels = sample['feature'], sample['label']
            # assert len(seq_features) == len(seq_labels)
            id_bucket, bucket = size_bucket_to_put(len(seq_features), buckets)
            if bucket is None:
                continue
            caches[bucket][0].append(seq_features)
            caches[bucket][1].append(seq_labels)

            caches[bucket][2] += 1
            if caches[bucket][2] >= self.args.list_batch_size[id_bucket]:
                batch = (caches[bucket][0], caches[bucket][1])
                yield self.padding_list_seq_with_labels(*batch)
                caches[bucket] = [[], [], 0]

        # Clean remain samples.
        for bucket in buckets:
            if caches[bucket][0]:
                batch = (caches[bucket][0], caches[bucket][1])
                yield self.padding_list_seq_with_labels(*batch)
                caches[bucket] = [[], [], 0]
                logging.info('empty the bucket {}'.format(bucket))

    def feed_queue(self):
        logging.info('enter the feed queue thread!')
        with ThreadPoolExecutor(self.num_thread) as ex:
            for idx in range(0, len(self.dataset)-self.num_thread, self.num_thread):
                batch_samples = ex.map(self.dataset, range(idx, idx+self.num_thread))
                # logging.info('add success!')
                [self.queue_sample.put(sample) for sample in batch_samples]

        self.dataset.shuffle_list_files()

    def bucket_with_queue(self):
        '''
        caches: {bucket_size: [list_feats, list_labels, num_frame]}
        '''
        self.thread_queue_put.start()
        logging.info('the activate num threads to prepare data is: {}'.format(threading.active_count()-2))
        index_loop = 0
        # feed_queue()
        buckets = self.args.list_bucket_boundaries
        batch_size = self.args.list_batch_size

        # max_length = buckets[-1]
        caches = collections.defaultdict(lambda: [[], [], 0])

        logging.info("size of the dataset: {}".format(len(self.dataset)))

        while True:
            sample = self.queue_sample.get()
            seq_features, seq_labels = sample['feature'], sample['label']

            # assert len(seq_features) == len(seq_labels)
            id_bucket, bucket = size_bucket_to_put(len(seq_features), buckets)
            if bucket is None:
                continue
            caches[bucket][0].append(seq_features)
            caches[bucket][1].append(seq_labels)
            caches[bucket][2] += 1

            if caches[bucket][2] >= batch_size[id_bucket]:
                batch = (caches[bucket][0], caches[bucket][1])
                yield self.padding_list_seq_with_labels(*batch)
                caches[bucket] = [[], [], 0]

            if self.queue_sample.empty():
                if threading.active_count() > 2:
                    logging.info('waitting for sample into the queue...')
                    print()
                    # logging.info('the activate num threads to prepare data is: {}'.format(threading.active_count()-2))
                    time.sleep(3)
                elif index_loop < self.num_loops-1:
                    index_loop +=1
                    # logging.info('brefore the activate num threads to prepare data is: {}'.format(threading.active_count()-2))
                    self.thread_queue_put.join()
                    # logging.info('after the activate num threads to prepare data is: {}'.format(threading.active_count()-2))
                    self.thread_queue_put = threading.Thread(target=self.feed_queue)
                    self.thread_queue_put.start()
                    logging.info('***=======  loop {}/{} for the dataset  =======***'.format(index_loop+1, self.num_loops))
                else:
                    logging.info('finish iter dataset {} times'.format(self.num_loops))
                    break

        # Clean remain samples.
        for bucket in buckets:
            if caches[bucket][0]:
                batch = (caches[bucket][0], caches[bucket][1])
                yield self.padding_list_seq_with_labels(*batch)
                caches[bucket] = [[], [], 0]
                logging.info('empty the bucket {}'.format(bucket))

        self.thread_queue_put.join()

    def batch_with_map(self, size_batch):
        with ThreadPoolExecutor(self.num_thread) as ex:
            for idx in range(0, len(self.dataset)-size_batch, size_batch):
                batch_samples = ex.map(self.dataset, range(idx, idx+size_batch))

                self.list_seq_features = []
                self.list_seq_labels = []

                for sample in batch_samples:
                    if not sample: continue
                    seq_features, seq_labels = sample['feature'], sample['label']

                    self.list_seq_features.append(seq_features)
                    self.list_seq_labels.append(seq_labels)

                yield self.padding_list_seq_with_labels(
                    self.list_seq_features,
                    self.list_seq_labels)

    def batch_with_tfReader(self, batch_size):
        for _ in range(len(self)*self.num_loops):
            seq_features, seq_labels = self.sess.run([self.feat, self.label])

            self.list_seq_features.append(seq_features)
            self.list_seq_labels.append(seq_labels)

            if len(self.list_seq_labels) >= batch_size:
                yield self.padding_list_seq_with_labels(
                    self.list_seq_features,
                    self.list_seq_labels)
                self.list_seq_features = []
                self.list_seq_labels = []

        logging.info("clean the rest of data")
        if len(self.list_seq_features) > 0:
            yield self.padding_list_seq_with_labels(
                self.list_seq_features,
                self.list_seq_labels)
            self.list_seq_features = []
            self.list_seq_labels = []


class ASRDataLoader(DataLoader):
    def __init__(self, dataset, args, feat, label, batch_size, num_loops, num_thread=4, size_queue=2000):
        super().__init__(dataset, args, num_loops=num_loops, num_thread=num_thread, size_queue=size_queue)
        self.sess = None
        self.feat = feat
        self.label = label
        self.size_dataset = len(dataset)
        self.batch_size = batch_size

    def __iter__(self):
        return self.batch_with_tfReader(self.batch_size)

    def __len__(self):
        return self.size_dataset


if __name__ == '__main__':
    dataset = FakeDataSet()
    dataloader = SimpleDataLoader(dataset, 1)

    for i in dataloader:
        import pdb; pdb.set_trace()
        print(i)
