#!/usr/bin/python
# coding=utf-8

import logging
import sys

from dataProcessing.dataIter import DataReader
# from utils.tools import padding_list_seqs
# from utils.unix_tools import *


class ScpReader(DataReader):
    '''
    read ark2txt files(kaldi data type)
    need to sort first according to the 1.filename then
    depend on random pointer to shuffle. the list is ordered

    - iter_seq(self)
    - fentch_batch_with_buckets(args, iter_seq):
    '''
    def __init__(self, path_scpfile, args=None):
        self.list_uttids = []
        self.pointer_uttid = 0
        self.uttid2path = {}
        self.uttid2seqfeats = {}
        self.path_current_pkl = None
        self.generate_uttid2path(path_scpfile)
        # self.sort()

    def __iter__(self):
        for uttid in self.list_uttids:
            if self.uttid2path[uttid] != self.path_current_pkl:
                self.path_current_pkl = self.uttid2path[uttid]
                self.uttid2seqfeats = readPkl(self.path_current_pkl)

            yield self.uttid2seqfeats[uttid]

    def generate_uttid2path(self, path_scpfile):
        with open(path_scpfile, "r") as f_scp:
            for line in f_scp:
                line = line.strip()
                if line:
                    utt_id, path_and_pos = line.split(' ')
                    path_ark, _ = path_and_pos.split(':')
                    self.list_uttids.append(utt_id)
                    path_pkl = path_ark.replace('.ark', '.pkl')
                    if not sys.path.exists(path_pkl):
                        raise FileNotFoundError('{} not exist!'.format(path_pkl))
                    else:
                        self.uttid2path[utt_id] = path_pkl

    def sort(self):
        '''
        make sure that filenames in scp_file are ordered by name!!!
        if not, you need to call this to sort the list_uttids after calling the
        generate_uttid2path() in __init__
        '''
        # self.uttid2path
        self.list_uttids.sort()

    # def fentch_batch(self, size_batch):
    #     batch_uttids = []
    #     filename = None
    #
    #     for i in range(size_batch):
    #         if (i+self.pointer_uttid) >= len(self.list_uttids):
    #             self.pointer_uttid = 0
    #             i = -1
    #             break
    #         uttid = self.list_uttids[i+self.pointer_uttid]
    #         if not filename: filename = self.uttid2path[uttid]
    #
    #         if self.uttid2path[uttid] == filename:
    #             batch_uttids.append(uttid)
    #             continue
    #         else:
    #             i -=1
    #             break
    #     self.pointer_uttid +=(i+1)
    #     # if self.pointer_uttid > len(self.list_uttids):
    #     #     self.pointer_uttid = 0
    #
    #     if filename != self.path_current_pkl:
    #         self.uttid2seqfeats = readPkl(filename)
    #         self.path_current_pkl = filename
    #
    #     return padding_list_seqs([self.uttid2seqfeats[uttid] for uttid in batch_uttids])


def readPkl(filename):
    import pickle
    with open(filename, 'rb') as f:
        uttid2seqfeats = pickle.load(f)
        if not uttid2seqfeats:
            logging.info('empty pkl file')

        return uttid2seqfeats


if __name__ == '__main__':
    from argparse import ArgumentParser

    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')

    def config():
        parser = ArgumentParser()
        parser.add_argument("--size_bucket_start", type=int, default=0)
        parser.add_argument("--size_bucket_end", type=int, default=1000)
        parser.add_argument("--size_bucket_gap", type=int, default=20)
        parser.add_argument("--size_features_in_bucket", type=int, dest='size_features_in_bucket', default=2000)
        parser.add_argument("-f", type=str, dest='path_scp_file')
        return parser.parse_args()
    args = config()

    scpReader = ScpReader(args.path_scp_file)
    for batch in ScpReader.fentch_batch_with_buckets(scpReader, args):
        import pdb; pdb.set_trace()
