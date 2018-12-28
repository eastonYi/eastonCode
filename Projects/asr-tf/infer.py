#!/usr/bin/env python
from datetime import datetime
from time import time
import sys
import os
import logging
import tensorflow as tf
from arguments import args
from tqdm import tqdm
import numpy as np
import editdistance as ed

from tfTools.tfTools import get_session
from tfModels.tools import create_embedding, size_variables

from tfTools.tfRecord import TFReader, readTFRecord
from datasets import ASRDataLoader

from utils.summaryTools import Summary
from utils.performanceTools import dev, decode_test
from utils.textTools import array_idx2char, array2text, batch_cer
from utils.tools import check_to_stop
from utils.ctc_decode import rna_reduce_map

from tfSeq2SeqModels.ctc_LM_Model import CTCLMModel as Model


def infer():
    tensor_global_step = tf.train.get_or_create_global_step()
    dataset_dev = args.dataset_test if args.dataset_test else args.dataset_dev

    model_lm = args.Model_LM(
        tensor_global_step,
        is_train=False,
        args=args.args_lm)

    args.lm = model_lm
    saver_lm = tf.train.Saver(model_lm.variables())

    model_infer = args.Model(
        tensor_global_step,
        encoder=args.model.encoder.type,
        decoder=args.model.decoder.type,
        decoder2=args.model.decoder2.type,
        is_train=False,
        args=args)
    saver = tf.train.Saver(model_infer.variables())

    size_variables()

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    with tf.train.MonitoredTrainingSession(config=config) as sess:
        checkpoint = tf.train.latest_checkpoint(args.dirs.checkpoint_init)
        checkpoint_lm = tf.train.latest_checkpoint(args.dirs.lm_checkpoint)
        saver.restore(sess, checkpoint)
        saver_lm.restore(sess, checkpoint_lm)

        total_cer_dist = 0
        total_cer_len = 0
        total_wer_dist = 0
        total_wer_len = 0
        with open(args.dir_model.name+'_decode.txt', 'w') as fw:
            for sample in tqdm(dataset_dev):
                if not sample:
                    continue
                dict_feed = {model_infer.list_pl[0]: np.expand_dims(sample['feature'], axis=0),
                             model_infer.list_pl[1]: np.array([len(sample['feature'])])}
                sample_id, shape_batch, beam_decoded = sess.run(model_infer.list_run, feed_dict=dict_feed)
                # decoded, sample_id, decoded_sparse = sess.run(model_infer.list_run, feed_dict=dict_feed)
                res_txt = array2text(sample_id[0], args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output-1)
                ref_txt = array2text(sample['label'], args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output-1)
                for decoded in beam_decoded:
                    print(array2text(decoded, args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output-1))

                list_res_char = list(res_txt)
                list_ref_char = list(ref_txt)
                list_res_word = res_txt.split()
                list_ref_word = ref_txt.split()
                cer_dist = ed.eval(list_res_char, list_ref_char)
                cer_len = len(list_ref_char)
                wer_dist = ed.eval(list_res_word, list_ref_word)
                wer_len = len(list_ref_word)
                total_cer_dist += cer_dist
                total_cer_len += cer_len
                total_wer_dist += wer_dist
                total_wer_len += wer_len
                if cer_len == 0:
                    cer_len = 1000
                    wer_len = 1000
                if wer_dist/wer_len > 0:
                    fw.write('id:\t{} \nres:\t{}\nref:\t{}\n\n'.format(sample['id'], res_txt, ref_txt))
                logging.info('current cer: {:.3f}, wer: {:.3f};\tall cer {:.3f}, wer: {:.3f}'.format(
                    cer_dist/cer_len, wer_dist/wer_len, total_cer_dist/total_cer_len, total_wer_dist/total_wer_len))
        logging.info('dev CER {:.3f}:  WER: {:.3f}'.format(total_cer_dist/total_cer_len, total_wer_dist/total_wer_len))


if __name__ == '__main__':
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode', default='train')
    parser.add_argument('--name', type=str, dest='name', default=None)
    parser.add_argument('--gpu', type=str, dest='gpu', default=0)
    parser.add_argument('-c', type=str, dest='config')

    param = parser.parse_args()

    print('CUDA_VISIBLE_DEVICES: ', args.gpus)

    if param.mode == 'infer':
        os.environ["CUDA_VISIBLE_DEVICES"] = param.gpu
        logging.info('enter the INFERING phrase')
        infer()
