#!/usr/bin/env python
import numpy as np
from time import time
import logging
from utils.textTools import array2text
import editdistance as ed
from itertools import repeat

from .textTools import array_idx2char, unpadding, batch_wer, batch_cer, array2text


def dev(step, dataloader, model, sess, unit, idx2token, eos_idx=None, min_idx=0, max_idx=None):
    start_time = time()
    batch_time = time()
    processed = 0

    total_cer_dist = 0
    total_cer_len = 0

    total_wer_dist = 0
    total_wer_len = 0

    for batch in dataloader:
        if not batch: continue
        dict_feed = {model.list_pl[0]: batch[0],
                     model.list_pl[1]: batch[2]}
        decoded, shape_batch, _ = sess.run(model.list_run, feed_dict=dict_feed)
        # import pdb; pdb.set_trace()

        batch_cer_dist, batch_cer_len = batch_cer(
            result=decoded,
            reference=batch[1],
            eos_idx=eos_idx,
            min_idx=min_idx,
            max_idx=max_idx)
        _cer = batch_cer_dist/batch_cer_len
        total_cer_dist += batch_cer_dist
        total_cer_len += batch_cer_len

        batch_wer_dist, batch_wer_len = batch_wer(
            result=decoded,
            reference=batch[1],
            idx2token=idx2token,
            unit=unit,
            eos_idx=eos_idx,
            min_idx=min_idx,
            max_idx=max_idx)
        _wer = batch_wer_dist/batch_wer_len
        total_wer_dist += batch_wer_dist
        total_wer_len += batch_wer_len

        used_time = time()-batch_time
        batch_time = time()
        processed += shape_batch[0]
        progress = processed/len(dataloader)
        logging.warning('batch cer: {:.3f}\twer: {:.3f} batch: {}\t time:{:.2f}s {:.3f}%'.format(
                     _cer, _wer, shape_batch, used_time, progress*100.0))
    used_time = time() - start_time
    cer = total_cer_dist/total_cer_len
    wer = total_wer_dist/total_wer_len
    logging.warning('=====dev info, total used time {:.2f}h==== \nWER: {:.4f}\ntotal_wer_len: {}'.format(
                 used_time/3600, wer, total_wer_len))

    return cer, wer


def decode_test(step, sample, model, sess, unit, idx2token, eos_idx=None, min_idx=0, max_idx=None):
    # sample = dataset_dev[0]
    dict_feed = {model.list_pl[0]: np.expand_dims(sample['feature'], axis=0),
                 model.list_pl[1]: np.array([len(sample['feature'])])}
    sampled_id, shape_sample, _ = sess.run(model.list_run, feed_dict=dict_feed)

    res_txt = array2text(sampled_id[0], unit, idx2token, eos_idx, min_idx, max_idx)
    ref_txt = array2text(sample['label'], unit, idx2token, eos_idx, min_idx, max_idx)

    logging.warning('length: {}, res: \n{}\nref: \n{}'.format(
                 shape_sample[1], res_txt, ref_txt))
