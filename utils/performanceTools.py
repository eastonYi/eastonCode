#!/usr/bin/env python

import numpy as np
from time import time
import logging
import editdistance as ed

from .textTools import array_idx2char, unpadding, batch_wer, batch_cer


def dev(step, dataloader, model, sess, unit, idx2token, eosid_res, eosid_ref):
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

        batch_cer_dist, batch_cer_len = batch_cer(
            result=decoded,
            reference=batch[1],
            eosid_res=eosid_res,
            eosid_ref=eosid_ref)
        _cer = batch_cer_dist/batch_cer_len
        total_cer_dist += batch_cer_dist
        total_cer_len += batch_cer_len

        batch_wer_dist, batch_wer_len = batch_wer(
            result=decoded,
            reference=batch[1],
            eosid_res=eosid_res,
            eosid_ref=eosid_ref,
            idx2token=idx2token,
            unit=unit)
        _wer = batch_wer_dist/batch_wer_len
        total_wer_dist += batch_wer_dist
        total_wer_len += batch_wer_len

        used_time = time()-batch_time
        batch_time = time()
        processed += shape_batch[0]
        progress = processed/len(dataloader)
        logging.info('batch cer: {:.3f}\twer: {:.3f} batch: {}\t time:{:.2f}s {:.3f}%'.format(
                     _cer, _wer, shape_batch, used_time, progress*100.0))
    used_time = time() - start_time
    cer = total_cer_dist/total_cer_len
    wer = total_wer_dist/total_wer_len
    logging.info('=====dev info, total used time {:.2f}h==== \nWER: {:.3f}'.format(used_time/3600, wer))

    return cer, wer


def decode_test(step, sample, model, sess, unit, idx2token, eosid_res, eosid_ref):
    # sample = dataset_dev[0]
    dict_feed = {model.list_pl[0]: np.expand_dims(sample['feature'], axis=0),
                 model.list_pl[1]: np.array([len(sample['feature'])])}
    sampled_id, shape_sample, _ = sess.run(model.list_run, feed_dict=dict_feed)

    res = unpadding(sampled_id[0], eosid_res)
    ref = unpadding(sample['label'], eosid_ref)
    if unit == 'char':
        result_txt = array_idx2char(res, idx2token, seperator='')
        ref_txt = array_idx2char(ref, idx2token, seperator='')
    elif unit == 'word':
        result_txt = ' '.join(array_idx2char(res, idx2token, seperator=' ').split())
        ref_txt = array_idx2char(ref, idx2token, seperator=' ')
    elif unit == 'subword':
        result_txt = array_idx2char(res, idx2token, seperator=' ').replace('@@ ', '')
        ref_txt = array_idx2char(ref, idx2token, seperator=' ').replace('@@ ', '')
    else:
        raise 'unknown model unit! '

    logging.info('length: {}, res: \n{}\nref:\n{}'.format(
                 shape_sample[1], result_txt, ref_txt))
