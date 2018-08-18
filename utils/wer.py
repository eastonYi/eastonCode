#-*- coding: utf-8 -*-
#!/usr/bin/env python
# modified from zszyellow/WER-in-python, https://github.com/zszyellow/WER-in-python
# not recommend for high-speed requirement! This is only for demostration!
# turn to https://github.com/aflc/editdistance and https://github.com/miohtama/python-Levenshtein
# for faaaaaast edit compution

import sys
import numpy as np

def editDistance(hyp, ref):
    '''
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.

    Main algorithm used is dynamic programming.

    Attributes:
        hyp: the list of words produced by splitting hypothesis sentence.
        ref: the list of words produced by splitting reference sentence.

    d:   r e f
       h
       y
       p
    '''
    assert (len(hyp) < 200) and (len(ref) < 200)
    d = np.zeros((len(hyp)+1, len(ref)+1), dtype=np.uint8)
    d[0, :] = np.arange(len(ref)+1)
    d[:, 0] = np.arange(len(hyp)+1)
    for i in range(1, len(hyp)+1):
        for j in range(1, len(ref)+1):
            if hyp[i-1] == ref[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d


def editDistance_v2(hyp, ref, table, idx_hyp):
    '''
    hyp' crresponding to table has changed after idx_hyp, and continue to
    calculate the table.
    d:   _ r e f
       |
       h
      y->i (idx_hyp+1)
       p
    '''
    assert (len(hyp) < 200) and (len(ref) < 200)
    d = np.zeros((len(hyp)+1, len(ref)+1), dtype=np.uint8)
    d[:idx_hyp+1] = table[:idx_hyp+1]
    d[:, 0] = np.arange(len(hyp)+1)
    for i in range(idx_hyp+1, len(hyp)+1):
        for j in range(1, len(ref)+1):
            if hyp[i-1] == ref[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d


def getStepList(hyp, ref, d):
    '''
    This function is to get the list of steps in the process of dynamic programming.

    Attributes:
        hyp -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calulating the editting distance of h and hyp.
    '''
    x = len(hyp)
    y = len(ref)
    list = []
    while True:
        if x == 0 and y == 0:
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1] and hyp[x-1] == ref[y-1]:
            list.append("e")
            x = x - 1
            y = y - 1
        elif y >= 1 and d[x][y] == d[x][y-1]+1:
            list.append("i")
            x = x
            y = y - 1
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1]+1:
            list.append("s")
            x = x - 1
            y = y - 1
        else:
            list.append("d")
            x = x - 1
            y = y
    return list[::-1]


def info_edit(hyp, ref, d):
    '''
    This function is to get the list of steps in the process of dynamic programming.

    Attributes:
        hyp -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calulating the editting distance of h and hyp.
    '''
    x = len(hyp)
    y = len(ref)
    list = []
    while True:
        if x == 0 and y == 0:
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1] and hyp[x-1] == ref[y-1]:
            # list.append("e")
            x -= 1
            y -= 1
        elif y >= 1 and d[x][y] == d[x][y-1]+1:
            list.append((x, "i", ref[y-1]))
            y -= 1
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1]+1:
            list.append((x-1, "s", ref[y-1]))
            x -= 1
            y -= 1
        else:
            list.append((x-1, "d", hyp[x-1]))
            x -= 1
    return list[::-1]


def alignedPrint(list, hyp, ref, wer):
    '''
    This funcition is to print the result of comparing reference and hypothesis sentences in an aligned way.

    Attributes:
        list   -> the list of steps.
        hyp      -> the list of words produced by splitting reference sentence.
        h      -> the list of words produced by splitting hypothesis sentence.
        result -> the rate calculated based on edit distance.
    '''
    blank = ' '
    sys.stdout.write("REF:")
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            sys.stdout.write(blank * (len(hyp[index])))
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(hyp[index1]) > len(ref[index2]):
                sys.stdout.write(ref[index2] + blank * (len(hyp[index1])-len(ref[index2])))
            else:
                sys.stdout.write(ref[index2])
        else:
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            sys.stdout.write(ref[index])
        sys.stdout.write(blank)
    sys.stdout.write("\nHYP:")
    for i in range(len(list)):
        if list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            sys.stdout.write(blank*(len(ref[index])))
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(hyp[index1]) < len(ref[index2]):
                sys.stdout.write(hyp[index1] + blank * (len(ref[index2])-len(hyp[index1])))
            else:
                sys.stdout.write(hyp[index1])
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            sys.stdout.write(hyp[index])
        sys.stdout.write(blank)
    sys.stdout.write("\nEVA:")
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            sys.stdout.write("D" + blank * (len(hyp[index])-1))
        elif list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            sys.stdout.write("I" + blank * (len(ref[index])-1))
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(hyp[index1]) > len(ref[index2]):
                sys.stdout.write("S" + blank * (len(hyp[index1])-1))
            else:
                sys.stdout.write("S" + blank * (len(ref[index2])-1))
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            sys.stdout.write(blank * (len(hyp[index])))
        sys.stdout.write(blank)
    sys.stdout.write("\nWER: " + str("%.2f" % (wer * 100.0)) + "%\n")


def wer(hyp, ref, no_print=False):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split())
    support Chinese compute but align-print is not fine.
    """
    if type(hyp) is str:
        assert type(ref) is str
        hyp = hyp.split()
        ref = ref.split()
    else:
        assert type(hyp) is list
        assert type(ref) is list

    # build the matrix
    d = editDistance(hyp, ref)

    # find out the manipulation steps
    result_list = getStepList(hyp, ref, d)

    # print the result in aligned way
    wer = float(d[len(hyp)][len(ref)]) / len(ref)
    if not no_print:
        alignedPrint(result_list, hyp, ref, wer)

    return wer


def cer(hyp, ref, no_print=False):
    """
    remove the space and insert a space between every two characters.
    """
    return wer(list(hyp), list(ref), no_print)


def segment(a, b):
    """
    edit a to match b
    """
    if min(len(a), len(b)) == 0:
        print(a, b)
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            break
    return i


def test_ed_reverse():
    """
    edit distances are equal when reverse the sequences
    """
    from utils.ctc_numpy import ctc_reduce_map
    from numpy.random import randint

    seqs = []
    for length in randint(20, 40, size=10000):
        seq = []
        for i in range(length):
            seq.append(randint(4))
        seqs.append(seq)

    samples = ctc_reduce_map(seqs)
    r = [1,1,2,1,2,2,1,2,2,2]
    for i in samples:
        assert editDistance(i, r)[-1, -1] == editDistance(i[::-1], r[::-1])[-1, -1]


def test_editDistance():
    hs = np.array([[1,2,3,2,2],
                   [2,2,1,2,2],
                   [3,1,2,3,1]], dtype=np.uint8)
    ref = [2,2,2,2,2]
    for hpy in hs:
        d = editDistance(hyp, ref)
        print(d[-1,-1])


def test_editDistance_v2():
    r = [1,1,2,3,4,4,1,2,1]
    h1 = [1,2,1,3,2,3,2,2,2]
    h2 = [1,2,1,3,1,1,2,3]

    e1 = editDistance(h1, r)
    e2 = editDistance(h2, r)
    print(e1[-1,-1], e2[-1,-1])

    print(editDistance_v2(h2, r, e1, 4)[-1,-1])


def editDistance_batch():
    """
    for list of hyps, there are only some place that is different between any two of them
    and the locs are given by list_loc
    """
    from utils.ctc_numpy import ctc_reduce_map
    from numpy.random import randint
    from time import time
    # align_sample = [0,0,1,1,0,0,0,2,2,0,0,2,2,0,3,3,3,3,0,0,1,0,1,1,0,0]
    align_sample = randint(low=0, high=4, size=250)
    # align_sample = [0,3,3,3,0,0,1,0]
    print('align_sample: ', align_sample)
    list_sample = [align_sample]
    for t in range(len(align_sample)):
        for k in range(3):
            align = align_sample.copy()
            align[t] = k
            list_sample.append(align)

    hs = ctc_reduce_map(list_sample)
    r = randint(low=1, high=4, size=randint(90, 140))

    d_0 = editDistance(hs[0], r)
    d_0_reverse = editDistance(hs[0][::-1], r[::-1])
    assert d_0[-1,-1] == d_0_reverse[-1,-1]

    time1 = time()
    ed_1 = []
    for i, h in enumerate(hs):
        if segment(h, hs[0]) > len(h)/2:
            d = editDistance_v2(h, r, d_0, idx_hyp=segment(h, hs[0]))
        else:
            d = editDistance_v2(h[::-1], r[::-1], d_0_reverse, idx_hyp=segment(h[::-1], hs[0][::-1]))
        ed_1.append(d[-1,-1])
    print('mix speed-up-ed used ', time()-time1, 's')

    ed_3 = []
    time3 = time()
    for i, h in enumerate(hs):
        d = editDistance_v2(h[::-1], r[::-1], d_0_reverse, idx_hyp=segment(h[::-1], hs[0][::-1]))
        ed_3.append(d[-1,-1])
    print('reverse speed-up-ed used ', time()-time3, 's')

    print(ed_1[:3])
    print(ed_3[:3])
    assert(ed_1 == ed_3)

    time2 = time()
    ed_2 = []
    for h in hs:
        d = editDistance(h, r)
        ed_2.append(d[-1,-1])
        # print('std: ', d[-1,-1])
    assert(ed_1 == ed_2)
    print('std ed used ', time()-time2, 's')


if __name__ == '__main__':
    hyp = 'he is a big gay , '.split()
    ref = 'she is a gay , dd'.split()
    # hyp = '它 是 一个 坏人'.split()
    # ref = '她 是 一个 大 大 好人 啊'.split()
    # cer(ref, hyp)
    # wer(hyp, ref)
    count1 = count2 =0
    editDistance_batch()
    # test_ed()
    # test_editDistance_v2()
    # tabel_d = editDistance(hyp, ref)
    # print(getStepList(hyp, ref, tabel_d))
    # print(info_edit(hyp, ref, tabel_d))
    # alignedPrint(getStepList(hyp, ref, tabel_d), hyp, ref, 0.9)
