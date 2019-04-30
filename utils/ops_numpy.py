import numpy as np


def pad_to(array, axis, pad_length):
    """
    array: [batch, time, size]
    an axis pad to specific length
    """
    if array.shape[axis] >= pad_length:
        slc = [slice(None)] * len(array.shape)
        slc[axis] = slice(0, pad_length)
        array_new = array[slc]
    else:
        npad = [(0, 0)] * len(array.shape)
        npad[axis] = (0, pad_length-array.shape[axis])
        array_new = np.pad(array, pad_width=npad, mode='constant', constant_values=0)

    return array_new

def testPad_to():
    a = np.arange(24, dtype=np.float32).reshape([2,3,4])
    print(pad_to(a, 2, 5))

if __name__ == '__main__':
    testPad_to()
