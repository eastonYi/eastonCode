import struct
# import pickle


def store_2d(array, fw):
    """
    fw = open('distribution.bin', 'wb', encoding='utf-8')
    array: np.array([])
    """
    fw.write(struct.pack('I', len(array)))
    for i, distrib in enumerate(array):
        for p in distrib:
            p = struct.pack('f', p)
            fw.write(p)
