import struct
# import pickle


def store_2d(array, fw):
    fw.write(struct.pack('I', len(array)))
    for i, distrib in enumerate(array):
        for p in distrib:
            p = struct.pack('f', p)
            fw.write(p)
