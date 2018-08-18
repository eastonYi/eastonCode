import threading
import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset


class DatasetPytorch(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        # it means self is exactly a iterator
        return self

    def next(self):
        # a iterator has the next() method
        with self.lock:
            return self.it.next()


def get_path_i(paths_count):
    """Cyclic generator of paths indice
    """
    current_path_id = 0
    while True:
        yield current_path_id
        current_path_id = (current_path_id + 1) % paths_count


class InputGen:
    def __init__(self, paths, batch_size):
        self.paths = paths
        self.index = 0
        self.batch_size = batch_size
        self.init_count = 0
        self.lock = threading.Lock()  #mutex for input path
        self.yield_lock = threading.Lock()  #mutex for generator yielding of batch
        self.path_id_generator = threadsafe_iter(get_path_i(len(self.paths)))
        self.images = []
        self.labels = []

    def get_samples_count(self):
        """ Returns the total number of images needed to train an epoch """
        return len(self.paths)

    def get_batches_count(self):
        """ Returns the total number of batches needed to train an epoch """
        return int(self.get_samples_count() / self.batch_size)

    def pre_process_input(self, im, lb):
        """ Do your pre-processing here
                Need to be thread-safe function"""
        return im, lb

    def next(self):
        return self.__iter__()

    def __iter__(self):
        while True:
            #In the start of each epoch we shuffle the data paths
            with self.lock:
                # the shuffle only need to be done with one thread. other threads will pass the if statement
                if (self.init_count == 0):
                    random.shuffle(self.paths)
                    self.images, self.labels, self.batch_paths = [], [], []
                    self.init_count = 1
            #Iterates through the input paths in a thread-safe manner
            for path_id in self.path_id_generator:
                img, label = self.paths[path_id]
                img = cv2.imread(img, 1)
                label_img = cv2.imread(label, 1)
                img, label = self.pre_process_input(img, label_img)
                #Concurrent access by multiple threads to the lists below
                # threads filling the lists together obeying the lock mechanism
                with self.yield_lock:
                    if (len(self.images)) < self.batch_size:
                        self.images.append(img)
                        self.labels.append(label)
                    if len(self.images) % self.batch_size == 0:
                        yield np.float32(self.images), np.float32(self.labels)
                        self.images, self.labels = [], []
            #At the end of an epoch we re-init data-structures
            with self.lock:
                self.init_count = 0

    def __call__(self):
        return self.__iter__()


class thread_killer(object):
    """Boolean object for signaling a worker thread to terminate
    """
    def __init__(self):
        self.to_kill = False

    def __call__(self):
        return self.to_kill

    def set_tokill(self, tokill):
        self.to_kill = tokill


def threaded_batches_feeder(tokill, batches_queue, dataset_generator):
    """Threaded worker for pre-processing input data.
    tokill is a thread_killer object that indicates whether a thread should be terminated
    dataset_generator is the training/validation dataset generator
    batches_queue is a limited size thread-safe Queue instance.
    """
    while not tokill():
        for batch, (batch_images, batch_labels) in enumerate(dataset_generator):
        #We fill the queue with new fetched batch until we reach the maxsize.
            batches_queue.put((batch, (batch_images, batch_labels)), block=True)
            if tokill():
                return


def threaded_cuda_batches(tokill, cuda_batches_queue, batches_queue):
    """Thread worker for transferring pytorch tensors into
    GPU.
    batches_queue is the queue that fetches numpy cpu tensors.
    cuda_batches_queue receives numpy cpu tensors and transfers them to GPU space.
    """
    while not tokill():
        batch, (batch_images, batch_labels) = batches_queue.get(block=True)
        batch_images_np = np.transpose(batch_images, (0, 3, 1, 2))
        batch_images = torch.from_numpy(batch_images_np)
        batch_labels = torch.from_numpy(batch_labels)

        batch_images = Variable(batch_images).cuda()
        batch_labels = Variable(batch_labels).cuda()
        cuda_batches_queue.put((batch, (batch_images, batch_labels)), block=True)
        if tokill():
            return


if __name__ == '__main__':
    """
    - Init of model on the GPU
    - Init of two queues:
        – Input images queue: responsible for acquiring up-to 12 pre-processed
        input images along program execution lifetime in 4 different threads
    - Training loop, where we fetch in 0’s an input images batch and feed it to
        our “PytorchNetwork.train_batch” method for accomplishing an optimization step.
    - Cuda images queue: responsible for transferring input images from the
        “input images queue” to the GPU memory space in 1 different thread.
    - Resource termination, where we signal all threads to be terminated.
    """
    import time
    import Thread
    import sys
    from Queue import Empty, Full, Queue
    import torch
    from torch.autograd import Variable

    num_epoches = 1000
    batches_per_epoch = 64
    #Training set list suppose to be a list of full-paths for all
    #the training images.
    training_set_list = None
    #Our train batches queue can hold at max 12 batches at any given time.
    #Once the queue is filled the queue is locked.
    train_batches_queue = Queue(maxsize=12)
    #Our numpy batches cuda transferer queue.
    #Once the queue is filled the queue is locked
    #We set maxsize to 3 due to GPU memory size limitations
    cuda_batches_queue = Queue(maxsize=3)

    training_set_generator = InputGen(training_set_list, batches_per_epoch)
    train_thread_killer = thread_killer()
    train_thread_killer.set_tokill(False)
    preprocess_workers = 4

    #We launch 4 threads to do load & pre-process the input images
    for _ in range(preprocess_workers):
        t = Thread(target=threaded_batches_feeder,
                   args=(train_thread_killer, train_batches_queue, training_set_generator))
        t.start()
    cuda_transfers_thread_killer = thread_killer()
    cuda_transfers_thread_killer.set_tokill(False)
    cudathread = Thread(target=threaded_cuda_batches,
                   args=(cuda_transfers_thread_killer, cuda_batches_queue, train_batches_queue))
    cudathread.start()

    #We let queue to get filled before we start the training
    time.sleep(8)
    for epoch in range(num_epoches):
        for batch in range(batches_per_epoch):
            #We fetch a GPU batch in 0's due to the queue mechanism
            _, (batch_images, batcxh_labels) = cuda_batches_queue.get(block=True)

            #train batch is the method for your training step.
            #no need to pin_memory due to diminished cuda transfers using queues.
            # loss, accuracy = train_batch(batch_images, batch_labels)

    train_thread_killer.set_tokill(True)
    cuda_transfers_thread_killer.set_tokill(True)
    for _ in range(preprocess_workers):
        try:
            #Enforcing thread shutdown
            train_batches_queue.get(block=True, timeout=1)
            cuda_batches_queue.get(block=True, timeout=1)
        except Empty:
            pass
    print("Training done")
