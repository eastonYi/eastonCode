# Python multithreading example to print current date.
# 1. Define a subclass using Thread class.
# 2. Instantiate the subclass and trigger the thread.

import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


class SafeIter:
    """Takes an iterator/generator and return a thread-safe iterator by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        # it means itself is exactly an iterator
        return self

    def __next__(self):
        # a iterator has the next() method
        with self.lock:
            return next(self.it)

def testSafeIter():
    a = iter([1,2,3,4,5])
    m = SafeIter(a)
    import pdb; pdb.set_trace()
    print('here')

def simple_create_threads():
    """Create a thead to run a function. the main thread will stop until all the
    threads return from its function.
    """
    def sing():
        [print('singing...') for _ in range(3)]

    thread1 = threading.Thread(target=sing)
    thread2 = threading.Thread(target=sing)

    thread1.start()
    thread2.start()

    # the main thread will stop until all the theads has finish its func
    print('current numbers of threads is: %d' % len(threading.enumerate()))

    thread1.join()
    thread2.join()


def multithreading(func, args, workers):
    with ThreadPoolExecutor(workers) as ex:
        res = ex.map(func, args)
    return list(res)


def multiprocessing(func, args, workers):
    with ProcessPoolExecutor(workers) as ex:
        res = ex.map(func, args)
    return list(res)


class MyThread(threading.Thread):
    def __init__(self, sec):
        super().__init__()
        self.sec = sec

    def run(self):
        for i in range(3):
            time.sleep(self.sec)
            msg = "I'm "+self.name+' @ '+str(i)
            print(msg)


def test_thread_class():
    list_thread = []
    for i in range(5):
        t = MyThread(i)
        list_thread.append(t)
        t.start()

    for thread in list_thread:
        thread.join()
    print('here')


g_num = 0


def test_thread_lock():
    def test1():
        global g_num
        for i in range(1000000):
            mutexFlag = mutex.acquire(True)
            if mutexFlag:
                g_num += 1
                mutex.release()
        print("---test1---g_num=%d" % g_num)

    def test2():
        global g_num
        for i in range(1000000):
            mutexFlag = mutex.acquire(True)
            if mutexFlag:
                g_num += 1
                mutex.release()

        print("---test2---g_num=%d" % g_num)

    mutex = threading.Lock()

    p1 = threading.Thread(target=test1)
    p1.start()

    p2 = threading.Thread(target=test2)
    p2.start()


class myThread(threading.Thread):
    def __init__(self, name, counter, threadLock):
        threading.Thread.__init__(self)
        self.threadID = counter
        self.name = name
        self.counter = counter
        self.threadLock = threadLock

    def run(self):
        """The necessary func in the Thread class
        """
        print("Starting " + self.name)
        # Acquire lock to synchronize thread
        self.threadLock.acquire()
        print(self.name, self.counter)
        # Release lock for the next thread
        self.threadLock.release()
        # euqals to:
        # with self.threadLock:
        #     print(self.name, self.counter)
        print("Exiting " + self.name)


def test_synchronizing_threads():
    threadLock = threading.Lock()
    threads = []

    # Create new threads
    thread1 = myThread("Thread", 1, threadLock)
    thread2 = myThread("Thread", 2, threadLock)

    # Start new Threads
    thread1.start()
    thread2.start()

    # Add threads to thread list
    threads.append(thread1)
    threads.append(thread2)

    # Wait for all threads to complete
    for t in threads:
        t.join()
    print("Exiting the Program!!!")


def test_multiprocessing():
    """http://chriskiehl.com/article/parallelism-in-one-line/
    multi-processing but bot suitable for co-worker where different thread doing
    threads created at different times.
    """
    import time
    from urllib.request import urlopen
    from multiprocessing.dummy import Pool as ThreadPool

    urls = [
      'http://www.python.org',
      'http://www.python.org/about/',
      'http://www.onlamp.com/pub/a/python/2003/04/17/metaclasses.html',
      'http://www.python.org/doc/',
      'http://www.python.org/download/',
      'http://www.python.org/getit/',
      'http://www.python.org/community/',
      'https://wiki.python.org/moin/',
      'http://planet.python.org/',
      'https://wiki.python.org/moin/LocalUserGroups',
      'http://www.python.org/psf/',
      'http://docs.python.org/devguide/',
      'http://www.python.org/community/awards/'
      'http://www.python.org',
      'http://www.python.org/about/',
      'http://www.onlamp.com/pub/a/python/2003/04/17/metaclasses.html',
      'http://www.python.org/doc/',
      'http://www.python.org/download/',
      'http://www.python.org/getit/',
      'http://www.python.org/community/',
      'https://wiki.python.org/moin/',
      'http://planet.python.org/',
      'https://wiki.python.org/moin/LocalUserGroups',
      'http://www.python.org/psf/',
      'http://docs.python.org/devguide/',
      'http://www.python.org/community/awards/'
      # etc..
      ]

    start_time = time.time()
    # Make the Pool of workers
    pool = ThreadPool(2)
    # Open the urls in their own threads
    # and return the results
    try:
        pool.map(urlopen, urls)
    except:
        pass
    print('here')
    # close the pool and wait for the work to finish
    pool.close()
    pool.join()
    print(time.time() - start_time)


if __name__ == '__main__':
    # test_create_threads()
    # test_synchronizing_threads()
    # test_multiprocessing()
    # test_thread_class()
    testSafeIter()
