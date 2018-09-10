"""
Description
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import threading, multiprocessing
from collections import OrderedDict

DTYPE_DICT = {'int32': tf.int32, 'int64': tf.int64, 'float32': tf.float32, 'float64': tf.float32}


class PythonFeeder(object):
    """
    This class manages the the background threads needed to fill
        a queue full of datalib.
    """
    def __init__(self, dataset, batch_size=256, n_feeders=2, queue=tf.RandomShuffleQueue):
        if isinstance(dataset, (list, tuple)) and len(dataset) == 2:
            dataset = CompleteDataGenerator(dataset[0], dataset[1], shuffle=True, batch_size=batch_size)

        self.dataset = dataset
        self.batch_size = batch_size
        self.n_threads = n_feeders
        self.num_samples = dataset.num_samples()

        self.input_info = dataset.input_info()
        self.target_info = dataset.target_info()
        self._stop = False
        self.feeders = []

        self.queue = queue
        self.inputs = None
        self.targets = None
        self.enqueue_op = None

    def request_stop(self):
        self.queue.close(cancel_pending_enqueues=True)
        self._stop = True

    def join(self):
        import time
        while all([not f.isAlive() for f in self.feeders]):
            time.sleep(0.1)

    def build_batch(self):
        """
        Return's tensors containing a batch of test_images and labels
        """

        # Data Queue
        min_after_dequeue = 20 * self.batch_size
        capacity = min_after_dequeue + 5 * self.batch_size
        names = self.input_info.keys() + self.target_info.keys()
        shapes = [val['shape'] for _, val in self.input_info.iteritems()] + \
                 [val['shape'] for _, val in self.target_info.iteritems()]
        dtypes = [DTYPE_DICT[val['dtype']] for _, val in self.input_info.iteritems()] + \
                 [DTYPE_DICT[val['dtype']] for _, val in self.target_info.iteritems()]
        if self.queue is tf.RandomShuffleQueue:
            self.queue = self.queue(names=names, shapes=shapes, dtypes=dtypes, capacity=capacity, min_after_dequeue=min_after_dequeue)
        elif self.queue is tf.FIFOQueue:
            self.queue = self.queue(names=names, shapes=shapes, dtypes=dtypes, capacity=capacity)
        else:
            raise ValueError('Unknown queue type.')

        # Placeholders for feeding the queue
        self.inputs = OrderedDict(
            [(key, tf.placeholder(dtype=DTYPE_DICT[val['dtype']],
                                  shape=(None,)+val['shape'],
                                  name=key))
             for key, val in self.input_info.iteritems()]
        )
        self.targets = OrderedDict(
            [(key, tf.placeholder(dtype=DTYPE_DICT[val['dtype']],
                                  shape=(None,)+val['shape'],
                                  name=key))
             for key, val in self.target_info.iteritems()]
        )

        # The symbolic operation to add datalib to the queue
        enqueue_dict = dict(self.inputs.items() + self.targets.items())
        self.enqueue_op = self.queue.enqueue_many(enqueue_dict)

        samples = self.queue.dequeue_many(self.batch_size)
        inputs = [samples[key] for key in self.input_info.keys()]
        targets = [samples[key] for key in self.target_info.keys()]
        return inputs, targets


    def main_feed(self, sess, n):
        """
        Function run on alternate thread. Basically, keep adding datalib to the queue.
        """
        # import time
        try:
            feed_dict = {}
            feed_dict.update({self.inputs[key]: None for key in self.input_info.keys()})
            feed_dict.update({self.targets[key]: None for key in self.target_info.keys()})
            # i = 1
            # t0 = t1 = time.time()
            for j, (inputs_, targets_) in enumerate(self.dataset.batch_loop()):
                # t1_iter = time.time()-t1
                #
                # t2 = time.time()
                if inputs_ is None:
                    break
                if self._stop:
                    print('Stop requested. Feeder %d for queue (%s) will close...' % (n, type(self.queue).__name__))
                    return

                for key in self.input_info.keys():
                    feed_dict[self.inputs[key]] = inputs_[key]

                for key in self.target_info.keys():
                    feed_dict[self.targets[key]] = targets_[key]
                # t2_iter = time.time()-t2
                #
                # t3 = time.time()

                sess.run(self.enqueue_op, feed_dict=feed_dict)

                # t3_iter = time.time()-t3
                #
                # if j<5:
                #     t0 = t1 = time.time()
                #     continue
                #
                # print('Thread %d' % n, i, feed_dict[self.targets['answer']].shape[0]*i/(time.time()-t0), t1_iter, t2_iter, t3_iter, [feed_dict[key].shape for key in feed_dict.keys()])
                # i += 1
                #
                # t1 = time.time()

        except tf.errors.CancelledError:
            print('TF queue is closed. Feeder %d for queue (%s) will close...' % (n, type(self.queue).__name__))
            return

    def start_feeder(self, sess):
        """ Start background threads to feed queue """
        self.dataset.start_feeder()
        self._stop = False
        threads = []
        for n in xrange(self.n_threads):
            thread = threading.Thread(target=self.main_feed, args=(sess, n))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        self.feeders = threads


class TFFeeder(object):
    """
    This class manages the the background threads needed to fill
        a queue full of datalib.
    """
    def __init__(self, dataset, batch_size=256, n_feeders=2, n_dequeue=1, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_feeders = n_feeders
        self.shuffle = shuffle
        self.num_samples = dataset.num_samples()
        self.n_dequeue = n_dequeue

    def build_batch(self, min_after_dequeue=5, capacity=3, num_epochs=None, gpu_id=0):
        with tf.device('/cpu:0'):
            dataset, batch_size, n_feeders, shuffle = self.dataset, self.batch_size, self.n_feeders, self.shuffle

            input_info = dataset.input_info()
            target_info = dataset.target_info()

            filename_queue = tf.train.string_input_producer(dataset.tfrecords_fns(), shuffle=shuffle, num_epochs=num_epochs)

            dequeue_size = 1./(np.arange(self.n_dequeue, dtype=float)+3)
            dequeue_size = np.ceil(dequeue_size/dequeue_size.sum()*batch_size).astype(int)
            dequeue_size[0] = batch_size - dequeue_size[1:].sum()   # CPU->GPU copy is not parelelized. By having smaller dequeue, CPU->GPU copies can start sooner.

            inputs_batch, targets_batch = [None]*self.n_dequeue, [None]*self.n_dequeue
            for i in range(self.n_dequeue):
                min_after_dequeue = min_after_dequeue * dequeue_size[i]
                capacity = min_after_dequeue + capacity * dequeue_size[i]
                if shuffle and n_feeders == 1: batch_fnc = lambda x: tf.train.shuffle_batch(x, dequeue_size[i], capacity, min_after_dequeue)
                elif shuffle and n_feeders > 1: batch_fnc = lambda x: tf.train.shuffle_batch_join(x, dequeue_size[i], capacity, min_after_dequeue)
                elif not shuffle and n_feeders == 1: batch_fnc = lambda x: tf.train.batch(x, dequeue_size[i], capacity)
                elif not shuffle and n_feeders > 1: batch_fnc = lambda x: tf.train.batch_join(x, dequeue_size[i], capacity)

                tensors = []
                for _ in range(n_feeders):
                    reader = tf.TFRecordReader()
                    _, record_serialized = reader.read(filename_queue)
                    inputs, targets = dataset.parse_and_prep_record(record_serialized)
                    tensors.append(dict([(key, inputs[key]) for key in input_info.keys()] +
                                        [(key, targets[key]) for key in target_info.keys()]))
                if n_feeders == 1:
                    tensors = tensors[0]

                tensors_batch = batch_fnc(tensors)
                inputs_batch[i] = [tensors_batch[key] for key in input_info.keys()]
                targets_batch[i] = [tensors_batch[key] for key in target_info.keys()]

        with tf.device(gpu_id):
            if self.n_dequeue>1:
                inputs_batch = [tf.concat(0, inps) for inps in zip(*inputs_batch)]
                targets_batch = [tf.concat(0, trgts) for trgts in zip(*targets_batch)]
            else:
                inputs_batch = inputs_batch[0]
                targets_batch = targets_batch[0]

        return inputs_batch, targets_batch

    def request_stop(self):
        pass

    def join(self):
        pass

    def start_feeder(self, sess):
        pass


class Dataset(object):
    def __init__(self, init=True):
        if init:
            self._input_info = OrderedDict()
            self._target_info = OrderedDict()
            self.n_samples = None

    def input_info(self):
        return self._input_info.copy()

    def target_info(self):
        return self._target_info.copy()

    def num_samples(self):
        assert self.n_samples is not None, 'Dataset has to define self.n_samples'
        return self.n_samples

    def add_input(self, name, dtype='float32', shape=()):
        assert name not in self._input_info.keys() + self._target_info.keys()
        self._input_info[name] = {'dtype': dtype, 'shape': shape}

    def add_target(self, name, dtype='int32', shape=()):
        assert name not in self._input_info.keys() + self._target_info.keys()
        self._target_info[name] = {'dtype': dtype, 'shape': shape}


class DataGenerator(Dataset):
    """
    This class provides a template for the loaders used with TFFeeder
    """
    def __init__(self, shuffle, batch_size=64, reinit_dataset=True):
        Dataset.__init__(self, init=reinit_dataset)
        self.shuffle = shuffle
        self.batch_size = batch_size

    def start_feeder(self):
        # Do nothing
        pass

    def loop(self):
        raise NotImplementedError

    def batch_loop(self):
        raise NotImplementedError

    def get_sample(self, index):
        raise NotImplementedError


class CompleteDataGenerator(DataGenerator):
    """
    Instance of DataGenerator for complete datasets
    (i.e. that can be completely loaded into numpy arrays)
    """
    def __init__(self, inputs, targets, shuffle=True, batch_size=64):
        DataGenerator.__init__(self, shuffle, batch_size)
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
        if isinstance(inputs, list):
            inputs = {'X:%d' % i: inputs[i] for i in xrange(len(inputs))}

        if isinstance(targets, np.ndarray):
            targets = [targets]
        if isinstance(targets, list):
            targets = {'Y:%d' % i: targets[i] for i in xrange(len(targets))}

        for key, val in inputs.iteritems():
            self.add_input(key, val.dtype.name, tuple(val.shape[1:]))
        for key, val in targets.iteritems():
            self.add_target(key, val.dtype.name, tuple(val.shape[1:]))

        self.inputs = inputs
        self.targets = targets
        self.n_samples = self.inputs.values()[0].shape[0]

    def loop(self):
        from itertools import cycle
        order = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(order)
        for item in cycle(order):
            inputs = {key: val[item] for key, val in self.inputs.items()}
            targets = {key: val[item] for key, val in self.targets.items()}
            yield inputs, targets

    def batch_loop(self):
        from itertools import cycle
        order = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(order)
        batch_lims = [(i, i+self.batch_size) for i in xrange(0, self.n_samples, self.batch_size)]
        for lims in cycle(batch_lims):
            inputs = {key: val[lims[0]:lims[1]] for key, val in self.inputs.items()}
            targets = {key: val[lims[0]:lims[1]] for key, val in self.targets.items()}
            yield inputs, targets

    def get_sample(self, index):
        inputs = {key: val[index] for key, val in self.inputs.items()}
        targets = {key: val[index] for key, val in self.targets.items()}
        return inputs, targets


class MultiProcessWrapper(DataGenerator):
    """
    A python class that manages a donkey and feeder to fetch python datalib samples
    Works as a wrapper for a DataGenerator
    """
    def __init__(self, dataset, n_donkey=4):
        DataGenerator.__init__(self, dataset.shuffle, dataset.batch_size)
        self.dataset = dataset
        self.n_donkey = n_donkey
        self.n_samples = dataset.num_samples()
        self._input_info = dataset.input_info()
        self._target_info = dataset.target_info()

        self.donkey = Donkey(f=self.get_sample, n_proc=n_donkey)
        self._feeder = None

    def _random_ind_list(self):
        if self.shuffle:
            ind_list = np.random.permutation(self.n_samples)
        else:
            ind_list = xrange(self.n_samples)
        return ind_list

    def start_feeder(self, coord=None):
        '''
        a infinite feeder which generates job list every epoch
        then submit job to donkeys
        :return:
        '''
        def feed(coord=None):
            while True:
                self.ind_list = self._random_ind_list()
                for ind in self.ind_list:
                    if coord is not None:
                        if coord.should_stop():
                            break
                    self.donkey.add_job((ind, ))

        self._feeder = threading.Thread(target=feed, args=(coord,))
        self._feeder.daemon = True
        self._feeder.start()
        return self._feeder

    def loop(self):
        '''
        a infite loop which retrieves samples from donkeys
        '''
        while True:
            yield self.donkey.q_out.get()

    def batch_loop(self):
        '''
        a infite loop which accumulates samples from loop() to form batches
        '''
        inputs, targets, n = {}, {}, 0
        for inp, trg in self.loop():
            # Memory initialization
            for key in self._input_info.keys():
                if key not in inputs.keys():
                    inputs[key] = np.zeros((self.batch_size,)+inp[key].shape, dtype=inp[key].dtype)
                inputs[key][n] = inp[key]

            for key in self._target_info.keys():
                if key not in targets.keys():
                    targets[key] = np.zeros((self.batch_size,)+trg[key].shape, dtype=trg[key].dtype)
                targets[key][n] = trg[key]

            n += 1
            if n == self.batch_size:
                yield inputs, targets
                n = 0

    def get_sample(self, idx):
        return self.dataset.get_sample(idx)


# Auxiliary donkey function
def fun(f, var_dict, q_in, q_out, coord=None):
    while True:
        if coord is not None:
            if coord.should_stop():
                break
        x = q_in.get()
        if x is None:
            break
        if var_dict is not None:
            if not isinstance(x, tuple):
                x = (x, )
            x = (var_dict, ) + x
        res = f(*x)
        q_out.put(res)


class Donkey(object):
    def __init__(self, n_proc=8, f=None, var_dict=None):
        self.q_in = multiprocessing.Queue(1)
        self.q_out = multiprocessing.Queue(n_proc*25)
        self.n_proc = n_proc
        self.f = f
        assert (isinstance(var_dict, (list,tuple)) and len(var_dict)==n_proc) or isinstance(var_dict, dict)
        if not isinstance(var_dict, (list,tuple)):
            self.var_dict = [var_dict for _ in range(n_proc)]

        self.workers = [multiprocessing.Process(target=fun, args=(self.f, self.var_dict[i], self.q_in, self.q_out)) for i in xrange(n_proc)]

        for p in self.workers:
            p.daemon = True
            p.start()

    def add_job(self, args):
        self.q_in.put(args)

    def stop(self):
        [self.q_in.put((None)) for _ in range(self.n_proc)]
        [p.join() for p in self.workers]


def test_enqueue():
    from datalib.daquar.TFDataGenerator import DaquarDtGen
    import time

    daquar = DaquarDtGen(full=True, single_answer=False, features={'cnn': 'VGG19', 'layer': 'fc7'}, train=True, batch_size=128)
    #daquar = MultiProcessWrapper(dataset=daquar, n_donkey=8)

    tf_feeder = PythonFeeder(dataset=daquar, batch_size=512, n_feeders=2, queue=tf.RandomShuffleQueue)

    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        tf_feeder.start_feeder(sess=sess)

        a = 0
        while True:
            a += np.random.rand()
            pass

        tf_feeder.request_stop()
        tf_feeder.join()
        coord.request_stop()
        coord.join(threads)


def test_donkey():
    def do_job():
        return np.zeros((1, 120, 1024))

    import time
    donkey = Donkey(10, do_job)

    t1 = time.time()
    for i in range(4):
        t = time.time()
        donkey.add_job()
        print('addjob', time.time()-t)

    for i in range(4):
        print(donkey.q_out.get().shape)
    print(time.time()-t1)


if __name__ == '__main__':
    # test_donkey()
    test_enqueue()
    # test_batcher()
