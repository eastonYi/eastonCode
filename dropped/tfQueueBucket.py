import functools

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import input as input_py
from tensorflow.python.training import queue_runner

# pylint: disable=protected-access
_as_original_type = input_py._as_original_type
_as_tensor_list = input_py._as_tensor_list
_restore_sparse_tensors = input_py._restore_sparse_tensors
_dtypes = input_py._dtypes
_store_sparse_tensors = input_py._store_sparse_tensors
_validate_keep_input = input_py._validate_keep_input
_shapes = input_py._shapes
_smart_cond = input_py._smart_cond
_which_queue = input_py._which_queue

# pylint: enable=protected-access


def _validate_bucket(tensor_list):
    tensor_list = ops.convert_n_to_tensor_or_indexed_slices(tensor_list)
    if not tensor_list:
        raise ValueError("Expected at least one tensor in bucket().")
    return tensor_list


def bucket(tensors,
           which_bucket,
           batch_size,
           num_buckets,
           num_threads=1,
           capacity=32,
           bucket_capacities=None,
           shapes=None,
           dynamic_pad=False,
           allow_smaller_final_batch=False,
           keep_input=True,
           shared_name=None,
           name=None):

    batch_size_per_bucket = False
    if isinstance(batch_size, (list, tuple)):
        batch_size_per_bucket = True
        if len(batch_size) != num_buckets:
            raise ValueError("If batch_size is a list it must have num_buckets elements")
    else:
        batch_size = [batch_size] * num_buckets

    if bucket_capacities is None:
        bucket_capacities = [capacity] * num_buckets
    if len(bucket_capacities) != num_buckets:
        raise ValueError(
            "The list bucket_capacities (%s) must have exactly num_buckets (%d) "
            "elements." % (str(bucket_capacities), num_buckets))

    tensor_list = _as_tensor_list(tensors)
    with ops.name_scope(name, "bucket", tensor_list) as name:
        tensor_list = _validate_bucket(tensor_list)
        keep_input = _validate_keep_input(keep_input,
                                          enqueue_many=False)
        (tensor_list, sparse_info) = _store_sparse_tensors(tensor_list,
                                                           enqueue_many=False,
                                                           keep_input=keep_input)

        # Round-trip batch_size to a tensor, and possibly back
        for i, bucket_batch_size in enumerate(batch_size):
            bucket_batch_size = ops.convert_to_tensor(bucket_batch_size,
                                                      dtype=dtypes.int32,
                                                      name="batch_size")
            static_batch_size = tensor_util.constant_value(bucket_batch_size)
            batch_size[i] = (static_batch_size if static_batch_size is not None else bucket_batch_size)

        types = _dtypes([tensor_list])
        shapes = _shapes([tensor_list], shapes, enqueue_many=False)

        which_bucket = ops.convert_to_tensor(
            which_bucket, dtype=dtypes.int32, name="which_bucket")

        queue_creator = _which_queue(dynamic_pad)
        bucket_queues = []
        for i in range(num_buckets):
            shared_name_i = ("%s_%d" % (shared_name, i) if shared_name is not None else None)
            bucket_queues.append(queue_creator(capacity=bucket_capacities[i],
                                               dtypes=types,
                                               shapes=shapes,
                                               shared_name=shared_name_i,
                                               name="bucket_queue_%d" % i))

        maybe_static_batch_size = (
            None if (allow_smaller_final_batch or batch_size_per_bucket)
            else static_batch_size)

        bucket_shapes = [
            tensor_shape.vector(maybe_static_batch_size).concatenate(s)
            for s in bucket_queues[0].shapes
        ]
        # top_queue is a PaddingFIFOQueue even if the bucket queues are regular FIFO
        # queues because if we use allow_smaller_final_batch, shapes will
        # contain Nones in their first entry; as a result, a regular
        # FIFOQueue would die when being passed shapes that are not fully defined.
        top_queue = data_flow_ops.PaddingFIFOQueue(
            capacity=capacity,
            dtypes=[dtypes.int32] + types,
            shapes=[tensor_shape.scalar()] + bucket_shapes,
            shared_name=shared_name,
            name="top_queue")

        def enqueue_which():
            """Return an op that enqueues conditionally in one of the queues."""
            def enqueue_single(i):
                return bucket_queues[i].enqueue(tensor_list)

            enqueues = [control_flow_ops.cond(math_ops.equal(which_bucket, i),
                                              functools.partial(enqueue_single, i),
                                              control_flow_ops.no_op) for i in range(num_buckets)]
            return control_flow_ops.group(*enqueues, name="group_enqueues")

        maybe_enqueue = _smart_cond(
            keep_input,
            enqueue_which,
            control_flow_ops.no_op)

        bucket_enqueue_ops = [maybe_enqueue] * num_threads

        if allow_smaller_final_batch:
            which_dequeue = lambda q: q.dequeue_up_to
        else:
            which_dequeue = lambda q: q.dequeue_many

        def make_list(t):
            if isinstance(t, (list, tuple)):
                return t
            else:
                return [t]

        enqueues_to_top = [
            top_queue.enqueue(
                [constant_op.constant(i)] + make_list(which_dequeue(q)(
                    bs, name="read_bucket_%d" % i)),
                name="enqueue_from_bucket_%d" % i)
            for i, (q, bs) in enumerate(zip(bucket_queues, batch_size))
        ]

        for i, q in enumerate(bucket_queues):
            queue_runner.add_queue_runner(
                queue_runner.QueueRunner(q, [enqueues_to_top[i]]))
        queue_runner.add_queue_runner(
            queue_runner.QueueRunner(top_queue, bucket_enqueue_ops))

        dequeued = top_queue.dequeue(name="dequeue_top")
        which_bucket_dequeued = dequeued[0]
        dequeued = dequeued[1:]
        if len(dequeued) == 1:
            dequeued = dequeued[0]
        dequeued = _restore_sparse_tensors(dequeued, sparse_info)
        return (which_bucket_dequeued, _as_original_type(tensors, dequeued), top_queue, bucket_queues)


def bucket_by_sequence_length(input_length,
                              tensors,
                              batch_size,
                              bucket_boundaries,
                              num_threads=1,
                              capacity=32,
                              bucket_capacities=None,
                              shapes=None,
                              dynamic_pad=False,
                              allow_smaller_final_batch=False,
                              keep_input=True,
                              shared_name=None,
                              name=None):
    tensor_list = _as_tensor_list(tensors)
    if not isinstance(bucket_boundaries, (list, tuple)):
        raise TypeError("bucket_boundaries must be a list or tuple, but received: %s" % bucket_boundaries)
    if not bucket_boundaries:
        raise ValueError("bucket_boundaries must not be empty")
    for (s, e) in zip(bucket_boundaries[:-1], bucket_boundaries[1:]):
        if not isinstance(s, int) or not isinstance(e, int):
            raise TypeError("bucket boundaries must be integers, but saw: %s and %s" %
                  (s, e))
        if s >= e:
            raise ValueError("Buckets must contain sequential increasing lengths, but saw: %d before %d" % (s, e))

    with ops.name_scope(name, "bucket_by_sequence_length", [input_length] + tensor_list) as name:
        input_length = ops.convert_to_tensor(input_length, dtype=dtypes.int32, name="input_length")
        buckets_min = [np.iinfo(np.int32).min] + list(bucket_boundaries)
        buckets_max = list(bucket_boundaries) + [np.iinfo(np.int32).max]
        conditions_c = math_ops.logical_and(
            math_ops.less_equal(buckets_min, input_length),
            math_ops.less(input_length, buckets_max))
        which_bucket = math_ops.reduce_min(array_ops.where(conditions_c))
        which_bucket = math_ops.to_int32(which_bucket)

        if shapes is not None:
            shapes = [tensor_shape.scalar()] + shapes

        _, dequeued, top_queue, bucket_queues = bucket(tensors=[input_length] + tensor_list,
                                                         which_bucket=which_bucket,
                                                         batch_size=batch_size,
                                                         num_buckets=len(bucket_boundaries) + 1,
                                                         num_threads=num_threads,
                                                         capacity=capacity,
                                                         bucket_capacities=bucket_capacities,
                                                         shapes=shapes,
                                                         dynamic_pad=dynamic_pad,
                                                         allow_smaller_final_batch=allow_smaller_final_batch,
                                                         keep_input=keep_input,
                                                         shared_name=shared_name)

        return (dequeued[0], _as_original_type(tensors, dequeued[1:]), top_queue, bucket_queues)


__all__ = ["bucket", "bucket_by_sequence_length"]
