from tensorflow.python.ops import gen_math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.framework import ops
import tensorflow as tf


def get_session(sess):
        session = sess
        while type(session).__name__ != 'Session':
            session = session._sess
        return session


def sequence_mask(lengths, maxlen=None, dtype=dtypes.bool, name=None):
    """Return a mask tensor representing the first N positions of each row.

    Example:

    ```python
    tf.sequence_mask([1, 3, 2], 5) =
    [[True, False, False, False, False],
     [True, True, True, False, False],
     [True, True, False, False, False]]
    ```

    Args:
    lengths: 1D integer tensor, all its values < maxlen.
    maxlen: scalar integer tensor, maximum length of each row. Default: use
            maximum over lengths.
    dtype: output type of the resulting tensor.
    name: name of the op.
    Returns:
    A 2D mask tensor, as shown in the example above, cast to specified dtype.

    Raises:
    ValueError: if the arguments have invalid rank.
    """
    with ops.name_scope(name, "SequenceMask", [lengths, maxlen]):
        lengths = ops.convert_to_tensor(lengths)
    # if lengths.get_shape().ndims != 1:
    #     raise ValueError("lengths must be 1D for sequence_mask")

    if maxlen is None:
        maxlen = gen_math_ops._max(lengths, [0])
    else:
        maxlen = ops.convert_to_tensor(maxlen)
    # if maxlen.get_shape().ndims != 0:
    #     raise ValueError("maxlen must be scalar for sequence_mask")

    # The basic idea is to compare a range row vector of size maxlen:
    # [0, 1, 2, 3, 4]
    # to length as a matrix with 1 column: [[1], [3], [2]].
    # Because of broadcasting on both arguments this comparison results
    # in a matrix of size (len(lengths), maxlen)
    row_vector = gen_math_ops._range(constant(0, maxlen.dtype),
                                     maxlen,
                                     constant(1, maxlen.dtype))
    # Since maxlen >= max(lengths), it is safe to use maxlen as a cast
    # authoritative type. Whenever maxlen fits into tf.int32, so do the lengths.
    matrix = gen_math_ops.cast(tf.expand_dims(lengths, 1), maxlen.dtype)
    result = row_vector < matrix

    if dtype is None or result.dtype.base_dtype == dtype.base_dtype:
        return result
    else:
        return gen_math_ops.cast(result, dtype)


def range_batch(shape, range_down=True, dtype=tf.int32):
    """
    sess.run(range_batch([2,5]))
    range_down=False:
        array([[0, 1, 2, 3, 4],
               [0, 1, 2, 3, 4]], dtype=int32)
    range_down=True:
        array([[0, 0, 0, 0, 0],
               [1, 1, 1, 1, 1]], dtype=int32)
    """
    if not range_down:
        result = tf.tile([tf.range(shape[1], dtype=dtype)],
                       [shape[0], 1])
    else:
        result = tf.tile(tf.reshape(tf.range(shape[0], dtype=dtype), (-1, 1)),
                   [1, shape[1]])
    return result


# TODO
def label_smoothing(z, cr=0.8):
    # Label smoothing
    table = tf.convert_to_tensor([[cr, 1.-cr]])

    return tf.nn.embedding_lookup(table, z)


def dense_sequence_to_sparse(sequences, sequence_lengths):
    '''convert sequence dense representations to sparse representations
    Args:
        sequences: the dense sequences as a [batch_size x max_length] tensor
        sequence_lengths: the sequence lengths as a [batch_size] vector
    Returns:
        the sparse tensor representation of the sequences

    the reverse op:
        tf.sparse_tensor_to_dense(sp_input, default_value=0, validate_indices=True, name=None)
        [[1, 0, 0, 0]
         [0, 0, 2, 0]
         [0, 0, 0, 0]]
        indices：[[0, 0], [1, 2]]
        values：[1, 2]
        dense_shape：[3, 4]
        the default value is `0`

        a_dense = tf.sparse_to_dense(
            sparse_indices=a_sparse.indices,
            output_shape=a_sparse.dense_shape,
            sparse_values=a_sparse.values,
            default_value=0)
    '''

    with tf.name_scope('dense_sequence_to_sparse'):

        #get all the non padding sequences
        indices = tf.cast(get_indices(sequence_lengths), tf.int64)

        #create the values
        values = tf.gather_nd(sequences, indices)

        #the shape
        shape = tf.cast(tf.shape(sequences), tf.int64)

        sparse = tf.SparseTensor(indices, values, shape)

    return sparse


def batch_pad(p, length, pad, direct='r'):
    """
    add the length
    Demo:
        p = tf.ones([4, 3], dtype=tf.float32)
        pad = tf.convert_to_tensor(0.0)
        sess.run(right_length_rows(p, 1, pad, direct='head'))
        array([[ 0.,  1.,  1.,  1.],
               [ 0.,  1.,  1.,  1.],
               [ 0.,  1.,  1.,  1.],
               [ 0.,  1.,  1.,  1.]], dtype=float32)
    """
    assert type(length) is int
    if direct == 'head':
        res = tf.concat([tf.fill(dims=[tf.shape(p)[0], length], value=pad), p],
                        axis=1)
    elif direct == 'tail':
        res = tf.concat([p, tf.fill(dims=[tf.shape(p)[0], length], value=pad)],
                        axis=1)
    else:
        raise NotImplementedError

    return res


def right_shift_rows(p, shift, pad):
    assert type(shift) is int

    return tf.concat([tf.fill(dims=[tf.shape(p)[0], 1], value=pad), p[:, :-shift]], axis=1)


def get_indices(sequence_length):
    '''get the indices corresponding to sequences (and not padding)
    Args:
        sequence_length: the sequence_lengths as a N-D tensor
    Returns:
        A [sum(sequence_length) x N-1] Tensor containing the indices'''

    with tf.name_scope('get_indices'):

        numdims = len(sequence_length.shape)

        #get the maximal length
        max_length = tf.reduce_max(sequence_length)

        sizes = tf.shape(sequence_length)

        range_tensor = tf.range(max_length)
        for i in range(1, numdims):
            tile_dims = [1]*i + [sizes[i]]
            range_tensor = tf.tile(tf.expand_dims(range_tensor, i), tile_dims)

        indices = tf.where(tf.less(range_tensor,
                                   tf.expand_dims(sequence_length, numdims)))

    return indices
