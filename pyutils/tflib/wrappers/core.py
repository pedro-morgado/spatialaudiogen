import tensorflow as tf
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib import layers
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.contrib.layers import batch_norm as _bn
import numpy as np


def add_bias(x, n_units, biases_initializer, dtype, trainable):
    # Initializer
    biases_shape = [n_units]
    if biases_initializer is None:
        biases_initializer = tf.constant_initializer(0.0, dtype=tf.float32)
    elif isinstance(biases_initializer, np.ndarray):
        if biases_initializer.ndim != 1 or biases_initializer.shape[0] != biases_shape[0]:
            raise ValueError('Shape of constant initializer ('+str(biases_initializer.shape)+') does not match expected shape ('+str(biases_shape)+'). ')
        biases_shape = None    # Shape is inferred from initializer

    # Create variable for bias
    biases = variables.model_variable('biases',
                                      shape=biases_shape,
                                      dtype=dtype,
                                      initializer=biases_initializer,
                                      trainable=trainable)

    # Add bias
    return tf.nn.bias_add(x, biases)


def var_initializer(shape, initializer=None):
    if initializer is None:
        # initializer = tf.truncated_normal_initializer(stddev=0.001)
        initializer = tf.contrib.layers.xavier_initializer()
    elif isinstance(initializer, np.ndarray):
        if any([s is None for s in shape]): raise ValueError('All kernel dimensions must be known.')
        if initializer.ndim != len(shape) or not all([s1==s2 for s1, s2 in zip(initializer.shape, shape)]):
            raise ValueError('Shape of constant initializer ('+str(initializer.shape)+') does not match expected shape ('+str(shape)+'). ')
        shape = None    # Shape is inferred from initializer
    return shape, initializer


def fully_connected(x, n_units,
                    use_bias=True,
                    use_batch_norm=False,
                    activation_fn=tf.nn.relu,
                    weight_decay=0.0005,
                    trainable=True,
                    reuse=False,
                    is_training=None,
                    weights_initializer=None,
                    biases_initializer=None,
                    name='fc'):
    """Wrapper for fully connected layer."""

    with tf.variable_scope(name, 'fully_connected', [x], reuse=reuse) as sc:
        dtype = x.dtype.base_dtype
        input_rank = x.get_shape().ndims
        if input_rank is None: raise ValueError('Rank of inputs must be known')
        if input_rank < 2: raise ValueError('Rank of inputs is %d, which is < 2' % input_rank)

        inputs_shape = x.get_shape()
        inp_size = utils.last_dimension(inputs_shape, min_rank=2)

        static_shape = inputs_shape.as_list()

        weights_shape, weights_initializer = var_initializer([inp_size, n_units], weights_initializer)
        weights_regularizer = l2_regularizer(weight_decay) if weight_decay > 0 and trainable else None
        weights = variables.model_variable('weights',
                                           shape=weights_shape,
                                           dtype=dtype,
                                           initializer=weights_initializer,
                                           regularizer=weights_regularizer,
                                           trainable=trainable)

        if len(static_shape) > 2:
            # Reshape inputs
            x = tf.reshape(x, [-1, inp_size])
        x = tf.matmul(x, weights)

        if use_batch_norm:
            x = _bn(x, decay=0.99, scale=True, is_training=is_training, trainable=trainable, reuse=reuse, scope='bn')
        elif use_bias:
            x = add_bias(x, n_units, biases_initializer, dtype, trainable)

        if activation_fn is not None:
            x = activation_fn(x)

        if len(static_shape) > 2:
            # Reshape back outputs
            x = tf.reshape(x, static_shape[:-1]+[-1,])
            # x.set_shape(static_shape)
        return utils.collect_named_outputs(tf.GraphKeys.ACTIVATIONS, sc.original_name_scope, x)


def deconv_2d(x, n_units, kernel_size,
              stride=1,
              use_bias=True,
              padding="SAME",
              activation_fn=tf.nn.relu,
              weight_decay=0.0005,
              trainable=True,
              reuse=None,
              weights_initializer=None,
              biases_initializer=None,
              name='deconv2d'):
    """Deconvolution wrapper."""
    with tf.variable_scope(name, 'Deconv2D', [x], reuse=reuse) as sc:
        dtype = x.dtype.base_dtype
        input_rank = x.get_shape().ndims

        if input_rank is None: raise ValueError('Rank of inputs must be known')
        if input_rank < 3: raise ValueError('Rank of inputs is %d, which is < 3' % input_rank)
        if input_rank == 3:
            x = tf.expand_dims(x, 3)

        kernel_size = utils.n_positive_integers(2, kernel_size)
        w_shape = list(kernel_size) + [n_units, x.get_shape().as_list()[-1]]
        if len(w_shape) < input_rank:
            w_shape = [1] * (input_rank - len(w_shape)) + w_shape

        # print w_shape

        # Create variable for kernel
        w_shape, weights_initializer = var_initializer(w_shape, weights_initializer)
        weights_regularizer = l2_regularizer(weight_decay) if weight_decay > 0 and trainable else None
        weights = variables.model_variable('weights',
                                           shape=w_shape,
                                           dtype=dtype,
                                           initializer=weights_initializer,
                                           regularizer=weights_regularizer,
                                           trainable=trainable)
        # print weights
        # print ' * {:15s} | {:20s} | {:10s}'.format(name+' W', str(weights.get_shape()), str(weights.dtype))

        # Deconvolution
        sz = x.get_shape().as_list()
        stide = utils.n_positive_integers(2, stride)
        output_shape = (sz[0], sz[1]*stride[0]+kernel_size[0]-stride[0], sz[2]*stride[1]+kernel_size[1]-stride[1], n_units)
        x = tf.nn.conv2d_transpose(x, weights, output_shape, strides=[1, stride[0], stride[1], 1], padding=padding)
        # print x

        # Bias
        if use_bias:
            x = add_bias(x, n_units, biases_initializer, dtype, trainable)
            # print x

        # Activation
        if activation_fn is not None:
            x = activation_fn(x)
            # print x

        return utils.collect_named_outputs(tf.GraphKeys.ACTIVATIONS, sc.original_name_scope, x)


def conv_2d(x, n_units, kernel_size,
            stride=1,
            dilation=None,
            padding="SAME",
            use_bias=True,
            use_batch_norm=False,
            activation_fn=tf.nn.relu,
            weight_decay=0.0005,
            trainable=True,
            reuse=None,
            is_training=None,
            weights_initializer=None,
            biases_initializer=None,
            bn_initializer=None,
            name='conv2d'):
    """Convolution wrapper."""

    with tf.variable_scope(name, 'Conv2D', [x], reuse=reuse) as sc:
        dtype = x.dtype.base_dtype
        input_rank = x.get_shape().ndims

        if input_rank is None: raise ValueError('Rank of inputs must be known')
        if input_rank < 3: raise ValueError('Rank of inputs is %d, which is < 3' % input_rank)
        if input_rank == 3:
            x = tf.expand_dims(x, 3)

        # Kernel dimensions
        kernel_size = utils.n_positive_integers(2, kernel_size)
        w_shape = list(kernel_size) + [x.get_shape().as_list()[-1], n_units]
        if len(w_shape) < input_rank:
            w_shape = [1]*(input_rank-len(w_shape)) + w_shape

        # Create variable for kernel
        w_shape, weights_initializer = var_initializer(w_shape, weights_initializer)
        weights_regularizer = l2_regularizer(weight_decay) if weight_decay > 0 and trainable else None
        weights = variables.model_variable('weights',
                                           shape=w_shape,
                                           dtype=dtype,
                                           initializer=weights_initializer,
                                           regularizer=weights_regularizer,
                                           trainable=trainable)

        # Convolution
        stride = utils.n_positive_integers(2, stride)
        if len(stride) < input_rank-2:
            stride = (1,)*(input_rank-len(stride)-2) + stride
        if dilation is not None:
            dilation = utils.n_positive_integers(2, dilation)
            if len(dilation) < input_rank-2:
                dilation = (1,)*(input_rank-len(dilation)-2) + dilation
        x = tf.nn.convolution(input=x, filter=weights, strides=stride, dilation_rate=dilation, padding=padding)

        # Batch normalization
        if use_batch_norm:
            x = _bn(x, decay=0.99, scale=True, param_initializers=bn_initializer, is_training=is_training, trainable=trainable, reuse=reuse, scope='bn')

        # Bias
        elif use_bias:
            x = add_bias(x, n_units, biases_initializer, dtype, trainable)

        # Activation
        if activation_fn is not None:
            x = activation_fn(x)

        return utils.collect_named_outputs(tf.GraphKeys.ACTIVATIONS, sc.original_name_scope, x)


def conv_1d(x, n_units, kernel_size,
            stride=1,
            dilation=None,
            padding="SAME",
            use_bias=True,
            use_batch_norm=False,
            activation_fn=None,
            weight_decay=0.0005,
            trainable=True,
            reuse=None,
            is_training=None,
            weights_initializer=None,
            biases_initializer=None,
            name='conv1d'):
    """Wrapper for 1d convolutional layer."""

    with tf.variable_scope(name, 'Conv1D', [x], reuse=reuse) as sc:
        input_rank = x.get_shape().ndims

        if input_rank is None: raise ValueError('Rank of inputs must be known')
        if input_rank not in [2, 3]: raise ValueError('Rank of inputs is %d, which is not 2 or 3' % input_rank)
        if input_rank == 2:
            x = tf.expand_dims(x, 2)

        if dilation is not None:
            dilation = [1, dilation]

        x = tf.expand_dims(x, axis=1)
        x = conv_2d(x, n_units,
                    kernel_size=[1, kernel_size],
                    stride=[1, stride],
                    dilation=dilation,
                    padding=padding,
                    use_bias=use_bias,
                    use_batch_norm=use_batch_norm,
                    activation_fn=activation_fn,
                    weight_decay=weight_decay,
                    trainable=trainable,
                    reuse=reuse,
                    is_training=is_training,
                    weights_initializer=weights_initializer,
                    biases_initializer=biases_initializer)
        x = tf.squeeze(x, axis=1, name=name)
        return utils.collect_named_outputs(tf.GraphKeys.ACTIVATIONS, sc.original_name_scope, x)


def causal_conv1d(x, n_units, kernel_size,
                  axis=1,
                  stride=1,
                  dilation=1,
                  use_bias=True,
                  use_batch_norm=False,
                  activation_fn=None,
                  weight_decay=0.0005,
                  trainable=True,
                  reuse=None,
                  is_training=None,
                  weights_initializer=None,
                  biases_initializer=None,
                  bn_initializer=None,
                  name='CausalConv1D'):

    with tf.variable_scope(name, 'CausalConv1D', [x], reuse=reuse) as sc:
        dtype = x.dtype.base_dtype
        input_rank = x.get_shape().ndims
        if input_rank is None: raise ValueError('Rank of inputs must be known')
        if input_rank < 2: raise ValueError('Rank of inputs is %d, which is < 2' % input_rank)
        if input_rank == 2:
            x = tf.expand_dims(x, 2)

        input_rank = x.get_shape().ndims
        n_inp_channels = x.get_shape().as_list()[-1]
        n_inp_steps = x.get_shape().as_list()[axis]

        # Kernel dimensions
        w_shape = [kernel_size] + [1]*(input_rank-3+1-axis) + [n_inp_channels, n_units]

        # Create variable for kernel
        weights_shape, weights_initializer = var_initializer(w_shape, weights_initializer)
        weights_regularizer = l2_regularizer(weight_decay) if weight_decay > 0 and trainable else None
        weights = variables.model_variable('weights',
                                           shape=w_shape,
                                           dtype=dtype,
                                           initializer=weights_initializer,
                                           regularizer=weights_regularizer,
                                           trainable=trainable)

        # Convolution
        if dilation > 1:
            dilation_rate = [1 for _ in range(x.get_shape().ndims-2)]
            dilation_rate[axis-1] = dilation
            out = tf.nn.convolution(x, weights, padding='VALID', dilation_rate=dilation_rate)
        else:
            strides = [1 for _ in range(input_rank-2)]
            strides[axis-1] = stride
            out = tf.nn.convolution(x, weights, padding='VALID', strides=strides)

        # Remove excess elements at the end.
        out_width = (n_inp_steps - (kernel_size - 1) * dilation) / stride
        x = tf.slice(out, [0]*input_rank, [-1, out_width] + [-1]*(input_rank-2))

        # Batch normalization
        if use_batch_norm:
            x = _bn(x, decay=0.99, scale=True, param_initializers=bn_initializer, is_training=is_training, trainable=trainable, reuse=reuse, scope='bn')

        # Bias
        elif use_bias:
            x = add_bias(x, n_units, biases_initializer, dtype, trainable)

        # Activation
        if activation_fn is not None:
            x = activation_fn(x)

        return utils.collect_named_outputs(tf.GraphKeys.ACTIVATIONS, sc.original_name_scope, x)


def max_pool2d(x, window, stride=1, padding='SAME', name='MaxPool'):
    input_rank = x.get_shape().ndims

    if input_rank is None: raise ValueError('Rank of inputs must be known')
    if input_rank < 3: raise ValueError('Rank of inputs is %d, which is < 3' % input_rank)
    if input_rank == 3:
        x = tf.expand_dims(x, 3)

    window = utils.n_positive_integers(2, window)
    if len(window) < input_rank-2:
        window = (1,)*(input_rank-len(window)-2) + window

    stride = utils.n_positive_integers(2, stride)
    if len(stride) < input_rank-2:
        stride = (1,)*(input_rank-len(stride)-2) + stride

    out = tf.nn.pool(x, window,'MAX', padding, strides=stride, name=name)
    return utils.collect_named_outputs(tf.GraphKeys.ACTIVATIONS, name, out)


def max_pool1d(x, kernel_size, stride=1, padding='SAME', name='MaxPool'):
    with tf.variable_scope(name, 'MaxPool1D', [x]) as sc:
        input_rank = x.get_shape().ndims

        if input_rank is None: raise ValueError('Rank of inputs must be known')
        if input_rank not in [2, 3]: raise ValueError('Rank of inputs is %d, which is not 2 or 3' % input_rank)
        if input_rank == 2:
            x = tf.expand_dims(x, 2)
        x = tf.expand_dims(x, axis=1)
        x = max_pool2d(x, [1, kernel_size], [1, stride], padding=padding, name=name)
        x = tf.squeeze(x, axis=1, name=name)
        return x


def avg_pool2d(x, kernel_size, stride=1, padding='SAME', name='AvgPool'):
    return layers.avg_pool2d(x, kernel_size, stride, padding=padding, outputs_collections=tf.GraphKeys.ACTIVATIONS, scope=name)


def avg_pool1d(x, kernel_size, stride=1, padding='SAME', name='AvgPool'):
    with tf.variable_scope(name, 'AvgPool1D', [x]) as sc:
        input_rank = x.get_shape().ndims

        if input_rank is None: raise ValueError('Rank of inputs must be known')
        if input_rank not in [2, 3]: raise ValueError('Rank of inputs is %d, which is not 2 or 3' % input_rank)
        if input_rank == 2:
            x = tf.expand_dims(x, 2)
        x = tf.expand_dims(x, axis=1)
        x = avg_pool2d(x, [1, kernel_size], [1, stride], padding=padding, name=name)
        x = tf.squeeze(x, axis=1, name=name)
        return x

def dropout(x,
            keep_prob=0.5,
            is_training=False,
            name='drop'):
    with tf.variable_scope(name, 'dropout', [x]) as sc:
        x = utils.smart_cond(is_training,
                             lambda: tf.nn.dropout(x, keep_prob, name=name),
                             lambda: x, name=name)
        return utils.collect_named_outputs(tf.GraphKeys.ACTIVATIONS, sc.original_name_scope, x)
