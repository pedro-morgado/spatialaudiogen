import tensorflow as tf


def _rnn_cell(cell_type, num_units, num_layers=1, activation=tf.nn.tanh, keep_prob=None, is_training=None):
    if cell_type == 'rnn':
        cell_fn = tf.nn.rnn_cell.BasicRNNCell
    elif cell_type == 'gru':
        cell_fn = tf.nn.rnn_cell.GRUCell
    elif cell_type == 'lstm':
        cell_fn = tf.nn.rnn_cell.BasicLSTMCell
    else:
        raise Exception("model type not supported: {}".format(cell_type))

    if cell_type == 'lstm':
        cell = cell_fn(num_units, activation=activation, forget_bias=1.0, state_is_tuple=True)
    else:
        cell = cell_fn(num_units, activation=activation)

    # Dropout
    if keep_prob:
        if isinstance(keep_prob, float):
            keep_prob = [keep_prob, keep_prob]
        if is_training is None:
            input_keep_prob = keep_prob[0]
            output_keep_prob = keep_prob[1]
        else:
            input_keep_prob = tf.cond(is_training, lambda: tf.constant(keep_prob[0]), lambda: tf.constant(1.0))
            output_keep_prob = tf.cond(is_training, lambda: tf.constant(keep_prob[1]), lambda: tf.constant(1.0))
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob)

    # Multi-layer LSTM
    cell = tf.nn.rnn_cell.MultiRNNCell([cell]*num_layers, state_is_tuple=True) if num_layers > 1 else cell

    return cell


def _rnn(cell, x,
         activation=tf.nn.sigmoid,
         sequence_length=None,
         initial_state=None,
         return_seq=False,
         return_final_state=False):
    x = tf.unstack(x, axis=1)
    x, s = tf.nn.rnn(cell, x,
                     initial_state=initial_state,
                     dtype=tf.float32,
                     sequence_length=sequence_length)

    if not return_seq:
        x = x[-1]

    if activation:
        with tf.variable_scope('activation'):
            x = activation(x)

    if return_final_state:
        return x, s
    else:
        return x


def _check_dropout(keep_prob, is_training):
    if keep_prob is None:
        dropout = None
    else:
        if is_training is None:
            dropout = keep_prob
        #elif isinstance(is_training, tf.Tensor):
        #    dropout = tf.cond(is_training, lambda: tf.constant(keep_prob), lambda: tf.constant(1.0))
        elif is_training is True and keep_prob < 1:
            dropout = keep_prob
        else:
            dropout = None
    return dropout


def _prep_outputs(layer_outputs, return_seq=False, return_all_layers=False, return_inp_bow=False, sequence_length=None):
    if not return_all_layers:
        [[w.remove(todel) for todel in w[:-1]] for w in layer_outputs]
    else:
        if not return_inp_bow:
            [w.remove(w[0]) for w in layer_outputs]
        else:
            if sequence_length is not None:
                zeros = tf.zeros_like(layer_outputs[0][0])
                for i in range(len(layer_outputs)):
                    layer_outputs[i][0] = tf.select(tf.less(tf.cast(i, tf.int64), sequence_length), layer_outputs[i][0], zeros)
                den = tf.cast(tf.expand_dims(sequence_length,1), tf.float32)
            else:
                den = float(len(layer_outputs))
            inp_avg = tf.add_n([w[0] for w in layer_outputs])/den
            for i in range(len(layer_outputs)):
                layer_outputs[i][0] = inp_avg

    layer_outputs = [tf.concat(1, w) for w in layer_outputs]
    if return_seq:
        if sequence_length is not None:
            zeros = [tf.zeros_like(w) for w in layer_outputs]
            outputs = [tf.select(tf.less(tf.cast(i, tf.int64), sequence_length), o, z) for i, (o, z) in enumerate(zip(layer_outputs, zeros))]
        else:
            outputs = layer_outputs
    else:
        if sequence_length is None:
            outputs = layer_outputs[-1]
        else:
            outputs = tf.stack(layer_outputs, 1)
            indices = tf.concat(1, [tf.reshape(tf.range(0, tf.size(sequence_length)), (-1, 1)),
                                    tf.cast(tf.reshape(sequence_length, (-1, 1))-1, dtype=tf.int32)])
            outputs = tf.gather_nd(outputs, indices)
    return outputs




def rnn(x, num_units,
        num_layers=1,
        activation=tf.nn.sigmoid,
        sequence_length=None,
        initial_state=None,
        keep_prob=None,
        is_training=False,
        return_seq=False,
        return_all_layers=False,
        variables_collections=tf.GraphKeys.MODEL_VARIABLES,
        outputs_collections=tf.GraphKeys.ACTIVATIONS,
        scope='SimpleRNN'):
    """Basic RNN wrapper"""
    from tflearn import simple_rnn as tfrnn

    if not (type(x) in (list, tuple)):
        x = tf.unstack(x, axis=1)

    dropout = _check_dropout(keep_prob, is_training)

    layer_outputs = [None] * (num_layers+1)
    layer_outputs[0] = x
    with tf.variable_scope(scope) as scope:
        for n in xrange(num_layers):
            layer_outputs[n+1] = tfrnn(layer_outputs[n], num_units,
                                       activation=activation,
                                       dropout=dropout,
                                       return_seq=True,
                                       initial_state=initial_state,
                                       sequence_length=sequence_length)    # Dropout during evaluation mode!!!
        x = _prep_outputs(layer_outputs, return_seq, return_all_layers)

    rnn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
    if variables_collections is not None:
        [tf.add_to_collection(variables_collections, var) for var in rnn_vars]
    if outputs_collections is not None:
        tf.add_to_collection(outputs_collections, x)

    return x

def lstm(x, num_units,
         num_layers=1,
         activation=tf.nn.sigmoid,
         inner_activation=tf.nn.tanh,
         sequence_length=None,
         initial_state=None,
         keep_prob=None,
         is_training=None,
         return_seq=False,
         return_all_layers=False,
         return_input_bow=False,
         variables_collections=tf.GraphKeys.MODEL_VARIABLES,
         outputs_collections=tf.GraphKeys.ACTIVATIONS,
         scope='LSTM'):
    """LSTM wrapper"""

    from tflearn import lstm as _lstm
    from tflib.wrappers import dropout

    if not (type(x) in (list, tuple)):
        x = tf.unstack(x, axis=1)

    keep_prob = _check_dropout(keep_prob, is_training)

    layer_outputs = [[w] for w in x]
    with tf.variable_scope(scope, values=[layer_outputs, keep_prob, is_training]) as scope:
        for n in range(num_layers):
            outps = _lstm([w[-1] for w in layer_outputs],
                          num_units,
                          activation=activation,
                          inner_activation=inner_activation,
                          return_seq=True,
                          initial_state=initial_state,
                          sequence_length=sequence_length)
            if keep_prob:
                outps = [dropout(outp, keep_prob, is_training) for outp in outps]
            [w.append(o) for w, o in zip(layer_outputs, outps)]
        x = _prep_outputs(layer_outputs, return_seq, return_all_layers, return_input_bow, sequence_length)

    rnn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
    if variables_collections is not None:
        [tf.add_to_collection(variables_collections, var) for var in rnn_vars]
    if outputs_collections is not None:
        tf.add_to_collection(outputs_collections, x)

    return x


def gru(x, num_units,
        num_layers=1,
        activation=tf.nn.sigmoid,
        inner_activation=tf.nn.tanh,
        sequence_length=None,
        initial_state=None,
        keep_prob=None,
        is_training=False,
        return_seq=False,
        return_all_layers=False,
        return_input_bow=False,
        variables_collections=tf.GraphKeys.MODEL_VARIABLES,
        outputs_collections=tf.GraphKeys.ACTIVATIONS,
        scope='GRU'):
    """GRU wrapper"""

    from tflearn import gru as _gru
    from tflib.wrappers import dropout

    if not (type(x) in (list, tuple)):
        x = tf.unstack(x, axis=1)

    keep_prob = _check_dropout(keep_prob, is_training)

    layer_outputs = [[w] for w in x]
    with tf.variable_scope(scope) as scope:
        for n in xrange(num_layers):
            outps = _gru([w[-1] for w in layer_outputs],
                         num_units,
                         activation=activation,
                         inner_activation=inner_activation,
                         return_seq=True,
                         initial_state=initial_state,
                         sequence_length=sequence_length)    # Dropout during evaluation mode!!!
            if keep_prob:
                outps = [dropout(outp, keep_prob, is_training) for outp in outps]
            [w.append(o) for w, o in zip(layer_outputs, outps)]
    x = _prep_outputs(layer_outputs, return_seq, return_all_layers, return_input_bow, sequence_length)

    rnn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
    if variables_collections is not None:
        [tf.add_to_collection(variables_collections, var) for var in rnn_vars]
    if outputs_collections is not None:
        tf.add_to_collection(outputs_collections, x)

    return x


def bidirectional_lstm(x, num_units,
                       num_layers=1,
                       activation=tf.nn.sigmoid,
                       inner_activation=tf.nn.tanh,
                       sequence_length=None,
                       initial_state_fw=None,
                       initial_state_bw=None,
                       keep_prob=None,
                       is_training=None,
                       return_seq=False,
                       return_all_layers=False,
                       return_input_bow=False,
                       variables_collections=tf.GraphKeys.MODEL_VARIABLES,
                       outputs_collections=tf.GraphKeys.ACTIVATIONS,
                       scope='BidirectionalLSTM'):
    """LSTM wrapper"""
    from tflearn import bidirectional_rnn as _bidirectional_rnn
    from tflearn import BasicLSTMCell as _BasicLSTMCell
    from tflib.wrappers import dropout

    if not (type(x) in (list, tuple)):
        x = tf.unstack(x, axis=1)

    keep_prob = _check_dropout(keep_prob, is_training)

    layer_outputs = [[w] for w in x]
    with tf.variable_scope(scope) as scope:
        for n in xrange(num_layers):
            cell_fw = _BasicLSTMCell(num_units, activation, inner_activation)
            cell_bw = _BasicLSTMCell(num_units, activation, inner_activation)
            outps = _bidirectional_rnn([w[-1] for w in layer_outputs], cell_fw, cell_bw,
                                       initial_state_fw=initial_state_fw,
                                       initial_state_bw=initial_state_bw,
                                       return_seq=True,
                                       sequence_length=sequence_length)
            if keep_prob:
                outps = [dropout(outp, keep_prob, is_training) for outp in outps]
            [w.append(o) for w, o in zip(layer_outputs, outps)]
        x = _prep_outputs(layer_outputs, return_seq, return_all_layers, return_input_bow, sequence_length)

    rnn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
    if variables_collections is not None:
        [tf.add_to_collection(variables_collections, var) for var in rnn_vars]
    if outputs_collections is not None:
        tf.add_to_collection(outputs_collections, x)

    return x


def bidirectional_gru(x, num_units,
                       num_layers=1,
                       activation=tf.nn.sigmoid,
                       inner_activation=tf.nn.tanh,
                       sequence_length=None,
                       initial_state_fw=None,
                       initial_state_bw=None,
                       keep_prob=None,
                       is_training=None,
                       return_seq=False,
                       return_all_layers=False,
                       return_input_bow=False,
                       variables_collections=tf.GraphKeys.MODEL_VARIABLES,
                       outputs_collections=tf.GraphKeys.ACTIVATIONS,
                       scope='BidirectionalLSTM'):
    """Bidirectional GRU wrapper"""

    from tflearn import bidirectional_rnn as _bidirectional_rnn
    from tflearn import GRUCell as _GRUCell
    from tflib.wrappers import dropout

    if not (type(x) in (list, tuple)):
        x = tf.unstack(x, axis=1)

    keep_prob = _check_dropout(keep_prob, is_training)

    layer_outputs = [[w] for w in x]
    for n in xrange(num_layers):
        with tf.variable_scope(scope+str(n)) as scope:
            cell_fw = _GRUCell(num_units, activation, inner_activation)
            cell_bw = _GRUCell(num_units, activation, inner_activation)
            outps = _bidirectional_rnn([w[-1] for w in layer_outputs], cell_fw, cell_bw,
                                       initial_state_fw=initial_state_fw,
                                       initial_state_bw=initial_state_bw,
                                       return_seq=True,
                                       sequence_length=sequence_length)
            if keep_prob:
                outps = [dropout(outp, keep_prob, is_training) for outp in outps]
            [w.append(o) for w, o in zip(layer_outputs, outps)]
    with tf.variable_scope(scope+'-out') as scope:
        x = _prep_outputs(layer_outputs, return_seq, return_all_layers, return_input_bow, sequence_length)

    rnn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
    if variables_collections is not None:
        [tf.add_to_collection(variables_collections, var) for var in rnn_vars]
    if outputs_collections is not None:
        tf.add_to_collection(outputs_collections, x)

    return x

def _test_lstm():
    from tflib.wrappers import embedding
    x = tf.random_uniform((128, 25), 0, 5000, tf.int64)
    xlen = tf.random_uniform((128, ), 0, 24, tf.int64)
    x = embedding(x, 5000, 300)
    x = bidirectional_gru(x, 1024, 2, sequence_length=xlen, keep_prob=0.5, is_training=True, return_seq=False, return_all_layers=True, return_input_bow=False)
    print x


if __name__ == '__main__':
    _test_lstm()

