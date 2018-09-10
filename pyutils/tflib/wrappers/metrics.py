import tensorflow as tf

def accuracy(decisions, targets, weights=None):
    """ accuracy_op.
    An op that calculates mean accuracy, assuming predictiosn are targets
    are both one-hot encoded.
    Examples:
        ```python
        input_data = placeholder(shape=[None, 784])
        y_pred = my_network(input_data) # Apply some ops
        y_true = placeholder(shape=[None, 10]) # Labels
        acc_op = accuracy_op(y_pred, y_true)
        # Calculate accuracy by feeding datalib X and labels Y
        accuracy = sess.run(acc_op, feed_dict={input_data: X, y_true: Y})
        ```
    Arguments:
        predictions: `Tensor`.
        targets: `Tensor`.
    Returns:
        `Float`. The mean accuracy.
    """
    with tf.name_scope('Accuracy'):
        hits = tf.cast(tf.equal(decisions, targets), tf.float32)
        if weights is not None:
            den = tf.select(tf.equal(tf.reduce_mean(weights), 0), 1., tf.reduce_mean(weights))
            acc = tf.reduce_mean(hits * weights) / den
        else:
            acc = tf.reduce_mean(hits)
    return acc


def top_k(predictions, targets, k=1, weights=None):
    """ top_k_op.
    An op that calculates top-k mean accuracy.
    Examples:
        ```python
        input_data = placeholder(shape=[None, 784])
        y_pred = my_network(input_data) # Apply some ops
        y_true = placeholder(shape=[None, 10]) # Labels
        top3_op = top_k_op(y_pred, y_true, 3)
        # Calculate Top-3 accuracy by feeding datalib X and labels Y
        top3_accuracy = sess.run(top3_op, feed_dict={input_data: X, y_true: Y})
        ```
    Arguments:
        predictions: `Tensor`.
        targets: `Tensor`.
        k: `int`. Number of top elements to look at for computing precision.
    Returns:
        `Float`. The top-k mean accuracy.
    """
    with tf.name_scope('Top_' + str(k)):
        targets = tf.cast(targets, tf.int32)
        hits = tf.cast(tf.nn.in_top_k(predictions, targets, k), tf.float32)
        if weights is not None:
            acc = tf.reduce_mean(hits * weights) / tf.reduce_mean(weights)
        else:
            acc = tf.reduce_mean(hits)
    return acc
