import tensorflow as tf
from tensorflow.contrib.framework.python.ops import variables


def embedding(x, vocab_dim, emb_dim,
              trainable=True,
              dtype=tf.float32,
              initializer=None,
              activation_collection=tf.GraphKeys.ACTIVATIONS,
              variable_collection=tf.GraphKeys.MODEL_VARIABLES,
              scope='lookup'):
    if initializer is None:
        init_width = 0.5 / emb_dim
        initializer = tf.random_uniform_initializer(-init_width, init_width)
    W = variables.model_variable('embedding',
                                 shape=[vocab_dim, emb_dim],
                                 dtype=dtype,
                                 initializer=initializer,
                                 trainable=trainable)
    x = tf.nn.embedding_lookup(W, x, name=scope)
    if activation_collection is not None:
        tf.add_to_collection(activation_collection, x)
    if variable_collection is not None:
        tf.add_to_collection(variable_collection, W)

    return x


