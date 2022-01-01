import tensorflow as tf

precision = tf.dtypes.float16


def random_tensor(size) -> tf.Tensor:
    return tf.random.uniform(
        shape=size,
        minval=-1,
        maxval=1,
        dtype=precision)
