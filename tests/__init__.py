import tensorflow as tf

from roxene import Neuron


def random_tensor(size) -> tf.Tensor:
    return tf.random.uniform(
        shape=size,
        minval=-1,
        maxval=1,
        dtype=Neuron.PRECISION)

