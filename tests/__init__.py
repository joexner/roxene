import tensorflow as tf

from roxene import Neuron


def random_tensor(size) -> tf.Tensor:
    return tf.random.uniform(
        shape=size,
        minval=-1,
        maxval=1,
        dtype=Neuron.PRECISION)


def make_neuron(input_sz,
                hidden_sz,
                feedback_sz):

    input_initial_value = random_tensor([input_sz])
    feedback_initial_value = random_tensor([feedback_sz])
    output_initial_value = random_tensor([])
    input_hidden = random_tensor([input_sz, hidden_sz])
    hidden_feedback = random_tensor([hidden_sz, feedback_sz])
    feedback_hidden = random_tensor([feedback_sz, hidden_sz])
    hidden_output = random_tensor([hidden_sz, 1])

    return Neuron(
        input_initial_value=input_initial_value,
        feedback_initial_value=feedback_initial_value,
        output_initial_value=output_initial_value,
        input_hidden=input_hidden,
        hidden_feedback=hidden_feedback,
        feedback_hidden=feedback_hidden,
        hidden_output=hidden_output
    )