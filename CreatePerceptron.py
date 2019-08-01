import tensorflow as tf

precision = tf.float16

class CreatePerceptron:
    def __init__(self, input_sz, hidden_sz, feedback_sz):

        self.input = tf.random.uniform(
            shape=[input_sz],
            minval=-1,
            maxval=1,
            dtype=precision);

        self.hidden = tf.random.uniform(
            shape=[hidden_sz],
            minval=-1,
            maxval=1,
            dtype=precision);

        self.feedback = tf.random.uniform(
            shape=[feedback_sz],
            minval=-1,
            maxval=1,
            dtype=precision);

        self.input_hidden = tf.random.uniform(
            shape=[input_sz, hidden_sz],
            minval=-1,
            maxval=1,
            dtype=precision);

        self.input_hidden = tf.random.uniform(
            shape=[input_sz, hidden_sz],
            minval=-1,
            maxval=1,
            dtype=precision);

