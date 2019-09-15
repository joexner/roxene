import tensorflow as tf

from CreatePerceptron import CreatePerceptron
from Organism import Organism
from Perceptron import Perceptron


class CreatePerceptron_test(tf.test.TestCase):

    def setUp(self) -> None:
        tf.enable_eager_execution()


    def test_execute(self, input_size=12, feedback_size=3, hidden_size=38):

        input_initial_value=        tf.random_uniform([input_size],                 -1., 1., dtype=Perceptron.PRECISION)
        feedback_initial_value=     tf.random_uniform([feedback_size],              -1., 1., dtype=Perceptron.PRECISION)
        output_initial_value=       tf.random_uniform([1],                          -1., 1., dtype=Perceptron.PRECISION)
        input_hidden=               tf.random_uniform([input_size, hidden_size],    -1., 1., dtype=Perceptron.PRECISION)
        hidden_feedback=            tf.random_uniform([hidden_size, feedback_size], -1., 1., dtype=Perceptron.PRECISION)
        feedback_hidden=            tf.random_uniform([feedback_size, hidden_size], -1., 1., dtype=Perceptron.PRECISION)
        hidden_output=              tf.random_uniform([hidden_size],                -1., 1., dtype=Perceptron.PRECISION)

        cp = CreatePerceptron(
            input_initial_value,
            feedback_initial_value,
            output_initial_value,
            input_hidden,
            hidden_feedback,
            feedback_hidden,
            hidden_output
        )

        org = Organism()
        cp.execute(org)
        p: Perceptron = org.cells[0]
        self.assertAllEqual(p.input.numpy(), input_initial_value.numpy())
        self.assertAllEqual(p.feedback.numpy(), feedback_initial_value.numpy())
        self.assertAllEqual(p.output.numpy(), output_initial_value.numpy())
        self.assertAllEqual(p.input_hidden.numpy(), input_hidden.numpy())
        self.assertAllEqual(p.hidden_feedback.numpy(), hidden_feedback.numpy())
        self.assertAllEqual(p.hidden_output.numpy(), hidden_output.numpy())