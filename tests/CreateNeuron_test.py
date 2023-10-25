import tensorflow as tf

from roxene import CreateNeuron, Organism, random_neuron_state


class CreateNeuron_test(tf.test.TestCase):

    def test_execute(self, input_size=12, feedback_size=3, hidden_size=38):
        org = Organism()

        params1 = random_neuron_state(input_size, feedback_size, hidden_size)
        cn1 = CreateNeuron(**params1)

        cn1.execute(org)
        neuron1 = org.cells[0]
        self.assertAllEqual(params1["input"], neuron1.input.value())
        self.assertAllEqual(params1["feedback"], neuron1.feedback.value())
        self.assertAllEqual(params1["output"], neuron1.output.value())
        self.assertAllEqual(params1["input_hidden"], neuron1.input_hidden)
        self.assertAllEqual(params1["hidden_feedback"], neuron1.hidden_feedback)
        self.assertAllEqual(params1["hidden_output"], neuron1.hidden_output)

        # CreateNeuron adds a neuron to the *beginning* of the organism's cells list
        cn1.execute(org)
        self.assertLen(org.cells, 2)
        self.assertIs(org.cells[1], neuron1)
        neuron2 = org.cells[0]
        self.assertIsNot(neuron2, neuron1)
        self.assertAllEqual(params1["input"], neuron2.input.value())
        self.assertAllEqual(params1["feedback"], neuron2.feedback.value())
        self.assertAllEqual(params1["output"], neuron2.output.value())
        self.assertAllEqual(params1["input_hidden"], neuron2.input_hidden)
        self.assertAllEqual(params1["hidden_feedback"], neuron2.hidden_feedback)
        self.assertAllEqual(params1["hidden_output"], neuron2.hidden_output)

        params2 = random_neuron_state(input_size, feedback_size, hidden_size)
        cn2 = CreateNeuron(**params2)

        cn2.execute(org)
        self.assertLen(org.cells, 3)
        self.assertIs(org.cells[2], neuron1)
        self.assertIs(org.cells[1], neuron2)

        neuron3 = org.cells[0]
        self.assertIsNot(neuron3, neuron1)
        self.assertIsNot(neuron3, neuron2)
        self.assertAllEqual(params2["input"], neuron3.input.value())
        self.assertAllEqual(params2["feedback"], neuron3.feedback.value())
        self.assertAllEqual(params2["output"], neuron3.output.value())
        self.assertAllEqual(params2["input_hidden"], neuron3.input_hidden)
        self.assertAllEqual(params2["hidden_feedback"], neuron3.hidden_feedback)
        self.assertAllEqual(params2["hidden_output"], neuron3.hidden_output)
