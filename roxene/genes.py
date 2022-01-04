from tensorflow import Tensor

from roxene import Gene, Organism, Neuron, Cell


class CreateNeuron(Gene):
    def __init__(self,
        input_initial_value: Tensor,
        feedback_initial_value: Tensor,
        output_initial_value: Tensor,
        input_hidden: Tensor,
        hidden_feedback: Tensor,
        feedback_hidden: Tensor,
        hidden_output: Tensor):

        self.input_initial_value    = input_initial_value
        self.feedback_initial_value = feedback_initial_value
        self.output_initial_value   = output_initial_value
        self.input_hidden           = input_hidden
        self.hidden_feedback        = hidden_feedback
        self.feedback_hidden        = feedback_hidden
        self.hidden_output          = hidden_output


    def execute(self, organism: Organism):
        neuron = Neuron(
            input_initial_value=self.input_initial_value,
            feedback_initial_value=self.feedback_initial_value,
            output_initial_value=self.output_initial_value,
            input_hidden=self.input_hidden,
            hidden_feedback=self.hidden_feedback,
            feedback_hidden=self.feedback_hidden,
            hidden_output=self.hidden_output
        )
        organism.add(neuron)


class ConnectNeurons(Gene):

    def __init__(self, tx_cell_index, rx_input_port):
        self.tx_cell_index = tx_cell_index
        self.rx_port = rx_input_port


    def execute(self, organism: Organism):
        index: int = self.tx_cell_index % len(organism.cells)
        tx_cell: Cell = organism.cells[index]
        rx_cell: Neuron = organism.cells[0]
        rx_cell.add_input_connection(tx_cell, self.rx_port)


class RotateCells(Gene):

    def __init__(self, direction):
        self.direction = direction
        pass

    def execute(self, organism: Organism):
        organism.cells.rotate(self.direction)