from builtins import int
from enum import IntEnum

from tensorflow import Tensor

from roxene import Gene, Organism, Neuron, Cell, InputCell


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
        organism.cells.appendleft(neuron)


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
    class Direction(IntEnum):
        FORWARD = 1
        BACKWARD = -1

    def __init__(self, direction: Direction):
        self.direction = direction

    def execute(self, organism: Organism):
        organism.cells.rotate(self.direction)

class CreateInputCell(Gene):

    def __init__(self, initial_value):
        self.initial_value = initial_value


    def execute(self, organism: Organism):
        input_cell = InputCell(self.initial_value)
        organism.cells.appendleft(input_cell)

class CompositeGene(Gene):

    def __init__(self, genes: [Gene], iterations: int = 1):
        self.genes = genes
        self.iterations = iterations

    def execute(self, organism: Organism):
        for n in range(self.iterations):
            for gene in self.genes:
                gene.execute(organism)