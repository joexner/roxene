from builtins import int
from enum import IntEnum, Enum, auto
from numpy import ndarray

from .cells import Cell, InputCell
from .neuron import Neuron
from .organism import Gene, Organism


class CNLayer(Enum):
    input_initial_value = auto()
    feedback_initial_value = auto()
    output_initial_value = auto()
    input_hidden = auto()
    hidden_feedback = auto()
    feedback_hidden = auto()
    hidden_output = auto()


class CreateNeuron(Gene):
    def __init__(self,
                 input: ndarray,
                 feedback: ndarray,
                 output: ndarray,
                 input_hidden: ndarray,
                 hidden_feedback: ndarray,
                 feedback_hidden: ndarray,
                 hidden_output: ndarray,
                 parent_gene=None):
        super().__init__(parent_gene)
        self.input = input
        self.feedback = feedback
        self.output = output
        self.input_hidden = input_hidden
        self.hidden_feedback = hidden_feedback
        self.feedback_hidden = feedback_hidden
        self.hidden_output = hidden_output


    def execute(self, organism: Organism):
        neuron = Neuron(
            input=self.input,
            feedback=self.feedback,
            output=self.output,
            input_hidden=self.input_hidden,
            hidden_feedback=self.hidden_feedback,
            feedback_hidden=self.feedback_hidden,
            hidden_output=self.hidden_output
        )
        organism.addNeuron(neuron)


class ConnectNeurons(Gene):

    def __init__(self, tx_cell_index, rx_input_port, parent_gene=None):
        super().__init__(parent_gene)
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

    def __init__(self, direction: Direction = Direction.BACKWARD, parent_gene=None):
        super().__init__(parent_gene)
        self.direction = direction

    def execute(self, organism: Organism):
        if self.direction is RotateCells.Direction.FORWARD:
            popped = organism.cells.pop()
            organism.cells.insert(0, popped)
        elif self.direction is RotateCells.Direction.BACKWARD:
            popped = organism.cells.pop(0)
            organism.cells.append(popped)


class CreateInputCell(Gene):

    def __init__(self, initial_value, parent_gene=None):
        super().__init__(parent_gene)
        self.initial_value = initial_value

    def execute(self, organism: Organism):
        input_cell = InputCell(self.initial_value)
        organism.cells.insert(0, input_cell)


class CompositeGene(Gene):

    def __init__(self, genes: [Gene], iterations: int = 1, parent_gene=None):
        super().__init__(parent_gene)
        self.genes = genes
        self.iterations = iterations

    def execute(self, organism: Organism):
        for n in range(self.iterations):
            for gene in self.genes:
                gene.execute(organism)
