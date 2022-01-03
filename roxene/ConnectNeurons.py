from roxene import Cell, Gene, Neuron, Organism


class ConnectNeurons(Gene):

    def __init__(self, tx_cell_index, rx_input_port):
        self.tx_cell_index = tx_cell_index
        self.rx_port = rx_input_port


    def execute(self, organism: Organism):
        index: int = self.tx_cell_index % len(organism.cells)
        tx_cell: Cell = organism.cells[index]
        rx_cell: Neuron = organism.cells[0]
        rx_cell.add_input_connection(tx_cell, self.rx_port)


