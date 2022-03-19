import tensorflow as tf

PRECISION = tf.dtypes.float16

from .core import Gene, Organism, Cell, InputCell
from .neuron import Neuron
from .genes import CreateNeuron, ConnectNeurons, RotateCells, CreateInputCell, CompositeGene