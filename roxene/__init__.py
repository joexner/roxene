import tensorflow as tf

PRECISION = tf.dtypes.float16

from .core import Gene, Organism
from .Cell import Cell
from .Neuron import Neuron
from .InputCell import InputCell
from .ConnectNeurons import ConnectNeurons
from .ConnectNeurons import ConnectNeurons
from .CreateNeuron import CreateNeuron