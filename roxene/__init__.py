import tensorflow as tf

PRECISION = tf.dtypes.float16

from .core import Gene, Organism, Cell, InputCell
from .Neuron import Neuron
from .ConnectNeurons import ConnectNeurons
from .CreateNeuron import CreateNeuron