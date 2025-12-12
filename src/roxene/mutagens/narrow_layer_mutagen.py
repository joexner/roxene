import uuid

import numpy as np
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from ..constants import NP_PRECISION
from ..genes.create_neuron import CreateNeuron
from ..mutagen import Mutagen
from ..util import get_rng


class NarrowLayerMutagen(Mutagen):
    """
    Decreases the size of the hidden layer in a CreateNeuron gene.
    Removes neurons from the hidden layer by reducing weight matrices.
    """
    __tablename__ = "narrow_layer_mutagen"
    __mapper_args__ = {"polymorphic_identity": "narrow_layer_mutagen"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("mutagen.id"), primary_key=True)

    def __init__(self, base_susceptibility: float = 0.01, susceptibility_log_wiggle: float = 0.01):
        super().__init__(base_susceptibility, susceptibility_log_wiggle)

    def mutate_CreateNeuron(self, gene: CreateNeuron) -> CreateNeuron:
        susceptibility = self.get_mutation_susceptibility(gene)
        if get_rng().random() >= susceptibility:
            return gene

        # Get current hidden layer size from the shape of input_hidden
        current_hidden_size = gene.input_hidden.shape[1]
        
        # Don't narrow if already very small
        if current_hidden_size <= 2:
            return gene

        # Decrease hidden size by 1-2 neurons (but keep at least 1)
        max_remove = min(2, current_hidden_size - 1)
        neurons_to_remove = get_rng().integers(1, max_remove + 1).astype(int)
        new_hidden_size = current_hidden_size - neurons_to_remove

        # Select random indices to keep
        indices_to_keep = get_rng().choice(current_hidden_size, new_hidden_size, replace=False)
        indices_to_keep = np.sort(indices_to_keep)

        # Narrow input_hidden: from [input_size, hidden_size] to [input_size, new_hidden_size]
        new_input_hidden = gene.input_hidden[:, indices_to_keep]

        # Narrow feedback_hidden: from [feedback_size, hidden_size] to [feedback_size, new_hidden_size]
        new_feedback_hidden = gene.feedback_hidden[:, indices_to_keep]

        # Narrow hidden_feedback: from [hidden_size, feedback_size] to [new_hidden_size, feedback_size]
        new_hidden_feedback = gene.hidden_feedback[indices_to_keep, :]

        # Narrow hidden_output: from [hidden_size, 1] to [new_hidden_size, 1]
        new_hidden_output = gene.hidden_output[indices_to_keep, :]

        return CreateNeuron(
            input=gene.input,
            feedback=gene.feedback,
            output=gene.output,
            input_hidden=new_input_hidden,
            hidden_feedback=new_hidden_feedback,
            feedback_hidden=new_feedback_hidden,
            hidden_output=new_hidden_output,
            parent_gene=gene
        )
