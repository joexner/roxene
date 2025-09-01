import uuid

from numpy import ndarray
from sqlalchemy import ForeignKey, PickleType
from sqlalchemy.orm import Mapped, mapped_column

from roxene import Gene, Organism, Neuron


class CreateNeuron(Gene):
    __tablename__ = "create_neuron"
    __mapper_args__ = {"polymorphic_identity": "create_neuron"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("gene.id"), primary_key=True)

    input: Mapped[ndarray] = mapped_column(PickleType)
    feedback: Mapped[ndarray] = mapped_column(PickleType)
    output: Mapped[ndarray] = mapped_column(PickleType)
    input_hidden: Mapped[ndarray] = mapped_column(PickleType)
    hidden_feedback: Mapped[ndarray] = mapped_column(PickleType)
    feedback_hidden: Mapped[ndarray] = mapped_column(PickleType)
    hidden_output: Mapped[ndarray] = mapped_column(PickleType)


    def __init__(self,
                 input: ndarray,
                 feedback: ndarray,
                 output: ndarray,
                 input_hidden: ndarray,
                 hidden_feedback: ndarray,
                 feedback_hidden: ndarray,
                 hidden_output: ndarray,
                 parent_gene: Gene = None):
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
