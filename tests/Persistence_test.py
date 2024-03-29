import pickle
import tensorflow as tf
import uuid
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from typing import Dict

from roxene import Neuron, random_neuron_state


class Base(DeclarativeBase):
    pass


class NeuronDTO(Base):
    __tablename__ = "neuron"

    id: Mapped[str] = mapped_column(primary_key=True)
    input: Mapped[bytes]
    feedback: Mapped[bytes]
    output: Mapped[bytes]
    input_hidden: Mapped[bytes]
    hidden_feedback: Mapped[bytes]
    feedback_hidden: Mapped[bytes]
    hidden_output: Mapped[bytes]

    def __init__(self, neuron: Neuron, id: str):
        self.id = id
        self.input = pickle.dumps(neuron.input.numpy(), protocol=5)
        self.feedback = pickle.dumps(neuron.feedback.numpy(), protocol=5)
        self.output = pickle.dumps(neuron.output.numpy(), protocol=5)
        self.input_hidden = pickle.dumps(neuron.input_hidden.numpy(), protocol=5)
        self.hidden_feedback = pickle.dumps(neuron.hidden_feedback.numpy(), protocol=5)
        self.feedback_hidden = pickle.dumps(neuron.feedback_hidden.numpy(), protocol=5)
        self.hidden_output = pickle.dumps(neuron.hidden_output.numpy(), protocol=5)

    def rehydrate(self):
        return Neuron(
            input=pickle.loads(self.input),
            feedback=pickle.loads(self.feedback),
            output=pickle.loads(self.output),
            input_hidden=pickle.loads(self.input_hidden),
            hidden_feedback=pickle.loads(self.hidden_feedback),
            feedback_hidden=pickle.loads(self.feedback_hidden),
            hidden_output=pickle.loads(self.hidden_output),
        ), self.id


class Persistence_test(tf.test.TestCase):

    def test_save_new_neuron(self):
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)

        neurons: Dict[str, Neuron] = {}

        with Session(engine) as session:
            for n in range(100):
                neuron = Neuron(**random_neuron_state(input_size=14, hidden_size=28, feedback_size=12))
                id = str(uuid.uuid4())
                neurons[id] = neuron
                dto = NeuronDTO(neuron, id)
                session.add(dto)
            session.commit()

        with Session(engine) as session:
            for id, original_neuron in neurons.items():
                dto: NeuronDTO = session.get(NeuronDTO, id)
                reconstituted_neuron, re_id = dto.rehydrate()
                self.assertEqual(id, re_id)
                self.assertAllEqual(original_neuron.input, reconstituted_neuron.input)
                self.assertAllEqual(original_neuron.feedback, reconstituted_neuron.feedback)
                self.assertAllEqual(original_neuron.output, reconstituted_neuron.output)
                self.assertAllEqual(original_neuron.input_hidden, reconstituted_neuron.input_hidden)
                self.assertAllEqual(original_neuron.hidden_feedback, reconstituted_neuron.hidden_feedback)
                self.assertAllEqual(original_neuron.feedback_hidden, reconstituted_neuron.feedback_hidden)
                self.assertAllEqual(original_neuron.hidden_output, reconstituted_neuron.hidden_output)
