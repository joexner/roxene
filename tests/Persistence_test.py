import uuid
from unittest import TestCase
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

from roxene import Neuron, random_neuron_state


class Persistable(DeclarativeBase):
    pass

class NeuronDTO(Persistable):

    __tablename__ = "neuron"

    id: Mapped[str] = mapped_column(primary_key=True)


class Persistence_test(TestCase):

    def test_save_new_thing(self):
        engine = create_engine("sqlite:////tmp/persistence_test.db", echo=True)
        Persistable.metadata.create_all(engine)
        neuron_ids = set()

        with Session(engine) as session:
            for n in range(100):
                id = str(uuid.uuid4())
                neuron: NeuronDTO = NeuronDTO(id=id)
                session.add(neuron)
                neuron_ids.add(id)
                self.assertIs(id, neuron.id)
            session.commit()

        with Session(engine) as session:
            for id in neuron_ids:
                neuron: NeuronDTO = session.get(NeuronDTO, id)
                self.assertEqual(id, neuron.id)
