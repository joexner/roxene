import pickle
import uuid
from unittest import TestCase

import numpy as np
from numpy import ndarray
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

from roxene import Neuron, random_neuron_state


class Persistable(DeclarativeBase):
    pass


class NeuronDTO(Persistable):

    __tablename__ = "neuron"

    id: Mapped[str] = mapped_column(primary_key=True)
    input_data: Mapped[bytes] = mapped_column(name='input')


class Persistence_test(TestCase):

    def test_save_new_thing(self):
        engine = create_engine("sqlite:////tmp/persistence_test.db")
        Persistable.metadata.create_all(engine)
        neuron_ids = set()

        with Session(engine) as session:
            for n in range(100):
                id = str(uuid.uuid4())

                input_ndarr: ndarray = np.random.normal(size=23).astype(np.float16)
                # Add a checksum
                input_ndarr = np.append(input_ndarr, -1 * input_ndarr.sum())
                input_data = pickle.dumps(input_ndarr)

                neuron: NeuronDTO = NeuronDTO(
                    id=id,
                    input_data=input_data,
                )

                session.add(neuron)
                neuron_ids.add(id)
                self.assertIs(id, neuron.id)
            session.commit()

        with Session(engine) as session:
            for id in neuron_ids:
                neuron: NeuronDTO = session.get(NeuronDTO, id)
                self.assertEqual(id, neuron.id)
                data: ndarray = pickle.loads(neuron.input_data)
                self.assertEqual(len(data), 24)
                self.assertAlmostEqual(data.sum(), 0, places=2)
