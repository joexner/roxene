from unittest import TestCase

from numpy.random import default_rng, Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from unittest.mock import Mock

from roxene import Gene, CompositeGene, CreateNeuron, Organism, random_neuron_state
from roxene.persistence import EntityBase

SEED = 22049456


class Organism_test(TestCase):

    def test_constructor_genoytpe(self):
        root_gene = Mock(Gene)
        organism = Organism(genotype=root_gene)
        print(root_gene.method_calls)
        root_gene.execute.assert_called_once_with(organism)

    def test_constructor_input_output_names(self):
        input_names = {'I_0', 'I_1', 'I_2'}
        output_names = {'O1', '02'}
        gene = CreateNeuron(**random_neuron_state(input_size=2, feedback_size=3, hidden_size=1))
        organism = Organism(input_names, output_names, CompositeGene(genes=[gene], iterations=10))
        self.assertSetEqual(input_names, set(organism.input_cells.keys()))
        self.assertSetEqual(output_names, set(organism.outputs.keys()))

    def test_io(self):

        rng: Generator = default_rng(SEED)

        input_names = {'I_0', 'I_1', 'I_2'}
        output_names = {'O1', '02'}
        gene = CreateNeuron(**random_neuron_state(rng=rng))
        organism = Organism(input_names, output_names, CompositeGene(genes=[gene], iterations=10))


        for input_name in input_names:
            organism.set_input(input_name, rng.random())
        for output_name in output_names:
            organism.get_output(output_name)

        self.assertEqual(input_names, set(organism.input_cells.keys()))
        self.assertEqual(output_names, set(organism.outputs.keys()))

    def test_save_organism(self):

        engine = create_engine("sqlite:////tmp/test.db")
        EntityBase.metadata.create_all(engine)

        o1 = Organism(input_names={'1'})
        oid = o1.id

        with Session(engine) as session:
            session.add(o1)
            session.commit()

        with Session(engine) as session:
            o2 = session.get(Organism, oid)
            self.assertFalse(o2 is None)
            o2.set_input('1', 12.34)
            session.commit()

        with Session(engine) as session:
            o3 = session.get(Organism, oid)
            self.assertFalse(o2 is None)
            self.assertEqual(12.34, o3.input_cells['1'].value)
