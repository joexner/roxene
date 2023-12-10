from unittest import TestCase

from numpy.random import default_rng, Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from unittest.mock import Mock

from roxene import Gene, CompositeGene, CreateNeuron, Organism, random_neuron_state, Neuron
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

    def test_save_organism_inputs(self):

        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)

        o1 = Organism(input_names={'A', 'B', 'C'})
        oid = o1.id

        with Session(engine) as session:
            session.add(o1)
            session.commit()

        with Session(engine) as session:
            o2 = session.get(Organism, oid)
            self.assertFalse(o2 is None)

        # Test updating the organism outside a Session and merging it in a new Session
        o2.set_input('A', 12.34)
        o2.set_input('B', -5.67)
        o2.set_input('C', 890)

        with Session(engine) as session:
            session.add(o2)
            session.commit()

        with Session(engine) as session:
            o3 = session.get(Organism, oid)
            self.assertFalse(o3 is None)
            self.assertEqual(12.34, o3.input_cells['A'].get_output())
            self.assertEqual(-5.67, o3.input_cells['B'].get_output())
            self.assertEqual(890, o3.input_cells['C'].get_output())

    def test_neuron_update_order(self):
        # Create a list to store the order of update calls
        update_order = []

        # Create mock Neurons and add them to the Organism
        mock_neurons = [Mock(spec=Neuron) for _ in range(5)]
        organism = Organism(input_names={'A'})

        # Function to generate side effect function
        def side_effect_func(i):
            return lambda: update_order.append(i)

        for i, mock_neuron in enumerate(mock_neurons):
            # Each time update is called, append the index to update_order
            mock_neuron.update.side_effect = side_effect_func(i)
            organism.addNeuron(mock_neuron)

        # Call update on the Organism
        organism.update()

        # Check that the update calls on the Neurons happened in the correct order, LIFO (or LIFU?)
        expected_order = list(reversed(range(5)))
        self.assertEqual(update_order, expected_order)

        # Verify that update was called once on each mock Neuron
        for mock_neuron in mock_neurons:
            mock_neuron.update.assert_called_once()
