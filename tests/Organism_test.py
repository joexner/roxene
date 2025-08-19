import os
from unittest import TestCase
from uuid import UUID

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
        organism = Organism(input_names, output_names, CompositeGene(child_genes=[gene], iterations=10))
        self.assertSetEqual(input_names, set(organism.inputs.keys()))
        self.assertSetEqual(output_names, set(organism.outputs.keys()))

    def test_io(self):

        rng: Generator = default_rng(SEED)

        input_names = {'I_0', 'I_1', 'I_2'}
        output_names = {'O1', '02'}
        gene = CreateNeuron(**random_neuron_state(rng=rng))
        organism = Organism(input_names, output_names, CompositeGene(child_genes=[gene], iterations=10))


        for input_name in input_names:
            organism.set_input(input_name, rng.random())
        for output_name in output_names:
            organism.get_output(output_name)

        self.assertEqual(input_names, set(organism.inputs.keys()))
        self.assertEqual(output_names, set(organism.outputs.keys()))

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

    def test_save_organism_inputs(self):

        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)

        input_names = {'A', 'B', 'C'}
        o1 = Organism(input_names=input_names)
        oid = UUID(o1.id.hex)

        with Session(engine) as session:
            with session.begin():
                session.add(o1)

        with Session(engine) as session:
            o2 = session.get(Organism, oid)
            self.assertFalse(o2 is None)
            self.assertEqual(len(o2.inputs), len(input_names))

        # Test updating the organism outside a Session and merging it in a new Session
        o2.set_input('A', 12.34)
        o2.set_input('B', -5.67)
        o2.set_input('C', 890)

        with Session(engine) as session:
            with session.begin():
                session.add(o2)

        with Session(engine) as session:
            o3 = session.get(Organism, oid)
            self.assertFalse(o3 is None)
            self.assertEqual(12.34, o3.inputs['A'].get_output())
            self.assertEqual(-5.67, o3.inputs['B'].get_output())
            self.assertEqual(890, o3.inputs['C'].get_output())

    def test_save_organism_outputs(self):

        rng: Generator = default_rng(SEED)

        engine = create_engine("sqlite://")

        EntityBase.metadata.create_all(engine)

        output_names = ['1', '2', '3']

        o1 = Organism(output_names=set(output_names))

        oid = UUID(o1.id.hex)
        print(oid)
        for n in range(len(output_names)):
            neuron = Neuron(**random_neuron_state(rng=rng))
            o1.addNeuron(neuron)

        initial_outputs = [o1.get_output(label) for label in output_names]

        with Session(engine) as session:
            session.add(o1)
            session.commit()

        # Invalidate the Organism (and everything else) from cache
        session.expunge_all()

        with Session(engine) as session:
            o2 = session.get(Organism, oid)
            self.assertFalse(o2 is None)
            self.assertEqual(len(o2.outputs), len(output_names))
            current_outputs = [o2.get_output(label) for label in output_names]
            self.assertEqual(current_outputs, initial_outputs)

    def test_save_organism_cells(self):

        rng: Generator = default_rng(SEED)

        NUM_NEURONS = 2

        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)

        INPUT_NAMES = ['A', 'B', 'C']
        o1 = Organism(input_names=INPUT_NAMES)
        oid = UUID(o1.id.hex)

        self.assertEqual(len(o1.cells), len(INPUT_NAMES))

        # Add the InputCells' IDs to the expected list
        cell_ids = [cell.id for cell in o1.cells]

        for n in range(NUM_NEURONS):
            neuron = Neuron(**random_neuron_state(rng=rng))
            o1.addNeuron(neuron)
            cell_ids.insert(0, neuron.id)

        with Session(engine) as session:
            with session.begin():
                session.add(o1)

        # Add one more Neuron, but roll back
        with Session(engine) as session:
            with session.begin() as txn:
                o2 = session.get(Organism, oid)
                neuron = Neuron(**random_neuron_state(rng=rng))
                o2.addNeuron(neuron)
                txn.rollback()

        with Session(engine) as session:
            o3 = session.get(Organism, oid)
            self.assertFalse(o3 is None)
            self.assertEqual(len(o3.cells), len(cell_ids))
            current_cell_ids = [cell.id for cell in o3.cells]
            self.assertListEqual(current_cell_ids, cell_ids)

    def test_save_organism_unused_output_names(self):

        rng: Generator = default_rng(SEED)

        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)

        # 10 output names
        output_names = [str(n) for n in range(10)]

        # Add 3 Neurons
        with Session(engine) as session:
            with session.begin():
                o1 = Organism(output_names=set(output_names))
                session.add(o1)
                org_id = UUID(o1.id.hex)

                for n in range(3):
                    neuron = Neuron(**random_neuron_state(rng=rng))
                    o1.addNeuron(neuron)

                self.assertEqual(len(o1.unused_output_names), 7)

        with Session(engine) as session:
            with session.begin():
                o2 = session.get(Organism, org_id)
                self.assertFalse(o2 is None)
                self.assertEqual(len(o2.unused_output_names), 7)

                for n in range(5):
                    neuron = Neuron(**random_neuron_state(rng=rng))
                    o2.addNeuron(neuron)

                self.assertEqual(len(o2.unused_output_names), 2)

        with Session(engine) as session:
            with session.begin() as txn:
                o3 = session.get(Organism, org_id)
                self.assertFalse(o3 is None)
                self.assertEqual(len(o3.unused_output_names), 2)

                for n in range(2):
                    neuron = Neuron(**random_neuron_state(rng=rng))
                    o3.addNeuron(neuron)

                txn.rollback()

        with Session(engine) as session:
            with session.begin():
                o4 = session.get(Organism, org_id)
                self.assertFalse(o4 is None)
                self.assertEqual(len(o4.unused_output_names), 2)

                for n in range(2):
                    neuron = Neuron(**random_neuron_state(rng=rng))
                    o4.addNeuron(neuron)

        with Session(engine) as session:
            o5 = session.get(Organism, org_id)
            self.assertFalse(o5 is None)
            self.assertEqual(len(o5.unused_output_names), 0)
