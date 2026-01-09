import unittest

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase
from roxene.genes import ConnectNeurons, CompositeGene, RotateCells
from roxene.mutagens import AddConnectNeurons
from roxene.util import set_rng

SEED = 555


class AddConnectNeurons_test(unittest.TestCase):

    def test_add_connection_to_empty_composite(self):
        """
        Test that AddConnectNeurons adds a new ConnectNeurons gene to an empty CompositeGene.
        
        This verifies the basic functionality: given an empty composite gene, the mutagen
        should insert exactly one ConnectNeurons gene with the configured tx_cell_index
        and rx_port values.
        """
        set_rng(default_rng(SEED))
        composite = CompositeGene(child_genes=[], iterations=1)
        
        # Create mutagen with specific connection parameters
        mutagen = AddConnectNeurons(
            base_susceptibility=1.0,  # 100% chance of mutation
            tx_cell_index=3,           # Connect from cell at index 3
            rx_port=5                   # Connect to input port 5
        )
        
        mutant_composite = mutagen.mutate(composite)
        
        # Should have exactly 1 child gene added
        self.assertIsInstance(mutant_composite, CompositeGene)
        self.assertEqual(len(mutant_composite.child_genes), 1)
        
        # The added gene should be a ConnectNeurons with the correct parameters
        added_gene = mutant_composite.child_genes[0]
        self.assertIsInstance(added_gene, ConnectNeurons)
        self.assertEqual(added_gene.tx_cell_index, 3)
        self.assertEqual(added_gene.rx_port, 5)

    def test_add_connection_to_non_empty_composite(self):
        """
        Test that AddConnectNeurons correctly adds a connection gene to a composite
        that already has existing child genes.
        
        The mutagen should insert the new gene at a random position while preserving
        all existing genes.
        """
        set_rng(default_rng(SEED))
        existing_gene = RotateCells(RotateCells.Direction.FORWARD)
        composite = CompositeGene(child_genes=[existing_gene], iterations=1)
        
        mutagen = AddConnectNeurons(1.0, tx_cell_index=0, rx_port=1)
        
        mutant_composite = mutagen.mutate(composite)
        
        # Should have 2 child genes now
        self.assertEqual(len(mutant_composite.child_genes), 2)
        
        # One should be the original RotateCells, one should be new ConnectNeurons
        gene_types = {type(g).__name__ for g in mutant_composite.child_genes}
        self.assertEqual(gene_types, {'RotateCells', 'ConnectNeurons'})

    def test_add_connection_default_parameters(self):
        """
        Test that AddConnectNeurons uses default parameter values (tx_cell_index=0, rx_port=0)
        when not explicitly specified.
        """
        set_rng(default_rng(SEED))
        composite = CompositeGene(child_genes=[], iterations=1)
        
        # Only specify base_susceptibility, use defaults for connection params
        mutagen = AddConnectNeurons(1.0)
        
        mutant_composite = mutagen.mutate(composite)
        
        added_gene = mutant_composite.child_genes[0]
        self.assertIsInstance(added_gene, ConnectNeurons)
        # Default values
        self.assertEqual(added_gene.tx_cell_index, 0)
        self.assertEqual(added_gene.rx_port, 0)

    def test_persist_reload(self):
        """
        Test that AddConnectNeurons can be saved to and loaded from the database
        with all its configuration preserved.
        """
        mutagen = AddConnectNeurons(0.025, tx_cell_index=7, rx_port=3)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        
        with Session(engine) as session:
            reloaded = session.get(AddConnectNeurons, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.base_susceptibility, 0.025)
            self.assertEqual(reloaded.tx_cell_index, 7)
            self.assertEqual(reloaded.rx_port, 3)
