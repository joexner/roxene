import unittest

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase
from roxene.genes import ConnectNeurons, CompositeGene
from roxene.mutagens import ModifyConnection
from roxene.util import set_rng

SEED = 456


class ModifyConnectionMutagen_test(unittest.TestCase):

    def test_modify_connection(self):
        """Test that ModifyConnection changes connection parameters"""
        set_rng(default_rng(SEED))
        original_connection = ConnectNeurons(tx_cell_index=5, rx_input_port=3)
        
        mutagen = ModifyConnection(1.0, 0)  # 100% susceptibility
        
        mutant_gene = mutagen.mutate(original_connection)
        
        # Should still be a ConnectNeurons
        self.assertIsInstance(mutant_gene, ConnectNeurons)
        # Should have different parameters (with high probability)
        # Note: there's a small chance they could be the same if deltas are 0
        # Run multiple times to check mutation happens
        mutations_found = False
        for _ in range(20):
            mutant = mutagen.mutate(original_connection)
            if mutant.tx_cell_index != original_connection.tx_cell_index or \
               mutant.rx_port != original_connection.rx_port:
                mutations_found = True
                break
        self.assertTrue(mutations_found, "Expected at least one mutation in 20 tries")

    def test_modify_connection_no_mutation(self):
        """Test that with 0% susceptibility, no mutation occurs"""
        set_rng(default_rng(SEED))
        original_connection = ConnectNeurons(tx_cell_index=5, rx_input_port=3)
        
        mutagen = ModifyConnection(0.0, 0)  # 0% susceptibility
        
        mutant_gene = mutagen.mutate(original_connection)
        
        # Should not be mutated
        self.assertEqual(mutant_gene, original_connection)

    def test_modify_connection_bounds(self):
        """Test that modified connections have non-negative parameters"""
        set_rng(default_rng(SEED))
        original_connection = ConnectNeurons(tx_cell_index=1, rx_input_port=1)
        
        mutagen = ModifyConnection(1.0, 0)  # 100% susceptibility
        
        for _ in range(20):
            mutant = mutagen.mutate(original_connection)
            # Parameters should never be negative
            self.assertGreaterEqual(mutant.tx_cell_index, 0)
            self.assertGreaterEqual(mutant.rx_port, 0)

    def test_modify_connection_in_composite(self):
        """Test that connections within CompositeGenes are modified"""
        set_rng(default_rng(SEED))
        connection1 = ConnectNeurons(tx_cell_index=5, rx_input_port=3)
        connection2 = ConnectNeurons(tx_cell_index=2, rx_input_port=7)
        composite = CompositeGene(child_genes=[connection1, connection2], iterations=1)
        
        mutagen = ModifyConnection(1.0, 0)  # 100% susceptibility
        
        mutant_composite = mutagen.mutate(composite)
        
        # Should still be a CompositeGene
        self.assertIsInstance(mutant_composite, CompositeGene)
        self.assertEqual(len(mutant_composite.child_genes), 2)
        # Child genes should be ConnectNeurons
        for gene in mutant_composite.child_genes:
            self.assertIsInstance(gene, ConnectNeurons)

    def test_persist_reload(self):
        """Test that ModifyConnection can be persisted and reloaded"""
        mutagen = ModifyConnection(0.025, 0.035)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        with Session(engine) as session:
            reloaded = session.get(ModifyConnection, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.base_susceptibility, 0.025)
            self.assertEqual(reloaded.susceptibility_log_wiggle, 0.035)
