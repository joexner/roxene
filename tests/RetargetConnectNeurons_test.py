import unittest

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase
from roxene.genes import ConnectNeurons, CompositeGene
from roxene.mutagens import RetargetConnectNeurons
from roxene.util import set_rng

SEED = 456


class RetargetConnectNeuronsMutagen_test(unittest.TestCase):

    def setUp(self):
        set_rng(default_rng(SEED))

    def test_retarget_connect_neurons(self):
        """Test that RetargetConnectNeurons changes only tx_index, not rx_port"""
        original_connection = ConnectNeurons(tx_cell_index=5, rx_input_port=3)
        
        mutagen = RetargetConnectNeurons(0.01)
        
        mutant_gene = mutagen.mutate_ConnectNeurons(original_connection)
        
        # Should still be a ConnectNeurons
        self.assertIsInstance(mutant_gene, ConnectNeurons)
        # rx_port should remain unchanged
        self.assertEqual(mutant_gene.rx_port, original_connection.rx_port)


    def test_retarget_bounds(self):
        """Test that retargeted connections have non-negative tx_index"""
        original_connection = ConnectNeurons(tx_cell_index=1, rx_input_port=1)
        
        mutagen = RetargetConnectNeurons(0.01)
        
        for _ in range(20):
            mutant = mutagen.mutate_ConnectNeurons(original_connection)
            # tx_index should never be negative
            self.assertGreaterEqual(mutant.tx_cell_index, 0)
            # rx_port should remain unchanged
            self.assertEqual(mutant.rx_port, original_connection.rx_port)

    def test_retarget_in_composite(self):
        """Test that connections within CompositeGenes are retargeted"""
        connection1 = ConnectNeurons(tx_cell_index=5, rx_input_port=3)
        connection2 = ConnectNeurons(tx_cell_index=2, rx_input_port=7)
        composite = CompositeGene(child_genes=[connection1, connection2], iterations=1)
        
        mutagen = RetargetConnectNeurons(1.0)  # 100% susceptibility for recursive mutation
        
        mutant_composite = mutagen.mutate(composite)
        
        # Should still be a CompositeGene
        self.assertIsInstance(mutant_composite, CompositeGene)
        self.assertEqual(len(mutant_composite.child_genes), 2)
        # Child genes should be ConnectNeurons
        for gene in mutant_composite.child_genes:
            self.assertIsInstance(gene, ConnectNeurons)

    def test_persist_reload(self):
        """Test that RetargetConnectNeurons can be persisted and reloaded"""
        mutagen = RetargetConnectNeurons(0.025)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        with Session(engine) as session:
            reloaded = session.get(RetargetConnectNeurons, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.base_susceptibility, 0.025)
