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

    def test_delta_values_coverage(self):
        """Test with a moderate tx_cell_index that all valid delta values are hit.
        
        With tx_cell_index=10 and severity=1.0, max_delta = max(1, int(1.0 * 10)) = 10
        Raw deltas can be: -10, -9, ..., -1, +1, +2, ..., +10 (no zero)
        But new_tx_index is clamped to max(1, tx_cell_index + delta), so:
        - Negative deltas beyond -(tx_cell_index - 1) = -9 get clamped
        - Expected observed deltas: -9, -8, ..., -1, +1, +2, ..., +10
        """
        target = 10
        severity = 1.0
        original_connection = ConnectNeurons(tx_cell_index=target, rx_input_port=5)
        
        mutagen = RetargetConnectNeurons(base_susceptibility=0.01, severity=severity)
        
        # Calculate expected max delta based on severity and current target index
        max_delta = max(1, int(severity * target))  # max(1, 10) = 10
        
        observed_deltas = set()
        
        for _ in range(500):  # Run enough iterations to hit all values
            set_rng(default_rng())  # Use different random seeds
            mutant = mutagen.mutate_ConnectNeurons(original_connection)
            
            # Calculate the actual delta
            delta = mutant.tx_cell_index - original_connection.tx_cell_index
            
            # Delta should never be zero - mutation should always do something
            self.assertNotEqual(delta, 0, "Delta should never be zero")
            
            # Delta should be bounded appropriately
            # Positive deltas up to max_delta, negative deltas clamped by min tx_cell_index=1
            self.assertLessEqual(delta, max_delta, f"Delta {delta} exceeds max {max_delta}")
            self.assertGreaterEqual(delta, -(target - 1), f"Delta {delta} below min {-(target - 1)}")
            
            observed_deltas.add(int(delta))  # Convert numpy int to Python int
        
        # All valid delta values should be hit:
        # Positive: +1 to +max_delta
        # Negative: -(target-1) to -1 (since new_tx_index must be >= 1)
        expected_positive = set(range(1, max_delta + 1))
        expected_negative = set(range(-(target - 1), 0))
        expected_deltas = expected_positive | expected_negative
        
        self.assertEqual(
            observed_deltas, expected_deltas,
            f"Expected all deltas {sorted(expected_deltas)}, got {sorted(observed_deltas)}"
        )
