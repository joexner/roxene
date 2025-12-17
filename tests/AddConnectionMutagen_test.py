import unittest
from typing import List

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, Gene
from roxene.genes import CompositeGene, RotateCells, ConnectNeurons
from roxene.mutagens import AddConnectionMutagen
from roxene.util import set_rng

SEED = 123


class AddConnectionMutagen_test(unittest.TestCase):

    def test_add_connection_to_composite(self):
        """Test that AddConnectionMutagen adds a ConnectNeurons gene"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [
            RotateCells(RotateCells.Direction.FORWARD),
            RotateCells(RotateCells.Direction.BACKWARD)
        ]
        original_gene = CompositeGene(child_genes=child_genes, iterations=3)
        
        mutagen = AddConnectionMutagen(1.0, 0)  # 100% susceptibility
        
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should still be a CompositeGene
        self.assertIsInstance(mutant_gene, CompositeGene)
        self.assertEqual(mutant_gene.iterations, 3)
        # Should have one more child gene (the connection)
        self.assertEqual(len(mutant_gene.child_genes), len(child_genes) + 1)
        # Should contain exactly one ConnectNeurons gene
        connect_neurons_genes = [g for g in mutant_gene.child_genes if isinstance(g, ConnectNeurons)]
        self.assertEqual(len(connect_neurons_genes), 1)

    def test_add_connection_no_mutation(self):
        """Test that with 0% susceptibility, no mutation occurs"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [RotateCells(RotateCells.Direction.FORWARD)]
        original_gene = CompositeGene(child_genes=child_genes, iterations=1)
        
        mutagen = AddConnectionMutagen(0.0, 0)  # 0% susceptibility
        
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should not be mutated
        self.assertEqual(mutant_gene, original_gene)

    def test_add_connection_parameters(self):
        """Test that the added connection has valid parameters"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [RotateCells(RotateCells.Direction.FORWARD)]
        original_gene = CompositeGene(child_genes=child_genes, iterations=1)
        
        mutagen = AddConnectionMutagen(1.0, 0)  # 100% susceptibility
        
        for _ in range(10):
            mutant_gene = mutagen.mutate(original_gene)
            # Find the ConnectNeurons gene (could be at any position due to random insertion)
            connect_neurons_genes = [g for g in mutant_gene.child_genes if isinstance(g, ConnectNeurons)]
            self.assertEqual(len(connect_neurons_genes), 1)
            
            connection = connect_neurons_genes[0]
            # Check that parameters are within expected range
            self.assertIsInstance(connection, ConnectNeurons)
            self.assertGreaterEqual(connection.tx_cell_index, 0)
            self.assertGreaterEqual(connection.rx_port, 0)

    def test_persist_reload(self):
        """Test that AddConnectionMutagen can be persisted and reloaded"""
        mutagen = AddConnectionMutagen(0.03, 0.04)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        with Session(engine) as session:
            reloaded = session.get(AddConnectionMutagen, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.base_susceptibility, 0.03)
            self.assertEqual(reloaded.susceptibility_log_wiggle, 0.04)
