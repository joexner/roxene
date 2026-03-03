import unittest
from typing import List

from numpy.random import default_rng

from roxene import Gene, random_neuron_state
from roxene.genes import CompositeGene, RotateCells, CreateNeuron, ConnectNeurons, CreateInputCell
from roxene.mutagens import DuplicateGene
from roxene.util import set_rng

SEED = 321


class DuplicateGene_test(unittest.TestCase):

    def setUp(self):
        set_rng(default_rng(SEED))

    def test_get_new_gene_empty_composite(self):
        """Test that get_new_gene returns None for empty CompositeGene"""
        child_genes: List[Gene] = []
        original_gene = CompositeGene(child_genes=child_genes, iterations=1)
        
        mutagen = DuplicateGene()
        
        result = mutagen.get_new_gene(original_gene)
        
        self.assertIsNone(result)

    def test_get_new_gene_random_selection(self):
        """Test that get_new_gene randomly selects from available genes with diverse gene types"""
        original_child_genes: List[Gene] = [
            CreateNeuron(**random_neuron_state(input_size=2, feedback_size=2, hidden_size=2)),
            ConnectNeurons(tx_cell_index=0, rx_input_port=1),
            CreateInputCell(initial_value=0.5),
            RotateCells(RotateCells.Direction.FORWARD),
            RotateCells(RotateCells.Direction.BACKWARD),
            CompositeGene(child_genes=[
                RotateCells(RotateCells.Direction.FORWARD),
                ConnectNeurons(tx_cell_index=0, rx_input_port=1)
            ])
        ]
        original_gene = CompositeGene(child_genes=original_child_genes)
        
        mutagen = DuplicateGene()
        
        # Collect selections over multiple runs
        selected_gene_ids = set()
        for seed in range(200):
            mutant: Gene = mutagen.get_new_gene(original_gene)
            # Verify the returned gene is one of the original genes (same object identity)
            self.assertIn(mutant, original_child_genes)
            selected_gene_ids.add(mutant.id)
        
        # All genes should have been selected at some point
        original_child_ids = {gene.id for gene in original_child_genes}
        self.assertEqual(original_child_ids, selected_gene_ids,
                        f"All genes should be selected. Expected: {original_child_ids}, Got: {selected_gene_ids}")
