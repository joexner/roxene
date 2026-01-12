import unittest

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase
from roxene.genes import RotateCells, CompositeGene
from roxene.mutagens import AddRotateCells
from roxene.util import set_rng

SEED = 789


class AddRotateCells_test(unittest.TestCase):

    def setUp(self):
        set_rng(default_rng(SEED))

    def test_add_rotate_cells_forward(self):
        """Test that AddRotateCells adds a RotateCells gene with FORWARD direction"""
        composite = CompositeGene(child_genes=[], iterations=1)
        
        mutagen = AddRotateCells(0.01, RotateCells.Direction.FORWARD)
        
        mutant_composite = mutagen.mutate_CompositeGene(composite)
        
        self.assertIsInstance(mutant_composite, CompositeGene)
        self.assertEqual(len(mutant_composite.child_genes), 1)
        added_gene = mutant_composite.child_genes[0]
        self.assertIsInstance(added_gene, RotateCells)
        self.assertEqual(added_gene.direction, RotateCells.Direction.FORWARD)

    def test_add_rotate_cells_backward(self):
        """Test that AddRotateCells adds a RotateCells gene with BACKWARD direction"""
        composite = CompositeGene(child_genes=[], iterations=1)
        
        mutagen = AddRotateCells(0.01, RotateCells.Direction.BACKWARD)
        
        mutant_composite = mutagen.mutate_CompositeGene(composite)
        
        self.assertIsInstance(mutant_composite, CompositeGene)
        self.assertEqual(len(mutant_composite.child_genes), 1)
        added_gene = mutant_composite.child_genes[0]
        self.assertIsInstance(added_gene, RotateCells)
        self.assertEqual(added_gene.direction, RotateCells.Direction.BACKWARD)

    def test_add_rotate_cells_default_direction(self):
        """Test that AddRotateCells defaults to BACKWARD direction"""
        composite = CompositeGene(child_genes=[], iterations=1)
        
        mutagen = AddRotateCells(0.01)  # No direction specified
        
        mutant_composite = mutagen.mutate_CompositeGene(composite)
        
        added_gene = mutant_composite.child_genes[0]
        self.assertEqual(added_gene.direction, RotateCells.Direction.BACKWARD)

    def test_add_to_non_empty_composite(self):
        """Test that AddRotateCells adds to a CompositeGene with existing children"""
        existing_gene = RotateCells(RotateCells.Direction.FORWARD)
        composite = CompositeGene(child_genes=[existing_gene], iterations=1)
        
        mutagen = AddRotateCells(0.01, RotateCells.Direction.BACKWARD)
        
        mutant_composite = mutagen.mutate_CompositeGene(composite)
        
        self.assertEqual(len(mutant_composite.child_genes), 2)
        # One should be FORWARD (existing), one should be BACKWARD (new)
        directions = {gene.direction for gene in mutant_composite.child_genes}
        self.assertEqual(directions, {RotateCells.Direction.FORWARD, RotateCells.Direction.BACKWARD})

    def test_persist_reload(self):
        """Test that AddRotateCells can be persisted and reloaded"""
        mutagen = AddRotateCells(0.025, RotateCells.Direction.FORWARD)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        with Session(engine) as session:
            reloaded = session.get(AddRotateCells, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.base_susceptibility, 0.025)
            self.assertEqual(reloaded.direction, RotateCells.Direction.FORWARD)
