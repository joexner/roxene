import unittest

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase
from roxene.genes import CreateInputCell, CompositeGene, RotateCells
from roxene.mutagens import AddCreateInputCell
from roxene.util import set_rng

SEED = 666


class AddCreateInputCell_test(unittest.TestCase):

    def test_add_input_cell_to_empty_composite(self):
        """
        Test that AddCreateInputCell adds a new CreateInputCell gene to an empty CompositeGene.
        
        This verifies the basic functionality: given an empty composite gene, the mutagen
        should insert exactly one CreateInputCell gene with the configured initial_value.
        """
        set_rng(default_rng(SEED))
        composite = CompositeGene(child_genes=[], iterations=1)
        
        # Create mutagen with specific initial value
        mutagen = AddCreateInputCell(
            base_susceptibility=1.0,  # 100% chance of mutation
            initial_value=0.5         # Initial value for the new input cell
        )
        
        mutant_composite = mutagen.mutate(composite)
        
        # Should have exactly 1 child gene added
        self.assertIsInstance(mutant_composite, CompositeGene)
        self.assertEqual(len(mutant_composite.child_genes), 1)
        
        # The added gene should be a CreateInputCell with the correct initial value
        added_gene = mutant_composite.child_genes[0]
        self.assertIsInstance(added_gene, CreateInputCell)
        self.assertAlmostEqual(added_gene.initial_value, 0.5, places=3)

    def test_add_input_cell_to_non_empty_composite(self):
        """
        Test that AddCreateInputCell correctly adds an input cell gene to a composite
        that already has existing child genes.
        
        The mutagen should insert the new gene at a random position while preserving
        all existing genes.
        """
        set_rng(default_rng(SEED))
        existing_gene = RotateCells(RotateCells.Direction.FORWARD)
        composite = CompositeGene(child_genes=[existing_gene], iterations=1)
        
        mutagen = AddCreateInputCell(1.0, initial_value=1.0)
        
        mutant_composite = mutagen.mutate(composite)
        
        # Should have 2 child genes now
        self.assertEqual(len(mutant_composite.child_genes), 2)
        
        # One should be the original RotateCells, one should be new CreateInputCell
        gene_types = {type(g).__name__ for g in mutant_composite.child_genes}
        self.assertEqual(gene_types, {'RotateCells', 'CreateInputCell'})

    def test_add_input_cell_default_value(self):
        """
        Test that AddCreateInputCell uses default initial_value (0.0) when not specified.
        """
        set_rng(default_rng(SEED))
        composite = CompositeGene(child_genes=[], iterations=1)
        
        # Only specify base_susceptibility, use default for initial_value
        mutagen = AddCreateInputCell(1.0)
        
        mutant_composite = mutagen.mutate(composite)
        
        added_gene = mutant_composite.child_genes[0]
        self.assertIsInstance(added_gene, CreateInputCell)
        self.assertEqual(added_gene.initial_value, 0.0)

    def test_add_input_cell_negative_value(self):
        """
        Test that AddCreateInputCell correctly handles negative initial values.
        """
        set_rng(default_rng(SEED))
        composite = CompositeGene(child_genes=[], iterations=1)
        
        mutagen = AddCreateInputCell(1.0, initial_value=-0.75)
        
        mutant_composite = mutagen.mutate(composite)
        
        added_gene = mutant_composite.child_genes[0]
        self.assertAlmostEqual(added_gene.initial_value, -0.75, places=3)

    def test_persist_reload(self):
        """
        Test that AddCreateInputCell can be saved to and loaded from the database
        with all its configuration preserved.
        """
        mutagen = AddCreateInputCell(0.03, initial_value=0.25)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        
        with Session(engine) as session:
            reloaded = session.get(AddCreateInputCell, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.base_susceptibility, 0.03)
            self.assertAlmostEqual(reloaded.initial_value, 0.25, places=3)
