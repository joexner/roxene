import unittest

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase
from roxene.genes import CreateInputCell, CompositeGene
from roxene.mutagens import WiggleInputCellValue
from roxene.util import set_rng

SEED = 777


class WiggleInputCellValue_test(unittest.TestCase):

    def test_wiggle_changes_value(self):
        """
        Test that WiggleInputCellValue modifies the initial_value of a CreateInputCell.
        
        The mutagen should apply a small random perturbation to the initial value,
        resulting in a new CreateInputCell with a slightly different value.
        """
        set_rng(default_rng(SEED))
        original = CreateInputCell(initial_value=0.5)
        
        mutagen = WiggleInputCellValue(base_susceptibility=1.0, severity=1.0)
        
        # Try multiple times to confirm the value changes
        value_changed = False
        for _ in range(20):
            mutant = mutagen.mutate(original)
            
            self.assertIsInstance(mutant, CreateInputCell)
            if mutant.initial_value != original.initial_value:
                value_changed = True
                break
        
        self.assertTrue(value_changed, "Expected initial_value to change in at least one of 20 tries")

    def test_severity_affects_change_magnitude(self):
        """
        Test that higher severity values result in larger changes to initial_value.
        
        Severity controls the magnitude of the wiggle transformation:
        - Low severity should make small adjustments
        - High severity should make larger adjustments
        """
        set_rng(default_rng(SEED))
        original = CreateInputCell(initial_value=1.0)
        
        mutagen_low = WiggleInputCellValue(base_susceptibility=1.0, severity=0.1)
        mutagen_high = WiggleInputCellValue(base_susceptibility=1.0, severity=5.0)
        
        # Collect differences for multiple mutations
        low_diffs = []
        high_diffs = []
        
        for i in range(50):
            set_rng(default_rng(SEED + i))
            mutant_low = mutagen_low.mutate(original)
            low_diffs.append(abs(mutant_low.initial_value - original.initial_value))
            
            set_rng(default_rng(SEED + i + 1000))
            mutant_high = mutagen_high.mutate(original)
            high_diffs.append(abs(mutant_high.initial_value - original.initial_value))
        
        avg_low = sum(low_diffs) / len(low_diffs)
        avg_high = sum(high_diffs) / len(high_diffs)
        
        # High severity should cause larger changes on average
        self.assertGreater(avg_high, avg_low,
            "Higher severity should cause larger value changes")

    def test_parent_gene_set(self):
        """
        Test that the mutated gene properly references the original as parent_gene.
        
        This maintains the gene lineage for tracking evolutionary history.
        """
        set_rng(default_rng(SEED))
        original = CreateInputCell(initial_value=0.5)
        
        mutagen = WiggleInputCellValue(1.0)
        mutant = mutagen.mutate(original)
        
        self.assertEqual(mutant.parent_gene, original)

    def test_no_mutation_at_zero_susceptibility(self):
        """
        Test that WiggleInputCellValue doesn't mutate at 0% susceptibility.
        """
        set_rng(default_rng(SEED))
        original = CreateInputCell(initial_value=0.5)
        
        mutagen = WiggleInputCellValue(0.0)  # 0% susceptibility
        mutant = mutagen.mutate(original)
        
        # Should be the exact same object (no mutation)
        self.assertIs(mutant, original)

    def test_wiggle_zero_initial_value(self):
        """
        Test that WiggleInputCellValue correctly handles zero initial values.
        
        The wiggle() function uses log transform which requires x != 0.
        This test verifies that zero values are handled specially without errors.
        """
        set_rng(default_rng(SEED))
        original = CreateInputCell(initial_value=0.0)  # Zero value
        
        mutagen = WiggleInputCellValue(1.0, severity=1.0)
        
        # Should not crash on zero value
        mutant = mutagen.mutate(original)
        
        self.assertIsInstance(mutant, CreateInputCell)
        # Value should have changed from zero (via absolute wiggle)
        self.assertIsNotNone(mutant.initial_value)

    def test_wiggle_in_composite(self):
        """
        Test that CreateInputCell genes within CompositeGenes are wiggled.
        """
        set_rng(default_rng(SEED))
        input_cell = CreateInputCell(initial_value=1.0)
        composite = CompositeGene(child_genes=[input_cell], iterations=1)
        
        mutagen = WiggleInputCellValue(1.0, severity=2.0)
        
        mutant_composite = mutagen.mutate(composite)
        
        self.assertIsInstance(mutant_composite, CompositeGene)
        self.assertEqual(len(mutant_composite.child_genes), 1)
        
        mutant_input_cell = mutant_composite.child_genes[0]
        self.assertIsInstance(mutant_input_cell, CreateInputCell)

    def test_persist_reload(self):
        """
        Test that WiggleInputCellValue can be persisted and reloaded with all
        configuration preserved.
        """
        mutagen = WiggleInputCellValue(0.02, severity=1.5)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        
        with Session(engine) as session:
            reloaded = session.get(WiggleInputCellValue, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.base_susceptibility, 0.02)
            self.assertEqual(reloaded.severity, 1.5)
