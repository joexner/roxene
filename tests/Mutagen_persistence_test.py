import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, Mutagen
from roxene.mutagens.create_neuron_mutagen import CreateNeuronMutagen, CNLayer
from roxene.mutagens.composite_gene_split_mutagen import CompositeGeneSplitMutagen


class Mutagen_persistence_test(unittest.TestCase):

    def test_create_neuron_mutagen_persistence(self):
        """Test that CreateNeuronMutagen can be persisted and reloaded"""
        # Create a mutagen with specific parameters
        layer = CNLayer.input_hidden
        base_sus = 0.005
        wiggle = 0.02
        
        mutagen = CreateNeuronMutagen(
            layer_to_mutate=layer,
            base_susceptibility=base_sus,
            susceptibility_log_wiggle=wiggle
        )
        
        mutagen_id = mutagen.id
        
        # Create in-memory database
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        
        # Save to database
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        
        # Reload from database
        with Session(engine) as session:
            reloaded = session.get(CreateNeuronMutagen, mutagen_id)
            
            # Verify all properties are preserved
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.layer_to_mutate, layer)
            self.assertEqual(reloaded.base_susceptibility, base_sus)
            self.assertEqual(reloaded.susceptibility_log_wiggle, wiggle)
            
            # Verify susceptibilities dict is initialized (not persisted)
            self.assertIn(None, reloaded.susceptibilities)
            self.assertEqual(reloaded.susceptibilities[None], base_sus)

    def test_composite_gene_split_mutagen_persistence(self):
        """Test that CompositeGeneSplitMutagen can be persisted and reloaded"""
        base_sus = 0.015
        wiggle = 0.03
        
        mutagen = CompositeGeneSplitMutagen(
            base_susceptibility=base_sus,
            susceptibility_log_wiggle=wiggle
        )
        
        mutagen_id = mutagen.id
        
        # Create in-memory database
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        
        # Save to database
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        
        # Reload from database
        with Session(engine) as session:
            reloaded = session.get(CompositeGeneSplitMutagen, mutagen_id)
            
            # Verify all properties are preserved
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.base_susceptibility, base_sus)
            self.assertEqual(reloaded.susceptibility_log_wiggle, wiggle)
            
            # Verify susceptibilities dict is initialized
            self.assertIn(None, reloaded.susceptibilities)
            self.assertEqual(reloaded.susceptibilities[None], base_sus)

    def test_multiple_mutagens_persistence(self):
        """Test persisting multiple mutagens of different types"""
        mutagen1 = CreateNeuronMutagen(CNLayer.feedback_initial_value, 0.001, 0.01)
        mutagen2 = CreateNeuronMutagen(CNLayer.hidden_output, 0.002, 0.015)
        mutagen3 = CompositeGeneSplitMutagen(0.01, 0.02)
        
        mutagen1_id = mutagen1.id
        mutagen2_id = mutagen2.id
        mutagen3_id = mutagen3.id
        
        # Create in-memory database
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        
        # Save all to database
        with Session(engine) as session:
            session.add_all([mutagen1, mutagen2, mutagen3])
            session.commit()
        
        # Reload and verify
        with Session(engine) as session:
            reloaded1 = session.get(CreateNeuronMutagen, mutagen1_id)
            reloaded2 = session.get(CreateNeuronMutagen, mutagen2_id)
            reloaded3 = session.get(CompositeGeneSplitMutagen, mutagen3_id)
            
            self.assertIsNotNone(reloaded1)
            self.assertIsNotNone(reloaded2)
            self.assertIsNotNone(reloaded3)
            
            self.assertEqual(reloaded1.layer_to_mutate, CNLayer.feedback_initial_value)
            self.assertEqual(reloaded2.layer_to_mutate, CNLayer.hidden_output)
            
            # Verify each has its own independent susceptibilities dict
            self.assertIsInstance(reloaded1.susceptibilities, dict)
            self.assertIsInstance(reloaded2.susceptibilities, dict)
            self.assertIsInstance(reloaded3.susceptibilities, dict)

    def test_mutagen_str_method(self):
        """Test the __str__ method returns expected format"""
        mutagen = CreateNeuronMutagen(CNLayer.input_initial_value, 0.001, 0.01)
        
        str_repr = str(mutagen)
        
        # Should be in format M-{last 7 chars of UUID}
        self.assertTrue(str_repr.startswith("M-"))
        self.assertEqual(len(str_repr), 9)  # "M-" + 7 chars
        
        # Last 7 chars should match the UUID
        expected_suffix = str(mutagen.id)[-7:]
        self.assertEqual(str_repr, f"M-{expected_suffix}")

    def test_polymorphic_loading(self):
        """Test that mutagens can be loaded polymorphically as base Mutagen class"""
        mutagen1 = CreateNeuronMutagen(CNLayer.input_hidden, 0.003, 0.01)
        mutagen2 = CompositeGeneSplitMutagen(0.02, 0.015)
        
        mutagen1_id = mutagen1.id
        mutagen2_id = mutagen2.id
        
        # Create in-memory database
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        
        # Save to database
        with Session(engine) as session:
            session.add_all([mutagen1, mutagen2])
            session.commit()
        
        # Reload as base Mutagen class to test polymorphism
        with Session(engine) as session:
            reloaded1 = session.get(Mutagen, mutagen1_id)
            reloaded2 = session.get(Mutagen, mutagen2_id)
            
            # Should be loaded as specific subclass types
            self.assertIsInstance(reloaded1, CreateNeuronMutagen)
            self.assertIsInstance(reloaded2, CompositeGeneSplitMutagen)
            
            # And should still have correct properties
            self.assertEqual(reloaded1.layer_to_mutate, CNLayer.input_hidden)
            self.assertEqual(reloaded2.base_susceptibility, 0.02)
