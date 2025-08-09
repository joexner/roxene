import logging
import unittest
import uuid

from numpy.random import default_rng, Generator
from sqlalchemy.orm import sessionmaker

from roxene import Organism
from roxene.tic_tac_toe import Population, Player, Trial
from util import get_engine

SEED = 1123581321

class Population_test(unittest.TestCase):

    def test_add(self):
        seshmaker = sessionmaker(get_engine())
        pop = Population()

        with seshmaker.begin() as session:
            num_orgs = 27
            for i in range(num_orgs):
                organism = Organism()
                pop.add(organism, session)
                self.assertEqual(i + 1, pop.count(session))

    def test_count(self):
        seshmaker = sessionmaker(get_engine())
        pop = Population()

        with seshmaker.begin() as session:
            # Initially should be empty
            self.assertEqual(0, pop.count(session))

            # Add organisms and verify count
            num_orgs = 8
            for i in range(num_orgs):
                organism = Organism()
                pop.add(organism, session)
                self.assertEqual(i + 1, pop.count(session))

    def test_remove(self):
        seshmaker = sessionmaker(get_engine())
        pop = Population()

        num_orgs = 9
        with seshmaker.begin() as session:
            for _ in range(num_orgs):
                organism = Organism()
                pop.add(organism, session)

            organism_id = organism.id
            initial_count = pop.count(session)

            # Remove the organism
            pop.remove(organism_id, session)

            # Verify count decreased by 1
            new_count = pop.count(session)
            self.assertEqual(initial_count - 1, new_count)

    def test_remove_nonexistent(self):
        seshmaker = sessionmaker(get_engine())
        pop = Population()

        pop_size = 8
        ids = set()

        with seshmaker.begin() as session:
            for i in range(pop_size):
                organism = Organism()
                ids.add(organism.id)
                pop.add(organism, session)

        with seshmaker.begin() as session:
            initial_count = pop.count(session)
            self.assertEqual(initial_count, pop_size)

            # Try to remove non-existent organism
            fake_id = uuid.uuid4()
            self.assertNotIn(fake_id, ids) # Not the ID of an Organism we added already
            pop.remove(fake_id, session)  # Should not raise exception

            # Count should remain the same
            self.assertEqual(initial_count, pop.count(session))

    def test_sample_basic(self):
        """Test basic sampling without idle restriction"""
        rng: Generator = default_rng(SEED)
        engine = get_engine()
        seshmaker = sessionmaker(engine)
        pop = Population()

        with seshmaker.begin() as session:
            # Add organisms
            num_orgs = 5
            for _ in range(num_orgs):
                organism = Organism()
                pop.add(organism, session)

            # Sample some organisms
            sample_size = 3
            org_ids = pop.sample(sample_size, False, rng, session)

            # Verify correct number returned
            self.assertEqual(sample_size, len(org_ids))

            # Verify all IDs are unique
            self.assertEqual(len(set(org_ids)), len(org_ids))

    def test_sample_all_organisms(self):
        """Test sampling all available organisms"""
        rng: Generator = default_rng(SEED)
        engine = get_engine()
        seshmaker = sessionmaker(engine)
        pop = Population()

        with seshmaker.begin() as session:
            # Add organisms
            num_orgs = 4
            for _ in range(num_orgs):
                organism = Organism()
                pop.add(organism, session)

            # Sample all organisms
            org_ids = pop.sample(num_orgs, False, rng, session)

            # Verify correct number returned
            self.assertEqual(num_orgs, len(org_ids))

    def test_sample_insufficient_candidates(self):
        """Test sampling when not enough organisms available"""
        rng: Generator = default_rng(SEED)
        engine = get_engine()
        seshmaker = sessionmaker(engine)
        pop = Population()

        with seshmaker.begin() as session:
            # Add only 2 organisms
            for _ in range(2):
                organism = Organism()
                pop.add(organism, session)

            # Try to sample 3 organisms - should raise ValueError
            with self.assertRaises(ValueError) as context:
                pop.sample(3, False, rng, session)

            self.assertIn("not enough candidates", str(context.exception))

    def test_add_and_sample_non_idle(self):
        rng: Generator = default_rng(SEED)
        engine = get_engine()
        seshmaker = sessionmaker(engine)
        pop: Population = Population()
        num_orgs = 100

        # Add a bunch of empty (no cells) organisms to the population
        for n in range(num_orgs):
            organism = Organism()
            with seshmaker.begin() as session:
                pop.add(organism, session)

        with seshmaker.begin() as session:
            org_ids = pop.sample(num_orgs, False, rng, session)

        self.assertEqual(num_orgs, len(org_ids))


    def test_sample_idle(self):
        rng: Generator = default_rng(SEED)
        engine = get_engine()
        seshmaker = sessionmaker(engine)
        with seshmaker() as session:
            pop = Population()
            num_orgs = 5  # Create 5 organisms
            logging.info(f"Populating {num_orgs} organisms")
            for _ in range(num_orgs):
                org = Organism()
                pop.add(org, session)
            session.commit()
            logging.info(f"Done populating organisms")

            # Start 2 trials, which will use 4 organisms (2 per trial)
            for _ in range(2):
                logging.info("Starting trial")
                orgs_ids = pop.sample(2, True, rng, session)
                player_1 = Player(organism=session.get(Organism, orgs_ids[0]), letter='X')
                player_2 = Player(organism=session.get(Organism, orgs_ids[1]), letter='O')
                trial = Trial(player_1, player_2)
                session.add(trial)
                session.commit()

            # Should have 1 idle organism left

            # Sample 1 idle organism - this should work
            pop.sample(1, True, rng, session)
            logging.info(f"Successfully sampled 1 idle organism")

            # Try to sample 2 idle organisms - this should fail since there's only 1
            try:
                pop.sample(2, True, rng, session)
                self.fail("Should not be able to sample 2 idle orgs when only 1 exists")
            except BaseException as expected_ex:
                logging.info(expected_ex)

    def test_add_same_organism_twice(self):
        """Test adding the same organism instance twice"""
        seshmaker = sessionmaker(get_engine())
        pop = Population()

        with seshmaker.begin() as session:
            organism = Organism()

            # Add organism first time
            pop.add(organism, session)
            count_after_first = pop.count(session)

            # Add same organism again - should not increase count
            pop.add(organism, session)
            count_after_second = pop.count(session)

            self.assertEqual(count_after_first, count_after_second)

    def test_remove_from_empty_population(self):
        """Test removing organism from empty population"""
        engine = get_engine()
        seshmaker = sessionmaker(engine)
        pop = Population()

        with seshmaker.begin() as session:
            # Verify population is empty
            self.assertEqual(0, pop.count(session))

            # Try to remove non-existent organism
            fake_id = uuid.uuid4()
            pop.remove(fake_id, session)

            # Population should still be empty
            self.assertEqual(0, pop.count(session))

    def test_count_after_mixed_operations(self):
        """Test count accuracy after mixed add/remove operations"""
        engine = get_engine()
        seshmaker = sessionmaker(engine)
        pop = Population()

        with seshmaker.begin() as session:
            # Start with empty population
            self.assertEqual(0, pop.count(session))

            # Add some organisms
            organisms = []
            for _ in range(3):
                organism = Organism()
                organisms.append(organism)
                pop.add(organism, session)

            self.assertEqual(3, pop.count(session))

            # Remove one organism
            pop.remove(organisms[1].id, session)
            self.assertEqual(2, pop.count(session))

            # Try removing it again
            pop.remove(organisms[1].id, session)
            self.assertEqual(2, pop.count(session))

            # Add another organism
            new_organism = Organism()
            pop.add(new_organism, session)
            self.assertEqual(3, pop.count(session))

            # Remove two more
            pop.remove(organisms[2].id, session)
            pop.remove(organisms[0].id, session)
            self.assertEqual(1, pop.count(session))
