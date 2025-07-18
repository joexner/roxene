import copy
import logging
import random
import uuid
from sqlalchemy import Engine, select, join
from sqlalchemy.orm import Session, sessionmaker
from typing import Set, List

import tensorflow as tf
from numpy.random import default_rng, Generator

from .population import Population
from .move import Move
from ..organism import Organism
from ..genes import CompositeGene, CreateNeuron, ConnectNeurons, RotateCells
from ..util import  random_neuron_state
from ..mutagens import CreateNeuronMutagen, Mutagen, wiggle, CNLayer
from .players import REQUIRED_INPUTS, REQUIRED_OUTPUTS, Player
from .trial import Trial
from .outcome import Outcome


class Environment(object):
    """
    Represents the environment in which organisms interact, evolve, and compete.

    This class manages the population of organisms, their mutations, trials, and evolutionary processes.
    It also handles interactions with the database and random number generation for reproducibility.

    Attributes:
        population (Population): The population of organisms in the environment.
        mutagens (List[Mutagen]): A list of mutagens used to modify genotypes.
        rng (Generator): A random number generator for reproducibility.
        engine (Engine): The SQLAlchemy engine for database interactions.
    """

    population: Population
    mutagens: [Mutagen]
    rng: Generator
    sessionmaker: sessionmaker

    def __init__(self,
                 seed: int,
                 engine: Engine,
                 ):
        self.population = Population()
        self.sessionmaker = sessionmaker(engine)
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Seed={seed}")

        # Set up the RNG
        self.rng: Generator = default_rng(seed)
        tf.random.set_seed(seed)
        uuid.uuid4 = lambda: uuid.UUID(bytes=self.rng.bytes(16))

    def populate(self,
                 num_organisms: int,
                 neuron_shape={"input_size": 10, "feedback_size": 5, "hidden_size": 10},
                 ):
        for _ in range(num_organisms):
            with self.sessionmaker.begin() as session:
                # Create a random initial state for the neuron
                neuron_initial_state = random_neuron_state(
                    neuron_shape["input_size"],
                    neuron_shape["feedback_size"],
                    neuron_shape["hidden_size"],
                    self.rng)

                # Build the genotype
                # For each required output, make an output neuron, wire it to all the inputs and rotate it to the back
                base_genotype = CompositeGene(
                    genes=[
                        CreateNeuron(**neuron_initial_state),
                        *[ConnectNeurons(n, n) for n in range(1, len(REQUIRED_INPUTS) + 1)],
                        RotateCells()],
                    iterations=len(REQUIRED_OUTPUTS)
                )

                # Create the organism and add it to the population
                new_organism: Organism = Organism(REQUIRED_INPUTS, REQUIRED_OUTPUTS, base_genotype)
                self.population.add(new_organism, session)

    def add_mutagens(self, num_mutagens):
        mutagen_severity_spread_log_wiggle = 3
        self.mutagens: [Mutagen] = list()
        for n in range(num_mutagens):
            layer = self.rng.choice(CNLayer)
            base_susceptibility: float = wiggle(0.001, self.rng, mutagen_severity_spread_log_wiggle)
            susceptibility_log_wiggle: float = 0.01
            new_mutagen = CreateNeuronMutagen(layer, base_susceptibility, susceptibility_log_wiggle)
            self.mutagens.append(new_mutagen)

    def start_trial(self) -> Trial:
        with (self.sessionmaker() as session):
            org_ids: [uuid.UUID] = self.population.sample(2, True, self.rng, session)
            orgs: List[Organism] = [session.get(Organism, oid) for oid in org_ids]
            p1, p2 = Player(orgs[0]), Player(orgs[1])
            self.logger.info(f'Starting a trial with {p1} and {p2}')

            trial = Trial(p1, p2)
            self.logger.info(f"Started trial between {orgs[0]} and {orgs[1]}")

            session.add(trial)
            session.commit()

            # Committing "expires" everything from the session, so refresh the trial before returning it
            session.refresh(trial)

            return trial


    def complete_trial(self, trial: Trial):
        with self.sessionmaker.begin() as session:
            session.merge(trial)


    def cull(self, num_to_cull: int, num_to_compare: int = 10):
        for n in range(num_to_cull):
            with self.sessionmaker.begin() as session:
                selectee_ids = self.population.sample(num_to_compare, False, self.rng, session)
                if num_to_compare == 1:
                    organism_id_to_kill = selectee_ids[0]
                else:
                    selectee_scores: dict[uuid.UUID, int] = dict([(oid, 0) for oid in selectee_ids])
                    relevant_moves = self.get_relevant_moves(selectee_ids, session)
                    for move in relevant_moves:
                        selectee_scores[move.organism_id] += self.score_move(move)

                    # Put the Organisms with the highest scores at the front of the list
                    sorted_orgs_and_scores = sorted(selectee_scores.items(), key=lambda item: item[1], reverse=True)

                    rand = self.rng.random()
                    index_to_kill = int((rand ^ 2) * num_to_compare)  # Squaring the random number to skew it towards the lower end
                    self.logger.info(f"Removing organism at index {index_to_kill} of {num_to_compare}")

                    organism_id_to_kill = sorted_orgs_and_scores[index_to_kill][0]
                    self.logger.info(f"Removing organism {organism_id_to_kill}")

                organism_to_kill: Organism = session.get(Organism, organism_id_to_kill)
                self.population.remove(organism_to_kill, session)


    def breed(self, num_to_breed: int, num_to_consider: int = 10):
        for n in range(num_to_breed):
            with self.sessionmaker.begin() as session:
                selectee_ids = self.population.sample(num_to_consider, False, self.rng, session)
                if num_to_consider == 1:
                    organism_id_to_breed = selectee_ids[0]
                else:
                    selectee_scores: dict[uuid.UUID, int] = dict([(oid, 0) for oid in selectee_ids])
                    relevant_moves = self.get_relevant_moves(selectee_ids, session)
                    for move in relevant_moves:
                        selectee_scores[move.organism_id] += self.score_move(move)

                    # Put the Organisms with the lowest scores at the front of the list
                    sorted_orgs_and_scores = sorted(selectee_scores.items(), key=lambda item: item[1], reverse=False)

                    rand = self.rng.random()
                    index_to_clone = int((rand ^ 2) * num_to_consider)  # Squaring the random number to skew it towards the lower end
                    self.logger.info(f"Cloning organism at index {index_to_clone} of {num_to_consider}")

                    organism_id_to_breed = sorted_orgs_and_scores[index_to_clone][0]
                    self.logger.info(f"Cloning organism {organism_id_to_breed}")

                old_organism = session.get(Organism, organism_id_to_breed)
                new_organism = self.clone(old_organism)
                self.logger.info(f"Bred organism {new_organism} from {old_organism}")
                self.population.add(new_organism, session)

    def clone(self, organism_to_breed: Organism):
        genotype = copy.deepcopy(organism_to_breed.genotype)
        for mutagen in self.mutagens:
            genotype = mutagen.mutate(genotype, self.rng)
        return Organism(REQUIRED_INPUTS, REQUIRED_OUTPUTS, genotype)

    def score_move(self, move):
        score = 0
        for outcome in move.outcomes:  # Zero-sum is insufficiently motivational
            if outcome is Outcome.WIN:          score += 100
            if outcome is Outcome.TIE:          score += 10
            if outcome is Outcome.VALID_MOVE:   score += 1
            if outcome is Outcome.LOSE:         score -= 10
            if outcome is Outcome.INVALID_MOVE: score -= 50
            if outcome is Outcome.TIMEOUT:      score -= 100
        return score

    def get_relevant_moves(self, selectee_ids: Set[str], session: Session):
        return session.scalars(
            select(Move)
            .join(Organism)
            .where(Organism.id.in_(selectee_ids)))