import logging
import uuid
from typing import Set, List

import torch
from numpy.random import default_rng, Generator
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from .move import Move
from .outcome import Outcome
from .players import REQUIRED_INPUTS, REQUIRED_OUTPUTS, Player
from .population import Population
from .trial import Trial
from ..gene import Gene
from ..genes.composite_gene import CompositeGene
from ..genes.connect_neurons import ConnectNeurons
from ..genes.create_neuron import CreateNeuron
from ..genes.rotate_cells import RotateCells
from ..mutagens.create_neuron_mutagen import CreateNeuronMutagen, CNLayer
from ..mutagen import Mutagen
from ..util import wiggle
from ..organism import Organism
from ..util import random_neuron_state

logger = logging.getLogger(__name__)

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
        self.mutagens = list()
        self.rng: Generator = default_rng(seed)
        self.sessionmaker = sessionmaker(engine)

        logger.info(f"Seed={seed}")
        torch.manual_seed(seed)
        uuid.uuid4 = lambda: uuid.UUID(bytes=self.rng.bytes(16))

    def populate(self,
                 num_organisms: int,
                 neuron_shape = None,
                 ):
        if neuron_shape is None:
            neuron_shape = {"input_size": 10, "feedback_size": 5, "hidden_size": 10}
        for _ in range(num_organisms):
            with (self.sessionmaker.begin() as session):
                # Build the genotype
                # For each required output, make an output neuron, wire it to all the inputs and rotate it to the back
                child_genes = list()
                for n in range(len(REQUIRED_OUTPUTS)):
                    child_genes.extend([
                        CreateNeuron(**random_neuron_state(**neuron_shape, rng=self.rng)),
                        *[ConnectNeurons(n, n) for n in range(1, len(REQUIRED_INPUTS) + 1)],
                        RotateCells()
                    ])
                base_genotype = CompositeGene(child_genes)

                # Create the organism and add it to the population
                new_organism: Organism = Organism(REQUIRED_INPUTS, REQUIRED_OUTPUTS, base_genotype)
                self.population.add(new_organism, session)

    def add_mutagens(self, num_mutagens):
        mutagen_severity_spread_log_wiggle = 3
        # Persist all mutagens to the database in a single transaction
        with self.sessionmaker.begin() as session:
            for n in range(num_mutagens):
                layer = self.rng.choice(CNLayer)
                base_susceptibility: float = wiggle(0.001, self.rng, mutagen_severity_spread_log_wiggle)
                susceptibility_log_wiggle: float = 0.01
                new_mutagen = CreateNeuronMutagen(layer, base_susceptibility, susceptibility_log_wiggle)
                self.mutagens.append(new_mutagen)
                session.add(new_mutagen)

    def start_trial(self) -> Trial:
        with (self.sessionmaker(expire_on_commit=False) as session):
            org_ids: [uuid.UUID] = self.population.sample(2, True, self.rng, session)
            orgs: List[Organism] = [session.get(Organism, oid) for oid in org_ids]
            p1, p2 = Player(orgs[0]), Player(orgs[1])

            trial = Trial(p1, p2)

            session.add(trial)
            session.commit()

            return trial


    def complete_trial(self, trial: Trial):
        with self.sessionmaker.begin() as session:
            session.merge(trial)


    def cull(self, num_to_cull: int, num_to_compare: int = 10):
        for n in range(num_to_cull):
            session: Session
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
                    index_to_kill = int((rand ** 2) * num_to_compare)  # Squaring the random number to skew it towards the lower end
                    logger.info(f"Removing organism at index {index_to_kill} of {num_to_compare}")

                    organism_id_to_kill = sorted_orgs_and_scores[index_to_kill][0]

                logger.info(f"Removing organism {organism_id_to_kill}")
                self.population.remove(organism_id_to_kill, session)


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
                    index_to_clone = int((rand ** 2) * num_to_consider)  # Squaring the random number to skew it towards the lower end
                    logger.info(f"Cloning organism at index {index_to_clone} of {num_to_consider}")

                    organism_id_to_breed = sorted_orgs_and_scores[index_to_clone][0]
                    logger.info(f"Cloning organism {organism_id_to_breed}")

                new_organism = self.clone(organism_id_to_breed, session)
                logger.info(f"Bred organism {new_organism.id} from {organism_id_to_breed}")
                self.population.add(new_organism, session)

    def clone(self, organism_id: uuid, session: Session, mutate=True) -> Organism:
        original_genotype = session.scalar(select(Gene)
                             .join_from(Organism, Gene)
                             .where(Organism.id == organism_id))
        clone_genotype = original_genotype
        if mutate:
            for mutagen in self.mutagens:
                clone_genotype = mutagen.mutate(clone_genotype, self.rng)
        return Organism(REQUIRED_INPUTS, REQUIRED_OUTPUTS, clone_genotype)

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