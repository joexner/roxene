import copy
import logging
import random
import uuid
from sqlalchemy import Engine, select, join
from sqlalchemy.orm import Session
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

    population: Population
    mutagens: [Mutagen]
    rng: Generator
    engine: Engine

    def __init__(self,
                 seed: int,
                 engine: Engine,
                 ):
        self.population = Population()
        self.engine = engine
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

        with Session(self.engine) as session:
            for _ in range(num_organisms):
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
                session.commit()

    def add_mutagens(self, num_mutagens):

        mutagen_severity_spread_log_wiggle = 3

        self.mutagens: [Mutagen] = list()
        for n in range(num_mutagens):
            layer = self.rng.choice(CNLayer)
            base_susceptibility: float = wiggle(0.001, self.rng, mutagen_severity_spread_log_wiggle)
            susceptibility_log_wiggle: float = 0.01
            new_mutagen = CreateNeuronMutagen(layer, base_susceptibility, susceptibility_log_wiggle)
            self.mutagens.append(new_mutagen)

    def run_trial(self):
        with Session(self.engine) as session:
            trial = self.population.start_trial(session)
            self.logger.info(
                f"Starting trial between {trial.participants[0].organism} and {trial.participants[1].organism}")
            trial.run()
            self.logger.info(f"Done trial, {len(trial.moves)} moves")
            self.population.complete_trial(trial, session)
            return trial.id


    def cull(self, num_to_cull: int, num_to_compare: int = 10):
        with Session(self.engine) as session:

            for n in range(num_to_cull):
                selectees = self.population.sample(num_to_compare, False, session)
                selectee_scores: dict[Organism, int] = dict([(o, 0) for o in selectees])
                relevant_moves = self.get_relevant_moves([o.id for o in selectees], session)
                if relevant_moves is not None:
                    for move, organism in relevant_moves:
                        selectee_scores[organism] += self.score_move(move)

                sorted_orgs_and_scores = sorted(selectee_scores.items(), key=lambda item: item[1], reverse=True)
                index_to_kill = int(abs(self.rng.normal()))
                organism_to_kill: Organism = sorted_orgs_and_scores[index_to_kill][0]
                self.logger.info(f"Culling index {index_to_kill} of")

                self.logger.info(f"Removing organism {organism_to_kill}")
                self.population.remove(organism_to_kill, session)


    def breed(self, num_to_breed: int, num_to_consider: int = 10):
        with Session(self.engine) as session:

            for n in range(num_to_breed):
                selectees = self.population.sample(num_to_consider, True, session)
                if num_to_consider == 1:
                    organism_to_breed = selectees[0]
                else:
                    selectee_scores: dict[Organism, int] = dict([(o, 0) for o in selectees])
                    for move, organism in self.get_relevant_moves(selectee_scores.keys()):
                        selectee_scores[organism] += self.score_move(move)

                    sorted_orgs_and_scores = sorted(selectee_scores.items(), key=lambda item: item[1], reverse=False)

                    organism_to_breed: Organism = sorted_orgs_and_scores[0][0]

                new_organism = self.clone(organism_to_breed)
                self.logger.info(f"Bred organism {new_organism} from {organism_to_breed}")
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
