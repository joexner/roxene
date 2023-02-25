import copy
import logging
import random
import uuid
from typing import Set, List

import tensorflow as tf
from numpy.random import default_rng, Generator

from roxene import Organism, CompositeGene, CreateNeuron, ConnectNeurons, RotateCells, random_neuron_state, \
    CreateNeuronMutagen, Mutagen, wiggle, CNLayer
from .players import REQUIRED_INPUTS, REQUIRED_OUTPUTS, OrganismPlayer
from .trial import Trial, Outcome


class Runner(object):
    organisms: Set[Organism]
    busy_organisms: Set[Organism]
    completed_trials: List[Trial]
    rng: Generator

    def __init__(self,
                 num_organisms: int,
                 num_mutagens: int,
                 neuron_shape= {"input_size": 10, "feedback_size": 5, "hidden_size": 10},
                 seed: int = None):
        self.logger = logging.getLogger(__name__)

        self.organisms = set()
        self.busy_organisms = set()
        self.completed_trials = list()

        if seed is None:
            seed = random.randint(0, 1_000_000)
        self.logger.info(f"Seed={seed}")
        self.rng: Generator = default_rng(seed)
        tf.random.set_seed(seed)

        # Monkey-patch in repeatable U(non-)UID generation a la https://stackoverflow.com/a/56757552/958533
        uuid.uuid4 = lambda: uuid.UUID(bytes=self.rng.bytes(16))

        # For each required output, make an output neuron, wire it to all the inputs and rotate it to the back
        neuron_initial_state = random_neuron_state(**neuron_shape)
        base_genotype = CompositeGene(
            genes=[
                CreateNeuron(**neuron_initial_state),
                *[ConnectNeurons(n, n) for n in range(1, len(REQUIRED_INPUTS) + 1)],
                RotateCells()],
            iterations=len(REQUIRED_OUTPUTS)
        )

        eve: Organism = Organism(REQUIRED_INPUTS, REQUIRED_OUTPUTS, base_genotype)
        self.organisms.add(eve)

        mutagen_severity_spread_log_wiggle = 3

        # Set up the mutagens
        self.mutagens: [Mutagen] = list()
        for n in range(num_mutagens):
            layer = self.rng.choice(CNLayer)
            base_susceptibility: float = wiggle(0.001, self.rng, mutagen_severity_spread_log_wiggle)
            susceptibility_log_wiggle: float = 0.01
            CreateNeuronMutagen(layer, base_susceptibility, susceptibility_log_wiggle)

        self.breed(num_organisms - 1, num_to_consider=1)

    def run_trial(self) -> Trial:
        available_organisms = list(self.organisms - self.busy_organisms)
        orgs = self.rng.choice(available_organisms, size=2, replace=False)
        self.busy_organisms.update(orgs)
        trial = Trial(OrganismPlayer(orgs[0], 'X'), OrganismPlayer(orgs[1], 'O'))
        self.logger.info(f"Starting trial {trial.id} with players {[str(o.id) for o in orgs]}")
        trial.run()
        self.logger.info(f"Finished trial {trial.id}")
        self.busy_organisms.difference_update(orgs)
        self.completed_trials.append(trial)
        return trial

    def cull(self, num_to_cull: int, num_to_compare: int = 10):
        for n in range(num_to_cull):
            selectees = self.rng.choice(list(self.organisms), size=num_to_compare, replace=False)
            selectee_scores: dict[Organism, int] = dict([(o, 0) for o in selectees])
            relevant_moves = self.get_relevant_moves(selectee_scores.keys())
            for move, organism in relevant_moves:
                selectee_scores[organism] += self.score_move(move)
            # TODO: Make stochastic
            organism_to_kill = sorted(selectee_scores.items(), key=lambda item: item[1], reverse=True)[0][0]
            self.logger.info(f"Removing organism {organism_to_kill}")
            self.organisms.discard(organism_to_kill)

    def breed(self, num_to_breed: int, num_to_consider: int = 10):
        for n in range(num_to_breed):
            selectees = self.rng.choice(list(self.organisms), size=num_to_consider, replace=False)
            if num_to_consider == 1:
                organism_to_breed = selectees[0]
            else:
                selectee_scores: dict[Organism, int] = dict([(o, 0) for o in selectees])
                relevant_moves = self.get_relevant_moves(selectee_scores.keys())
                for move, organism in relevant_moves:
                    selectee_scores[organism] += self.score_move(move)
                # TODO: Make stochastic
                organism_to_breed = sorted(selectee_scores.items(), key=lambda item: item[1], reverse=False)[0][0]
            new_organism = self.clone(organism_to_breed)
            self.logger.info(f"Bred organism {new_organism} from {organism_to_breed}")
            self.organisms.add(new_organism)

    def clone(self, organism_to_breed: Organism):
        genotype = copy.deepcopy(organism_to_breed.genotype)
        mutagens = self.mutagens
        for mutagen in mutagens:
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

    def get_relevant_moves(self, selectee_ids: Set[str]):
        for trial in self.completed_trials:
            for move in trial.moves:
                if trial.players[move.letter].organism.id in selectee_ids:
                    yield move, trial.players[move.letter].organism.id
