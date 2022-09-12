import logging
import random
import uuid
from random import Random
from typing import Set, List

import tensorflow as tf

from roxene import Organism, CompositeGene, CreateNeuron, Neuron, ConnectNeurons, RotateCells
from .players import REQUIRED_INPUTS, REQUIRED_OUTPUTS, OrganismPlayer
from .trial import Trial, Outcome


class Runner(object):
    organisms: Set[Organism]
    busy_organisms: Set[Organism]
    completed_trials: List[Trial]

    def __init__(self, num_organisms: int, seed: int = None):
        self.logger = logging.getLogger(__name__)

        self.organisms = set()
        self.busy_organisms = set()
        self.completed_trials = list()

        if seed is None:
            seed = random.randint()
        self.logger.info(f"Seed={seed}")
        self.rng = Random(seed)
        tf.random.set_seed(seed)

        # Monkey-patch in repeatable U(non-)UID generation a la https://stackoverflow.com/a/56757552/958533
        uuid.uuid4 = lambda: uuid.UUID(int=self.rng.getrandbits(128))

        # Make an output neuron for each required output, wire it to all the inputs and rotate it to the back
        neuron_initial_state = Neuron.random_neuron_state(input_size=20, feedback_size=5, hidden_size=10)
        base_genotype = CompositeGene(
            genes=[
                CreateNeuron(**neuron_initial_state),
                *[ConnectNeurons(n, n) for n in range(1, len(REQUIRED_INPUTS) + 1)],
                RotateCells()],
            iterations=len(REQUIRED_OUTPUTS)
        )

        # Just make a bunch of clones for now
        for org_num in range(num_organisms):
            organism = Organism(REQUIRED_INPUTS, REQUIRED_OUTPUTS, base_genotype)
            self.organisms.add(organism)

    def run_trial(self) -> Trial:
        orgs = self.rng.sample(*[self.organisms - self.busy_organisms], 2)
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
            selectees = self.rng.sample(self.organisms, num_to_compare)
            selectee_scores: dict[Organism, int] = dict([(o, 0) for o in selectees])
            for move, organism in self.get_relevant_moves(selectee_scores.keys()):
                selectee_scores[organism] += self.score_move(move)
            # Just kill the lamest of the bunch for now
            # TODO: Make stochastic
            organism_to_kill: Organism = sorted(selectee_scores.items(), key=lambda item: item[1], reverse=True)[0][0]
            self.logger.info(f"Removing organism {organism_to_kill}")
            self.organisms.discard(organism_to_kill)

    def breed(self, num_to_breed: int, num_to_consider: int = 10):
        for n in range(num_to_breed):
            selectees = self.rng.sample(self.organisms, num_to_consider)
            selectee_scores: dict[Organism, int] = dict([(o, 0) for o in selectees])
            for move, organism in self.get_relevant_moves(selectee_scores.keys()):
                selectee_scores[organism] += self.score_move(move)
            # Asexual reproduction, for now
            organism_to_breed: Organism = sorted(selectee_scores.items(), key=lambda item: item[1], reverse=False)[0][0]
            new_organism = self.clone(organism_to_breed)
            self.logger.info(f"Bred organism {new_organism} from {organism_to_breed}")
            self.organisms.add(new_organism)

    def clone(self, organism_to_breed: Organism):
        # TODO: Mutate
        return Organism(REQUIRED_INPUTS, REQUIRED_OUTPUTS, organism_to_breed.genotype)

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
