import random
from random import sample, Random
from sys import argv

from roxene import Organism, CompositeGene, CreateNeuron, Neuron, ConnectNeurons, RotateCells
from .players import REQUIRED_INPUTS, REQUIRED_OUTPUTS
from .trial import Trial, Move, run_trial


class Runner(object):
    organisms: set[Organism]
    busy_organisms: set[Organism]
    completed_trials: list[Trial]

    def __init__(self, num_organisms: int, rng: Random):
        self.organisms = set()
        self.busy_organisms = set()
        self.completed_trials = list()
        self.rng = rng

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
        trial = run_trial(orgs)
        self.busy_organisms.difference_update(orgs)
        self.completed_trials.append(trial)
        return trial

    def breed_mutants(self, num_selectees, num_winners):
        selectees = sample(self.organisms, num_selectees)
        selectees.sort(lambda selectee: selectee)
        # TODO: breed mutants
