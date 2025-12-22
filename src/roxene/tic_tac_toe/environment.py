import logging
import uuid
from typing import Set, List

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
from ..mutagen import Mutagen
from ..mutagens.wiggle_create_neuron import WiggleCreateNeuron, CNLayer
from ..organism import Organism
from ..util import random_neuron_state
from ..util import wiggle, get_rng

logger = logging.getLogger(__name__)

class Environment(object):
    """
    Represents the environment in which organisms interact, evolve, and compete.

    This class manages the population of organisms, their mutations, trials, and evolutionary processes.
    It also handles interactions with the database and random number generation for reproducibility.

    Attributes:
        population (Population): The population of organisms in the environment.
        mutagens (List[Mutagen]): A list of mutagens used to modify genotypes.
        sessionmaker (sessionmaker): The SQLAlchemy ORM sessionmaker for database interactions.
    """

    population: Population
    sessionmaker: sessionmaker

    def __init__(self, engine: Engine):
        self.population = Population()
        self.sessionmaker = sessionmaker(engine)

    def populate(self, num_organisms: int, neuron_shape = None):
        if neuron_shape is None:
            neuron_shape = {"input_size": 10, "feedback_size": 5, "hidden_size": 10}
        for _ in range(num_organisms):
            with (self.sessionmaker.begin() as session):
                # Build the genotype
                # For each required output, make an output neuron, wire it to all the inputs and rotate it to the back
                child_genes = list()
                for _ in REQUIRED_OUTPUTS:
                    child_genes.extend([
                        CreateNeuron(**random_neuron_state(**neuron_shape)),
                        *[ConnectNeurons(n, n) for n in range(1, len(REQUIRED_INPUTS) + 1)],
                        RotateCells()
                    ])
                base_genotype = CompositeGene(child_genes)

                # Create the organism and add it to the population
                new_organism: Organism = Organism(REQUIRED_INPUTS, REQUIRED_OUTPUTS, base_genotype)
                self.population.add(new_organism, session)

    def count_organisms(self) -> int:
        with self.sessionmaker() as session:
            return self.population.count(session)

    def add_mutagens(self, num_mutagens):
        mutagen_severity_spread_log_wiggle = 3
        for n in range(num_mutagens):
            with self.sessionmaker.begin() as session:
                layer = get_rng().choice(CNLayer)
                base_susceptibility: float = wiggle(0.001, mutagen_severity_spread_log_wiggle)
                susceptibility_log_wiggle: float = 0.01
                new_mutagen = WiggleCreateNeuron(layer, base_susceptibility, susceptibility_log_wiggle)
                session.add(new_mutagen)

    def add_mutagen(self, new_mutagen: Mutagen):
        with self.sessionmaker.begin() as session:
            session.add(new_mutagen)

    def get_mutagens(self, session: Session):
        return session.scalars(select(Mutagen)).all()

    def start_trial(self) -> Trial:
        with (self.sessionmaker(expire_on_commit=False) as session):
            org_ids: List[uuid.UUID] = self.population.sample(2, True, session)
            p1 = Player(session.get(Organism, org_ids[0]))
            p2 = Player(session.get(Organism, org_ids[1]))
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
                selectee_ids = self.population.sample(num_to_compare, False, session)
                if num_to_compare == 1:
                    organism_id_to_kill = selectee_ids[0]
                else:
                    selectee_scores: dict[uuid.UUID, int] = dict([(oid, 0) for oid in selectee_ids])
                    relevant_moves = self.get_relevant_moves(selectee_ids, session)
                    for move in relevant_moves:
                        selectee_scores[move.organism_id] += self.score_move(move)

                    # Put the Organisms with the highest scores at the front of the list
                    sorted_orgs_and_scores = sorted(selectee_scores.items(), key=lambda item: item[1], reverse=True)

                    rand = get_rng().random()
                    index_to_kill = int((rand ** 2) * num_to_compare)  # Squaring the random number to skew it towards the lower end
                    logger.info(f"Removing organism at index {index_to_kill} of {num_to_compare}")

                    organism_id_to_kill = sorted_orgs_and_scores[index_to_kill][0]

                logger.info(f"Removing organism {organism_id_to_kill}")
                self.population.remove(organism_id_to_kill, session)

    def breed(self, num_to_breed: int, num_to_consider: int = 10):
        for n in range(num_to_breed):
            with self.sessionmaker.begin() as session:
                selectee_ids = self.population.sample(num_to_consider, False, session)
                if num_to_consider == 1:
                    organism_id_to_breed = selectee_ids[0]
                else:
                    selectee_scores: dict[uuid.UUID, int] = dict([(oid, 0) for oid in selectee_ids])
                    relevant_moves = self.get_relevant_moves(selectee_ids, session)
                    for move in relevant_moves:
                        selectee_scores[move.organism_id] += self.score_move(move)

                    # Put the Organisms with the lowest scores at the front of the list
                    sorted_orgs_and_scores = sorted(selectee_scores.items(), key=lambda item: item[1], reverse=False)

                    # Squaring the random number skews it towards the front,
                    # but don't just take the very fittest always
                    #TODO: Examine / parameterize this
                    index_to_clone = int((get_rng().random() ** 2) * num_to_consider)
                    logger.info(f"Cloning organism at index {index_to_clone} of {num_to_consider}")

                    organism_id_to_breed = sorted_orgs_and_scores[index_to_clone][0]
                    logger.info(f"Cloning organism {organism_id_to_breed}")

                new_organism = self.clone(organism_id_to_breed, session)
                logger.info(f"Bred organism {new_organism.id} from {organism_id_to_breed}")
                self.population.add(new_organism, session)

    def clone(self, organism_id: uuid.UUID, session: Session, mutate=True) -> Organism:
        original_genotype = session.scalar(select(Gene)
                             .join_from(Organism, Gene)
                             .where(Organism.id == organism_id))
        clone_genotype = original_genotype
        if mutate:
            for mutagen in self.get_mutagens(session):
                clone_genotype = mutagen.mutate(clone_genotype)
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