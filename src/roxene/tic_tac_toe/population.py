from collections import defaultdict

import copy
import logging
from functools import reduce
from numpy.random import Generator
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import func
from typing import Set, List

from ..mutagens import Mutagen
from ..organism import Organism
from .players import REQUIRED_INPUTS, REQUIRED_OUTPUTS
from .trial import Trial, Outcome
from .trial import Participant


class Population:
    busy_organisms: Set[Organism]
    completed_trials: List[Trial]
    rng: Generator

    engine: Engine
    mutagens: List[Mutagen]

    def __init__(self, storage_engine: Engine, mutagens: List[Mutagen]):
        self.engine = storage_engine
        self.mutagens = mutagens
        self.logger = logging.getLogger(__name__)

    def add(self, organism: Organism):
        with Session(self.engine) as session:
            session.add(organism)
            session.commit()

    def sample(self, num_organisms: int = 1, idle_only=False, active_session=None):
        # TODO: Make the randomness depend on an RNG
        stmt = select(Organism)
        if idle_only:
            busy_organisms_query = (select(Organism.id)
                                    .join(Participant)
                                    .join(Trial)
                                    .where(Trial.end_date is None))
            stmt = stmt.where(~Organism.id.in_(busy_organisms_query.subquery()))
        stmt = stmt.order_by(func.random()).limit(num_organisms)
        if active_session:
            return active_session.scalars(stmt).all()
        else:
            with Session(self.engine) as session:
                return session.scalars(stmt).unique().all()

    def breed(self, num_to_breed: int, num_to_consider: int = 10):
        selectee_scores: dict[Organism, int] = defaultdict(int)
        for n in range(num_to_breed):
            selectees = self.sample(num_to_consider)
            selectees_to_score = filter(lambda org: org not in selectee_scores, selectees)
            self.logger.info(f"Scoring {len(selectees_to_score)} selectees of {len(selectees)}")
            moves_to_score = self.get_relevant_moves(selectees_to_score)
            self.logger.info(f"Found {len(moves_to_score)} moves")
            for move in moves_to_score:
                selectee_scores[move.organism] += self.score_move(move)
            organism_to_breed: Organism = \
                reduce(lambda a,b: a if selectee_scores[a] >= selectee_scores[b] else b, selectees)
            new_organism = self.clone(organism_to_breed)
            self.logger.info(f"Bred organism {new_organism.id} from {organism_to_breed.id}")
            self.add(new_organism)

    def start_trial(self) -> Trial:
        orgs = self.sample(2, idle_only=True)
        with Session(self.engine) as session:
            for org in orgs:
                busy_org = BusyOrganism
                session.add_all()
        self.busy_organisms.update(orgs)

    def cull(self, num_to_cull: int, num_to_compare: int = 10):
        for n in range(num_to_cull):
            selectees = self.rng.choice(list(self.organisms), size=num_to_compare, replace=False)
            selectee_scores: dict[Organism, int] = dict([(o, 0) for o in selectees])
            for move, organism in self.get_relevant_moves(selectee_scores.keys()):
                selectee_scores[organism] += self.score_move(move)
            # Just kill the lamest of the bunch for now
            # TODO: Make stochastic
            organism_to_kill: Organism = sorted(selectee_scores.items(), key=lambda item: item[1], reverse=True)[0][0]
            self.logger.info(f"Removing organism {organism_to_kill}")
            self.organisms.discard(organism_to_kill)

    def clone(self, organism_to_breed: Organism) -> Organism:
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
        with Session(self.engine) as session:
            stmt = select(Move)

        for trial in self.completed_trials:
            for move in trial.moves:
                if trial.players[move.letter].organism.id in selectee_ids:
                    yield move, trial.players[move.letter].organism.id
