import logging
import time
from numpy.random import Generator
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import func

from .players import Player
from .trial import Trial
from ..organism import Organism


class Population:

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def add(self, organism: Organism, session: Session):
        session.add(organism)

    def remove(self, organism_to_kill: Organism, session: Session):
        session.delete(organism_to_kill)

    def sample(self, num_to_select: int, idle_only: bool, rng: Generator, session: Session):

        # Build the base stmt once
        candidate_select_stmt = select(Organism.id).order_by(Organism.id)
        if idle_only:
            busy_organisms_query = (select(Organism.id)
                                    .join(Player)
                                    .join(Trial)
                                    .where(Trial.end_date.is_(None)))
            candidate_select_stmt = candidate_select_stmt.where(~Organism.id.in_(busy_organisms_query))

        if self.logger.isEnabledFor(logging.DEBUG):
            start = time.perf_counter()
        num_candidates = session.execute(select(func.count()).select_from(candidate_select_stmt)).scalar()
        if self.logger.isEnabledFor(logging.DEBUG):
            end = time.perf_counter()
            self.logger.debug(f"Count query took {end - start} seconds")

        if num_candidates < num_to_select:
            raise ValueError(f"Only {num_candidates} candidates available, not enough candidates to select {num_to_select} organisms. ")

        indexes = []
        results = []
        for _ in range(num_to_select):
            idx = rng.integers(0, num_candidates)
            while idx in indexes:
                idx = rng.integers(0, num_candidates)
            indexes.append(idx)

        indexes.sort()

        for idx in indexes:
            stmt_with_offset = candidate_select_stmt.offset(idx).limit(1)
            if self.logger.isEnabledFor(logging.DEBUG):
                start = time.perf_counter()
            result = session.scalars(stmt_with_offset).unique().all()[0]
            if self.logger.isEnabledFor(logging.DEBUG):
                end = time.perf_counter()
                self.logger.debug(f"Organism ID query took {end - start} seconds")
            results.append(result)

        return results
