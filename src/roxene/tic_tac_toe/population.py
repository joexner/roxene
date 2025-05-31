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
        pop_size = session.execute(select(func.count()).select_from(Organism)).scalar()
        indexes = []
        results = []
        for _ in range(num_to_select):
            # TODO: fix likely bug where this can be higher than the number of available organisms
            idx = rng.integers(0, pop_size)
            while idx in indexes:
                idx = rng.integers(0, pop_size)
            indexes.append(idx)

        for _ in range(num_to_select):
            stmt = select(Organism).order_by(Organism.id)
            if idle_only:
                busy_organisms_query = (select(Organism.id)
                                        .join(Player)
                                        .join(Trial)
                                        .where(Trial.end_date is None))
                stmt = stmt.where(~Organism.id.in_(busy_organisms_query))
            stmt = stmt.offset(indexes.pop())
            stmt = stmt.limit(1)
            if self.logger.isEnabledFor(logging.DEBUG):
                start = time.perf_counter()
            result = session.scalars(stmt).unique().all()[0]
            if self.logger.isEnabledFor(logging.DEBUG):
                end = time.perf_counter()
                self.logger.debug(f"Sample query took {end - start} seconds, idle_only={idle_only}")
            results.append(result)

        return results

