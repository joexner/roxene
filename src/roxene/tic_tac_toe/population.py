import logging
import time
import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import func

from .players import Player
from .trial import Trial
from ..organism import Organism
from ..util import get_rng

logger = logging.getLogger(__name__)

class Population:

    def add(self, organism: Organism, session: Session):
        session.add(organism)

    def remove(self, organism_id_to_kill: uuid.UUID, session: Session):
        organism = session.get(Organism, organism_id_to_kill)
        if organism is None:
            raise ValueError(f"No organism with ID {organism_id_to_kill} found to delete.")
        if organism.deleted_date is not None:
            raise ValueError(f"Organism {organism_id_to_kill} is already removed.")
        organism.deleted_date = datetime.now()

    def count(self, session: Session) -> int:
        """Count the total number of living organisms in the population."""
        return session.scalar(select(func.count(Organism.id)).where(Organism.deleted_date.is_(None)))

    def sample(self, num_to_select: int, idle_only: bool, session: Session):

        # Build the base stmt once
        candidate_select_stmt = select(Organism.id).where(Organism.deleted_date.is_(None)).order_by(Organism.id)
        if idle_only:
            busy_organisms_query = (select(Organism.id)
                                    .join(Player)
                                    .join(Trial)
                                    .where(Trial.end_date.is_(None)))
            candidate_select_stmt = candidate_select_stmt.where(~Organism.id.in_(busy_organisms_query))

        if logger.isEnabledFor(logging.DEBUG):
            start = time.perf_counter()
        num_candidates = session.execute(select(func.count()).select_from(candidate_select_stmt)).scalar()
        if logger.isEnabledFor(logging.DEBUG):
            end = time.perf_counter()
            logger.debug(f"Count query took {end - start} seconds")

        if num_candidates < num_to_select:
            raise ValueError(f"Only {num_candidates} candidates available, not enough candidates to select {num_to_select} organisms. ")

        indexes = []
        results = []
        for _ in range(num_to_select):
            idx = get_rng().integers(0, num_candidates)
            while idx in indexes:
                idx = get_rng().integers(0, num_candidates)
            indexes.append(idx)

        indexes.sort()

        for idx in indexes:
            stmt_with_offset = candidate_select_stmt.offset(idx).limit(1)
            if logger.isEnabledFor(logging.DEBUG):
                start = time.perf_counter()
            result = session.scalars(stmt_with_offset).unique().all()[0]
            if logger.isEnabledFor(logging.DEBUG):
                end = time.perf_counter()
                logger.debug(f"Organism fetch took {end - start} seconds")
            results.append(result)

        return results
