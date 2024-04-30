import logging
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import func
from typing import List

from .players import Player
from .trial import Trial
from ..organism import Organism


class Population:

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def add(self, organism: Organism, session: Session):
        session.add(organism)
        session.commit()

    def remove(self, organism_to_kill: Organism, session: Session):
        session.delete(organism_to_kill)
        session.commit()

    def sample(self, num_organisms: int, idle_only: bool, session: Session):
        # TODO: Make the randomness depend on an RNG
        stmt = select(Organism)
        if idle_only:
            busy_organisms_query = (select(Organism.id)
                                    .join(Player)
                                    .join(Trial)
                                    .where(Trial.end_date is None))
            stmt = stmt.where(~Organism.id.in_(busy_organisms_query.subquery()))
        stmt = stmt.order_by(func.random()).limit(num_organisms)  # TODO: Make this depend on the RNG
        return session.scalars(stmt).unique().all()

    def start_trial(self, session: Session) -> Trial:
        try:
            orgs: List[Organism] = self.sample(2, True, session)
        except:
            raise Exception("Not enough idle Organisms")
        p1 = Player(orgs[0])
        p2 = Player(orgs[1])
        print(f'Starting a trial with {p1} and {p2}')
        trial = Trial(p1, p2)
        session.add(trial)
        session.commit()
        return trial

    def complete_trial(self, trial: Trial, session: Session):
        session.merge(trial)
        session.commit()
