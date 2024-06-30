import unittest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from roxene import Organism, EntityBase
from roxene.tic_tac_toe import Trial, Player, Environment


class Environment_test(unittest.TestCase):

    def test_get_relevant_moves(self):
        engine: Engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)

        sut: Environment = Environment(123, engine)

        o1, o2, o3 = Organism(), Organism(), Organism()
        o1_id, o2_id, o3_id = o1.id, o2.id, o3.id

        trial_1 = Trial(Player(o1, queued_input=[(0, 0), (0, 1), (0, 2)]), Player(o2, queued_input=[(1, 0), (1, 1)]))
        trial_2 = Trial(Player(o1, queued_input=[(0, 0), (0, 1)]), Player(o3, queued_input=[(1, 0), (0, 0)]))

        with Session(engine) as session:
            session.add(trial_1)
            trial_1.run()
            session.commit()

        with Session(engine) as session:
            session.add(trial_2)
            trial_2.run()
            session.commit()

        with Session(engine) as session:
            self.assertEqual(5, len(list(sut.get_relevant_moves([o1_id], session))))
            self.assertEqual(4, len(list(sut.get_relevant_moves([o2_id, o3_id], session))))


if __name__ == '__main__':
    unittest.main()
