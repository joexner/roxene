from roxene import Organism
from enum import Enum, auto

from roxene.tic_tac_toe.players import ManualPlayer, OrganismPlayer

WIN_SETS = [
    {(0, 0), (0, 1), (0, 2)},
    {(1, 0), (1, 1), (1, 2)},
    {(2, 0), (2, 1), (2, 2)},
    {(0, 0), (1, 0), (2, 0)},
    {(0, 1), (1, 1), (2, 1)},
    {(0, 2), (1, 2), (2, 2)},
    {(0, 2), (1, 1), (2, 0)},
    {(0, 0), (1, 1), (2, 2)},
]

class Move:
    def __init__(self, letter: str, initial_board_state: list[list[str]]):
        self.letter = letter
        self.initial_board_state = [row.copy() for row in initial_board_state]
        self.position: tuple(int, int) = None
        self.outcomes: set[Outcome] = set()
        self.resultant_board_state: list[list[str]] = None

class Trial:
    def __init__(self, player_1, player_2):
        self.board = [[None, None, None], [None, None, None], [None, None, None]]
        self.next_player_letter = 'X'
        self.moves = []
        self.players = {
            'X': player_1,
            'O': player_2
        }

    def is_finished(self) -> bool:
        return len(self.moves) == 9 or \
               any(filter(lambda move: Outcome.WIN in move.outcomes or Outcome.LOSE in move.outcomes, self.moves))

    def run(self):
        while not (self.is_finished()):
            current_player_letter = self.next_player_letter
            self.next_player_letter = 'X' if self.next_player_letter == 'O' else 'O'
            current_player = self.players[current_player_letter]
            this_move = Move(letter=current_player_letter, initial_board_state=self.board)
            try:
                move_coords = current_player.get_move_coords(self.board)
                this_move.position = move_coords
                existing_square_value = self.board[move_coords[0]][move_coords[1]]
                if existing_square_value:
                    this_move.outcomes |= {Outcome.INVALID_MOVE, Outcome.LOSE}
                else:
                    this_move.outcomes |= {Outcome.VALID_MOVE}
                    self.board[move_coords[0]][move_coords[1]] = current_player_letter
                    this_move.resultant_board_state = [row.copy() for row in self.board]
                    win_sets_with_this_square = filter(lambda win_set: move_coords in win_set, WIN_SETS)
                    for win_set in win_sets_with_this_square:
                        square_values = [self.board[x][y] for x, y in win_set]
                        is_winning_move = all(square == current_player_letter for square in square_values)
                        if is_winning_move:
                            this_move.outcomes |= {Outcome.WIN}
            except TimeoutError:
                this_move.outcomes |= {Outcome.TIMEOUT, Outcome.LOSE}
            self.moves.append(this_move)

class Outcome(Enum):
    WIN = auto()
    LOSE = auto()
    TIE = auto()
    TIMEOUT = auto()
    VALID_MOVE = auto()
    INVALID_MOVE = auto()

def run_trial(self, *orgs: Organism) -> Trial:
    assert len(orgs) == 2, "Tic tac toe requires exactly 2 players"
    trial = Trial(OrganismPlayer(orgs[0], 'X'), OrganismPlayer(orgs[1], 'O'))
    trial.run()
    return trial