from . import ManualPlayer, Trial

trial = Trial(ManualPlayer('X'), ManualPlayer('O'))
trial.run()
print(f"Game finished with moves {[(move.letter, move.outcomes) for move in trial.moves]}")
exit(0)
