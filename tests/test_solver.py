import shutil
from functools import lru_cache

import numpy as np
import pytest

from connect4_zero.game import State

pytestmark = pytest.mark.skipif(
    not (shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")), reason="needs a C compiler"
)


@lru_cache(None)
def brute(cur: int, mask: int, moves: int) -> int:
    s = State(cur, mask, moves)
    best = -99
    for c in s.legal_moves():
        if s.is_winning_move(c):
            v = (43 - s.moves) // 2
        else:
            t = s.play(c)
            v = 0 if t.is_terminal() else -brute(t.current, t.mask, t.moves)
        best = max(best, v)
    return best


def test_solver_matches_brute_force_on_late_positions():
    from connect4_zero import solver

    rng = np.random.default_rng(0)
    checked = 0
    while checked < 40:
        s = State()
        target = int(rng.integers(28, 34))
        while s.moves < target and not s.is_terminal():
            s = s.play(int(rng.choice(s.legal_moves())))
        if s.is_terminal():
            continue
        assert solver.solve(s) == brute(s.current, s.mask, s.moves)
        checked += 1


def test_solver_tactics():
    from connect4_zero import solver

    scores = solver.move_scores(State.from_moves("010203"))
    assert max(scores, key=scores.get) == 0 and scores[0] > 0
    scores = solver.move_scores(State.from_moves("01020"))
    assert max(scores, key=scores.get) == 0
