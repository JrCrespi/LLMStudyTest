"""Reference opponents used to measure playing strength.

* RandomPlayer       - uniform over legal moves.
* GreedyPlayer       - wins if it can, blocks an immediate loss, else random.
* MinimaxPlayer      - depth-limited negamax with alpha-beta pruning and a
                       classic window-counting heuristic. Depth 1..7 gives a
                       ladder of increasingly strong opponents.

All players break ties uniformly at random so evaluation games are diverse.
"""
from __future__ import annotations

import numpy as np

from .game import HEIGHT, N_CELLS, WIDTH, State

_CENTER_ORDER = [3, 2, 4, 1, 5, 0, 6]
_WIN_SCORE = 1_000_000


def _windows() -> list[list[tuple[int, int]]]:
    out = []
    for r in range(HEIGHT):
        for c in range(WIDTH):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(r + i * dr, c + i * dc) for i in range(4)]
                if all(0 <= rr < HEIGHT and 0 <= cc < WIDTH for rr, cc in cells):
                    out.append(cells)
    return out


_WINDOWS = _windows()
# Each window as a bitmask over the bitboard layout (bit = col * 7 + row-from-bottom).
_WINDOW_MASKS = [sum(1 << (c * (HEIGHT + 1) + r) for r, c in w) for w in _WINDOWS]
_CENTER_MASK = sum(1 << (3 * (HEIGHT + 1) + r) for r in range(HEIGHT))


def heuristic(state: State) -> float:
    """Static evaluation from the point of view of the player to move."""
    own, opp = state.current, state.current ^ state.mask
    score = 3 * (bin(own & _CENTER_MASK).count("1") - bin(opp & _CENTER_MASK).count("1"))
    for m in _WINDOW_MASKS:
        o, p = own & m, opp & m
        if o and p:
            continue
        if o:
            k = bin(o).count("1")
            score += 2 if k == 2 else 5 if k == 3 else 0
        elif p:
            k = bin(p).count("1")
            score -= 2 if k == 2 else 5 if k == 3 else 0
    return float(score)


class RandomPlayer:
    name = "random"

    def __call__(self, state: State, rng: np.random.Generator) -> int:
        return int(rng.choice(state.legal_moves()))


class GreedyPlayer:
    name = "greedy"

    def __call__(self, state: State, rng: np.random.Generator) -> int:
        legal = state.legal_moves()
        for c in legal:
            if state.is_winning_move(c):
                return c
        opp_view = State(current=state.current ^ state.mask, mask=state.mask, moves=state.moves)
        threats = [c for c in legal if opp_view.is_winning_move(c)]
        if threats:
            return threats[0]
        return int(rng.choice(legal))


class MinimaxPlayer:
    def __init__(self, depth: int):
        self.depth = depth
        self.name = f"minimax-d{depth}"

    def _negamax(self, s: State, depth: int, alpha: float, beta: float) -> float:
        if s.winner_last:
            return -(_WIN_SCORE + (N_CELLS - s.moves))  # prefer faster wins / slower losses
        if s.moves == N_CELLS:
            return 0.0
        if depth == 0:
            return heuristic(s)
        for c in _CENTER_ORDER:
            if s.can_play(c) and s.is_winning_move(c):
                return _WIN_SCORE + (N_CELLS - s.moves - 1)
        best = -float("inf")
        for c in _CENTER_ORDER:
            if not s.can_play(c):
                continue
            val = -self._negamax(s.play(c), depth - 1, -beta, -alpha)
            if val > best:
                best = val
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def scores(self, state: State) -> dict[int, float]:
        return {
            c: -self._negamax(state.play(c), self.depth - 1, -float("inf"), float("inf"))
            for c in state.legal_moves()
        }

    def __call__(self, state: State, rng: np.random.Generator) -> int:
        scores = self.scores(state)
        best = max(scores.values())
        return int(rng.choice([c for c, v in scores.items() if v == best]))
