"""Head-to-head evaluation with balanced colours and diverse openings."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .game import WIDTH, State

Player = Callable[[State, np.random.Generator], int]


@dataclass
class MatchResult:
    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """Points per game: win = 1, draw = 0.5, loss = 0."""
        return (self.wins + 0.5 * self.draws) / max(self.games, 1)

    def score_ci(self, z: float = 1.96) -> tuple[float, float]:
        """Wilson interval on the score (draws counted as half a win)."""
        n = self.games
        if n == 0:
            return (0.0, 1.0)
        p = self.score
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (max(0.0, centre - half), min(1.0, centre + half))

    def elo_diff(self) -> float:
        """Elo difference implied by the score (clipped to avoid infinities)."""
        p = min(max(self.score, 1e-3), 1 - 1e-3)
        return -400.0 * math.log10(1.0 / p - 1.0)

    def as_dict(self) -> dict:
        lo, hi = self.score_ci()
        return {
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "games": self.games,
            "score": round(self.score, 4),
            "score_ci95": [round(lo, 4), round(hi, 4)],
            "elo_diff": round(self.elo_diff(), 1),
        }


def play_game(first: Player, second: Player, rng: np.random.Generator, opening: tuple[int, ...] = ()) -> int:
    """Return +1 if ``first`` wins, -1 if ``second`` wins, 0 for a draw."""
    s = State()
    for m in opening:
        s = s.play(m)
    players = (first, second)
    while not s.is_terminal():
        mover = players[s.moves % 2]
        s = s.play(mover(s, rng))
    if not s.winner_last:
        return 0
    # The player who made the last move won; plies are counted from the empty board.
    return 1 if (s.moves - 1) % 2 == 0 else -1


def openings(plies: int) -> list[tuple[int, ...]]:
    """All openings of ``plies`` moves (7**plies of them for plies <= 2)."""
    return list(itertools.product(range(WIDTH), repeat=plies))


def match(
    agent: Player,
    opponent: Player,
    games: int,
    seed: int = 0,
    opening_plies: int = 2,
) -> MatchResult:
    """Play ``games`` games, alternating colours, each from a forced random opening.

    The same opening is played twice, once with each colour, so neither side
    benefits from a lucky opening. Openings are forced for both players and
    are not part of the agent's decision making.
    """
    rng = np.random.default_rng(seed)
    book = openings(opening_plies)
    order = rng.permutation(len(book))
    w = d = l = 0
    for g in range(games):
        opening = book[order[(g // 2) % len(book)]]
        if g % 2 == 0:
            r = play_game(agent, opponent, rng, opening)
        else:
            r = -play_game(opponent, agent, rng, opening)
        w += r > 0
        d += r == 0
        l += r < 0
    return MatchResult(w, d, l)
