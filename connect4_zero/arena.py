"""Head-to-head evaluation with balanced colours, diverse openings and honest error bars.

Protocol
--------
* Games are played in pairs: the same forced opening twice, once with each
  colour, so neither side profits from a lucky opening.
* Openings are drawn *without replacement* from all 2-ply (49) or 3-ply (343)
  openings, so no pair is a duplicate of another.
* Confidence intervals treat the colour-swapped pair as the unit of
  independence (see ``MatchResult.score_ci``). Elo is reported at the
  interval bounds, never as a clipped point estimate for a 100% score.
"""
from __future__ import annotations

import itertools
import math
from statistics import NormalDist
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .game import WIDTH, State

Player = Callable[[State, np.random.Generator], int]


def elo_from_score(p: float) -> float | None:
    """Elo difference implied by an expected score; None means unbounded (score 0 or 1)."""
    if p <= 0.0 or p >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / p - 1.0)


@dataclass
class MatchResult:
    # per-game points for the agent (1 / 0.5 / 0), grouped in colour-swapped pairs
    pairs: list[tuple[float, float]] = field(default_factory=list)
    as_first: list[float] = field(default_factory=list)
    as_second: list[float] = field(default_factory=list)

    @property
    def points(self) -> list[float]:
        return [p for pair in self.pairs for p in pair]

    @property
    def games(self) -> int:
        return len(self.points)

    @property
    def wins(self) -> int:
        return sum(p == 1.0 for p in self.points)

    @property
    def draws(self) -> int:
        return sum(p == 0.5 for p in self.points)

    @property
    def losses(self) -> int:
        return sum(p == 0.0 for p in self.points)

    @property
    def score(self) -> float:
        return float(np.mean(self.points)) if self.points else 0.0

    def score_ci(self, level: float = 0.95, n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
        """Confidence interval on the score, with colour-swapped pairs as the unit.

        Union of two intervals on the mean pair score:
        * a percentile bootstrap over pairs (tight in the middle, accounts for
          draws and within-pair correlation);
        * a Wilson interval with n = number of pairs, which is conservative
          for any [0, 1]-bounded outcome and stays well-behaved at 0% / 100%,
          where the bootstrap collapses.
        """
        if not self.pairs:
            return (0.0, 1.0)
        pair_scores = np.array([np.mean(p) for p in self.pairs])
        n = len(pair_scores)
        alpha = 1 - level
        rng = np.random.default_rng(seed)
        boots = pair_scores[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
        b_lo, b_hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
        z = float(NormalDist().inv_cdf(1 - alpha / 2))
        p = float(pair_scores.mean())
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (float(min(b_lo, centre - half)), float(max(b_hi, centre + half)))

    def as_dict(self) -> dict:
        lo, hi = self.score_ci()

        def wdl(xs):
            return [sum(x == 1.0 for x in xs), sum(x == 0.5 for x in xs), sum(x == 0.0 for x in xs)]

        return {
            "games": self.games,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "wdl_as_first": wdl(self.as_first),
            "wdl_as_second": wdl(self.as_second),
            "score": round(self.score, 4),
            "score_ci95": [round(lo, 4), round(hi, 4)],
            # Elo difference at the CI bounds; None = unbounded (e.g. a perfect score)
            "elo_ci95": [None if e is None else round(e) for e in (elo_from_score(lo), elo_from_score(hi))],
            "pair_points": [list(p) for p in self.pairs],  # raw data, so any statistic can be recomputed
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
    """All openings of ``plies`` moves (none of them can be terminal for plies < 7)."""
    return list(itertools.product(range(WIDTH), repeat=plies))


def match(
    agent: Player,
    opponent: Player,
    pairs: int,
    seed: int = 0,
    opening_plies: int = 2,
) -> MatchResult:
    """Play ``pairs`` colour-swapped pairs of games from distinct random openings."""
    rng = np.random.default_rng(seed)
    book = openings(opening_plies)
    if pairs > len(book):
        raise ValueError(f"only {len(book)} distinct {opening_plies}-ply openings; use more plies")
    chosen = rng.choice(len(book), size=pairs, replace=False)
    res = MatchResult()
    for k in chosen:
        opening = book[k]
        a = (play_game(agent, opponent, rng, opening) + 1) / 2
        b = (1 - play_game(opponent, agent, rng, opening)) / 2
        res.pairs.append((a, b))
        res.as_first.append(a)
        res.as_second.append(b)
    return res
