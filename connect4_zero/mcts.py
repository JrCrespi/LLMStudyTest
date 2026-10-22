"""PUCT Monte Carlo Tree Search, batched across many independent trees.

Running many games at once and sending all their leaves to the network in a
single forward pass is what makes self-play affordable on a CPU: the Python
tree walk stays cheap and the network sees batches instead of single boards.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .game import WIDTH, State, cells_to_columns, winning_cells

EvalFn = Callable[[list[State]], tuple[np.ndarray, np.ndarray]]


@dataclass
class MCTSConfig:
    simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 1.0
    dirichlet_eps: float = 0.25
    fpu_reduction: float = 0.25  # first-play urgency: unvisited child Q = parent Q - fpu
    root_fpu_reduction_noise: float = 0.0  # at a noised self-play root, let noise drive exploration
    tactical_prior: bool = True  # force immediate wins / forced blocks into the prior


class Node:
    """Search node. Edge statistics are plain Python lists: with only 7 actions,
    list arithmetic is ~10x faster than NumPy calls on tiny arrays."""

    __slots__ = ("state", "prior", "N", "W", "children", "expanded", "value_sum", "visits")

    def __init__(self, state: State):
        self.state = state
        self.prior = [0.0] * WIDTH
        self.N = [0] * WIDTH  # edge visit counts
        self.W = [0.0] * WIDTH  # edge value sums (this node's player's view)
        self.children: list[Node | None] = [None] * WIDTH
        self.expanded = False
        self.value_sum = 0.0  # sum of backed-up values from this node's player's view
        self.visits = 0  # = 1 (own evaluation) + sum(N) once expanded

    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def _select_action(node: Node, c_puct: float, fpu: float) -> int:
    c = c_puct * math.sqrt(node.visits)
    q_unvisited = node.q() - fpu
    best_a, best = -1, -math.inf
    N, W, P = node.N, node.W, node.prior
    for a in range(WIDTH):
        p = P[a]
        if p <= 0.0:  # illegal or pruned
            continue
        n = N[a]
        score = (W[a] / n if n else q_unvisited) + c * p / (1 + n)
        if score > best:
            best_a, best = a, score
    return best_a


def tactical_moves(state: State) -> list[int] | None:
    """Moves forced by one-ply tactics, or None if nothing is forced.

    * If the side to move can win immediately, only the winning moves matter.
    * Otherwise, if the opponent threatens exactly one immediate win, the
      blocking move is the only move that does not lose on the spot.
    """
    playable = state.playable_cells()
    wins = winning_cells(state.current, state.mask) & playable
    if wins:
        return cells_to_columns(wins)
    threats = winning_cells(state.current ^ state.mask, state.mask) & playable
    if threats and threats & (threats - 1) == 0:  # exactly one threat
        return cells_to_columns(threats)
    return None


def _expand(node: Node, priors: np.ndarray, cfg: MCTSConfig) -> None:
    allowed = node.state.legal_moves()
    if cfg.tactical_prior:
        forced = tactical_moves(node.state)
        if forced is not None:
            allowed = forced
    p = [0.0] * WIDTH
    total = 0.0
    for a in allowed:
        p[a] = float(priors[a])
        total += p[a]
    if total > 0:
        node.prior = [x / total for x in p]
    else:
        node.prior = [1.0 / len(allowed) if a in allowed else 0.0 for a in range(WIDTH)]
    node.expanded = True


def _backup(path: list[tuple[Node, int]], leaf: Node, value: float) -> None:
    """``value`` is from the point of view of the player to move at ``leaf``."""
    leaf.visits += 1
    leaf.value_sum += value
    v = -value
    for node, a in reversed(path):
        node.N[a] += 1
        node.W[a] += v
        node.visits += 1
        node.value_sum += v
        v = -v


def add_dirichlet_noise(node: Node, cfg: MCTSConfig, rng: np.random.Generator) -> None:
    legal = [a for a in range(WIDTH) if node.prior[a] > 0]
    if len(legal) < 2:
        return
    noise = rng.dirichlet([cfg.dirichlet_alpha] * len(legal))
    for a, n in zip(legal, noise):
        node.prior[a] = (1 - cfg.dirichlet_eps) * node.prior[a] + cfg.dirichlet_eps * float(n)


def run_mcts(
    roots: list[Node],
    evaluate: EvalFn,
    cfg: MCTSConfig,
    add_noise: bool | list[bool] = False,
    rng: np.random.Generator | None = None,
    simulations: int | list[int] | None = None,
) -> None:
    """Search every root, batching leaf evaluations across roots.

    ``simulations`` and ``add_noise`` may be given per root, so roots with
    different budgets (e.g. full and fast searches in self-play) still share
    network calls. Every non-terminal root receives exactly its budget: a
    simulation that ends in a terminal node is backed up immediately.
    """
    rng = rng or np.random.default_rng()
    n = len(roots)
    budgets = [cfg.simulations if simulations is None else simulations] * n if not isinstance(simulations, list) else simulations
    noise = add_noise if isinstance(add_noise, list) else [add_noise] * n
    live = [k for k in range(n) if not roots[k].state.is_terminal()]

    fresh = [roots[k] for k in live if not roots[k].expanded]
    if fresh:
        priors, values = evaluate([r.state for r in fresh])
        for r, p, v in zip(fresh, priors, values):
            _expand(r, p, cfg)
            r.visits += 1
            r.value_sum += float(v)
    root_fpu = [cfg.root_fpu_reduction_noise if noise[k] else cfg.fpu_reduction for k in range(n)]
    for k in live:
        if noise[k]:
            add_dirichlet_noise(roots[k], cfg, rng)

    for sim in range(max((budgets[k] for k in live), default=0)):
        pending: list[tuple[list[tuple[Node, int]], Node]] = []
        for k in live:
            if sim >= budgets[k]:
                continue
            node, path, fpu = roots[k], [], root_fpu[k]
            while node.expanded:
                a = _select_action(node, cfg.c_puct, fpu)
                fpu = cfg.fpu_reduction
                path.append((node, a))
                child = node.children[a]
                if child is None:
                    child = Node(node.state.play(a))
                    node.children[a] = child
                node = child
            if node.state.is_terminal():
                _backup(path, node, node.state.terminal_value())
            else:
                pending.append((path, node))
        if pending:
            priors, values = evaluate([leaf.state for _, leaf in pending])
            for (path, leaf), p, v in zip(pending, priors, values):
                _expand(leaf, p, cfg)
                _backup(path, leaf, float(v))


def visit_policy(root: Node, temperature: float) -> np.ndarray:
    n = np.array(root.N, dtype=np.float64)
    if n.sum() == 0:  # e.g. zero simulations: fall back to the prior
        n = np.array(root.prior, dtype=np.float64)
    if temperature <= 1e-6:
        best = np.flatnonzero(n == n.max())
        pi = np.zeros(WIDTH)
        pi[best] = 1.0 / len(best)
        return pi
    n = n ** (1.0 / temperature)
    return n / n.sum()


def best_move(root: Node, rng: np.random.Generator) -> int:
    """Most visited move, ties broken at random."""
    m = max(root.N)
    return int(rng.choice([a for a in range(WIDTH) if root.N[a] == m]))


class MCTSPlayer:
    """Plays a single game position with MCTS (used for evaluation and play)."""

    def __init__(self, evaluate: EvalFn, cfg: MCTSConfig, name: str = "alphazero"):
        self.evaluate = evaluate
        self.cfg = cfg
        self.name = name

    def policy(self, state: State, rng: np.random.Generator | None = None) -> np.ndarray:
        if self.cfg.simulations == 0:
            priors, _ = self.evaluate([state])
            return priors[0]
        root = Node(state)
        run_mcts([root], self.evaluate, self.cfg, rng=rng)
        return visit_policy(root, temperature=0.0)

    def __call__(self, state: State, rng: np.random.Generator) -> int:
        pi = self.policy(state, rng)
        best = np.flatnonzero(pi == pi.max())
        return int(rng.choice(best))
