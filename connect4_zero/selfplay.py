"""Batched self-play: many games advance in lock-step, sharing network calls."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from .game import WIDTH, State
from .mcts import EvalFn, MCTSConfig, Node, best_move, run_mcts, visit_policy


@dataclass
class SelfPlayConfig:
    games: int = 64
    temperature_moves: int = 12  # sample moves proportionally to visits for the first N plies
    # Playout-cap randomisation (KataGo): only a fraction of moves get a full,
    # noised search and become training samples; the rest use a cheap search
    # just to advance the game. More games per CPU-second -> better value targets.
    full_search_prob: float = 0.25
    fast_simulations: int = 40
    mcts: MCTSConfig = field(default_factory=MCTSConfig)


@dataclass
class Sample:
    planes: np.ndarray  # (2, 6, 7)
    policy: np.ndarray  # (7,) MCTS visit distribution
    value: float  # final outcome from the side to move: +1 / 0 / -1
    root_q: float  # MCTS value estimate at this position (side to move)


def self_play(evaluate: EvalFn, cfg: SelfPlayConfig, rng: np.random.Generator) -> tuple[list[Sample], dict]:
    roots = [Node(State()) for _ in range(cfg.games)]
    histories: list[list[tuple[np.ndarray, np.ndarray, float, int]]] = [[] for _ in range(cfg.games)]
    outcomes: list[int | None] = [None] * cfg.games  # +1 first player won, -1 second, 0 draw
    lengths = [0] * cfg.games

    while any(o is None for o in outcomes):
        active = [i for i, o in enumerate(outcomes) if o is None]
        full = {i for i in active if rng.random() < cfg.full_search_prob}
        # one lock-step search: full and fast roots share every network batch
        run_mcts(
            [roots[i] for i in active],
            evaluate,
            cfg.mcts,
            add_noise=[i in full for i in active],
            rng=rng,
            simulations=[cfg.mcts.simulations if i in full else cfg.fast_simulations for i in active],
        )
        for i in active:
            root = roots[i]
            s = root.state
            if i in full:
                histories[i].append((s.planes(), visit_policy(root, temperature=1.0), root.q(), s.moves))
            if s.moves < cfg.temperature_moves:
                a = int(rng.choice(WIDTH, p=visit_policy(root, temperature=1.0)))
            else:
                a = best_move(root, rng)
            child = root.children[a] or Node(s.play(a))
            # Reuse the subtree but drop the parent so memory can be freed.
            roots[i] = child
            ns = child.state
            if ns.is_terminal():
                lengths[i] = ns.moves
                if ns.winner_last:
                    outcomes[i] = 1 if (ns.moves - 1) % 2 == 0 else -1
                else:
                    outcomes[i] = 0

    samples: list[Sample] = []
    for hist, outcome in zip(histories, outcomes):
        for planes, pi, q, moves in hist:
            side = 1 if moves % 2 == 0 else -1  # +1 if first player is to move
            samples.append(Sample(planes, pi.astype(np.float32), float(outcome * side), float(q)))

    stats = {
        "games": cfg.games,
        "positions": len(samples),
        "first_player_wins": sum(o == 1 for o in outcomes),
        "second_player_wins": sum(o == -1 for o in outcomes),
        "draws": sum(o == 0 for o in outcomes),
        "mean_length": float(np.mean(lengths)),
    }
    return samples, stats


# --------------------------------------------------------------------------
# Multi-process self-play: one single-threaded PyTorch worker per CPU core.
# On small CPUs this is ~2x faster than one process using intra-op threads,
# and it also parallelises the Python tree search.
# --------------------------------------------------------------------------
_WORKER_NET = None


def _worker_init(net_cfg_dict: dict) -> None:
    global _WORKER_NET
    import torch

    from .model import NetConfig, PolicyValueNet

    torch.set_num_threads(1)
    _WORKER_NET = PolicyValueNet(NetConfig(**net_cfg_dict))


def _worker_run(args):
    state_dict, cfg, seed = args
    from .model import Evaluator

    _WORKER_NET.load_state_dict(state_dict)
    return self_play(Evaluator(_WORKER_NET), cfg, np.random.default_rng(seed))


class SelfPlayPool:
    def __init__(self, net_cfg_dict: dict, workers: int):
        import multiprocessing as mp

        self.workers = workers
        self.pool = mp.get_context("spawn").Pool(workers, initializer=_worker_init, initargs=(net_cfg_dict,))

    def run(self, net, cfg: SelfPlayConfig, rng: np.random.Generator) -> tuple[list[Sample], dict]:
        state_dict = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        per = [cfg.games // self.workers + (i < cfg.games % self.workers) for i in range(self.workers)]
        jobs = [
            (state_dict, dataclasses.replace(cfg, games=n), int(rng.integers(2**31)))
            for n in per
            if n > 0
        ]
        samples: list[Sample] = []
        stats: dict = {}
        lengths = []
        for s, st in self.pool.map(_worker_run, jobs):
            samples.extend(s)
            lengths.append((st["mean_length"], st["games"]))
            for k, v in st.items():
                if k != "mean_length":
                    stats[k] = stats.get(k, 0) + v
        stats["mean_length"] = sum(m * g for m, g in lengths) / max(stats.get("games", 1), 1)
        return samples, stats

    def close(self) -> None:
        self.pool.close()
        self.pool.join()
