"""Ground-truth benchmark: how often does the model play a game-theoretically optimal move?

1. ``build``: sample positions from games between noisy minimax players and
   label every legal move with the exact solver (solver/c4solver.c). Only
   positions where the choice matters are kept (some move changes the
   theoretical outcome). The labelled set is saved so every model is scored
   on the same positions.
2. ``score``: for each model and search budget, report
   * outcome accuracy: the chosen move keeps the best achievable outcome
     (win stays win, draw stays draw), also for the reference players;
   * value head quality on a W/D/L-balanced set derived from the same
     positions (does the raw network know who is winning?).

By construction every benchmark position is a win or a draw for the side to
move: a lost position has no move that changes the outcome, so it tests nothing.

    python scripts/solver_benchmark.py build --n 600 --out results/benchmark_positions.json
    python scripts/solver_benchmark.py score --positions results/benchmark_positions.json \\
        --model models/connect4_zero.pt --simulations 0 50 200 800 --out results/solver_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from connect4_zero.baselines import MinimaxPlayer  # noqa: E402
from connect4_zero.game import State  # noqa: E402


def _sample_moves(n: int, seed: int, min_ply: int, max_ply: int) -> list[str]:
    rng = np.random.default_rng(seed)
    players = [MinimaxPlayer(2), MinimaxPlayer(3)]
    out, seen = [], set()
    while len(out) < n:
        s, moves = State(), []
        target = int(rng.integers(min_ply, max_ply + 1))
        while not s.is_terminal() and s.moves < target:
            m = int(rng.choice(s.legal_moves())) if rng.random() < 0.35 else players[s.moves % 2](s, rng)
            s = s.play(m)
            moves.append(m)
        if not s.is_terminal() and s.key() not in seen:
            seen.add(s.key())
            out.append("".join(map(str, moves)))
    return out


def _label(moves: str) -> dict | None:
    from connect4_zero import solver

    s = State.from_moves(moves)
    scores = solver.move_scores(s)
    outcomes = {c: int(np.sign(v)) for c, v in scores.items()}
    best = max(outcomes.values())
    if all(o == best for o in outcomes.values()):
        return None  # every move leads to the same outcome: nothing to test
    return {
        "moves": moves,
        "ply": s.moves,
        "value": best,  # theoretical outcome for the side to move
        "good_moves": sorted(c for c, o in outcomes.items() if o == best),
        "scores": {str(c): v for c, v in scores.items()},
    }


def build(args) -> None:
    candidates = _sample_moves(args.n * 3, args.seed, args.min_ply, args.max_ply)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        labelled = []
        for rec in pool.imap(_label, candidates, chunksize=4):
            if rec is not None:
                labelled.append(rec)
                if len(labelled) >= args.n:
                    break
        pool.terminate()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(labelled))
    vals = [r["value"] for r in labelled]
    print(f"{len(labelled)} positions, outcome for side to move: "
          f"win {vals.count(1)}, draw {vals.count(0)}, loss {vals.count(-1)}")


def _make_baseline(name: str):
    from connect4_zero.baselines import GreedyPlayer

    if name == "greedy":
        return GreedyPlayer()
    if name.startswith("minimax-d"):
        return MinimaxPlayer(int(name.removeprefix("minimax-d")))
    raise ValueError(name)


def _value_set(positions: list[dict], seed: int = 0) -> list[tuple[State, int]]:
    """Positions with exact W/D/L labels for the side to move, balanced by construction.

    Each benchmark position contributes itself (win or draw), the position after
    its best move (the opponent is then lost or drawn) and the position after a
    random worse move (the opponent is then winning or drawn). Labels come from
    the stored exact move scores; no new solving is needed.
    """
    rng = np.random.default_rng(seed)
    out = []
    for p in positions:
        s = State.from_moves(p["moves"])
        out.append((s, p["value"]))
        scores = {int(c): v for c, v in p["scores"].items()}
        good = max(scores, key=scores.get)
        bad = [c for c, v in scores.items() if np.sign(v) < p["value"]]
        for c in [good] + ([int(rng.choice(bad))] if bad else []):
            child = s.play(c)
            if not child.is_terminal():
                out.append((child, -int(np.sign(scores[c]))))
    return out


def _score_job(job):
    import torch

    from connect4_zero.mcts import MCTSConfig, MCTSPlayer
    from connect4_zero.model import Evaluator, load_checkpoint

    torch.set_num_threads(1)
    model, sims, positions = job
    rng = np.random.default_rng(0)
    if model == "random":
        # expected score of a uniformly random legal move
        rate = float(np.mean([len(p["good_moves"]) / len(p["scores"]) for p in positions]))
        return {"model": "random (expected)", "simulations": None, "positions": len(positions),
                "optimal_outcome_move_rate": round(rate, 4)}
    if model in ("greedy",) or model.startswith("minimax-d"):
        player, ev = _make_baseline(model), None
    else:
        net, _ = load_checkpoint(model)
        ev = Evaluator(net)
        player = MCTSPlayer(ev, MCTSConfig(simulations=sims))
    correct = 0
    by_ply: dict[str, list[int]] = {}
    for p in positions:
        s = State.from_moves(p["moves"])
        ok = player(s, rng) in p["good_moves"]
        correct += ok
        bucket = "10-17" if p["ply"] < 18 else "18-25" if p["ply"] < 26 else "26+"
        by_ply.setdefault(bucket, []).append(ok)
    rec = {
        "model": model,
        "simulations": sims if ev is not None else None,
        "positions": len(positions),
        "optimal_outcome_move_rate": round(correct / len(positions), 4),
        "by_ply": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_ply.items())},
    }
    if ev is not None and sims == 0:
        vs = _value_set(positions)
        _, values = ev([st for st, _ in vs])
        labels = np.array([lab for _, lab in vs])
        win_recall = float(np.mean(values[labels == 1] > 0))
        loss_recall = float(np.mean(values[labels == -1] < 0))
        rec["value_head"] = {
            "counts": {"win": int((labels == 1).sum()), "draw": int((labels == 0).sum()), "loss": int((labels == -1).sum())},
            "balanced_sign_accuracy_decisive": round((win_recall + loss_recall) / 2, 4),
            "win_recall": round(win_recall, 4),
            "loss_recall": round(loss_recall, 4),
            "mean_value": {k: round(float(values[labels == v].mean()), 4) for k, v in (("win", 1), ("draw", 0), ("loss", -1))},
        }
    return rec


def score(args) -> None:
    positions = json.loads(Path(args.positions).read_text())
    jobs = [(m, s, positions) for m in args.model for s in args.simulations]
    jobs += [(b, None, positions) for b in args.baselines]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = list(pool.imap(_score_job, jobs))
    for r in results:
        print(json.dumps(r))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)
    b = sub.add_parser("build")
    b.add_argument("--n", type=int, default=600)
    b.add_argument("--min-ply", type=int, default=10)
    b.add_argument("--max-ply", type=int, default=34)
    b.add_argument("--seed", type=int, default=2024)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--out", required=True)
    b.set_defaults(fn=build)
    s = sub.add_parser("score")
    s.add_argument("--positions", required=True)
    s.add_argument("--model", action="append", required=True)
    s.add_argument("--simulations", type=int, nargs="+", default=[0, 50, 200, 800])
    s.add_argument("--baselines", nargs="*", default=["random", "greedy", "minimax-d2", "minimax-d4"])
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
