"""Final evaluation of a trained model (and of the checkpoints along the way).

    python scripts/evaluate.py --model models/connect4_zero.pt --games 100 --out results/eval.json
    python scripts/evaluate.py --checkpoints "runs/main/iter_*.pt" --games 40 --out results/progress.json

Each (agent, opponent) match runs in its own single-threaded process.
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from connect4_zero.arena import match  # noqa: E402
from connect4_zero.baselines import GreedyPlayer, MinimaxPlayer, RandomPlayer  # noqa: E402


def make_opponent(name: str):
    if name == "random":
        return RandomPlayer()
    if name == "greedy":
        return GreedyPlayer()
    if name.startswith("minimax-d"):
        return MinimaxPlayer(int(name.removeprefix("minimax-d")))
    raise ValueError(name)


def run_job(job):
    import torch

    from connect4_zero.mcts import MCTSConfig, MCTSPlayer
    from connect4_zero.model import Evaluator, load_checkpoint

    torch.set_num_threads(1)
    model, sims, opp_name, games, seed = job
    net, _ = load_checkpoint(model)
    agent = MCTSPlayer(Evaluator(net), MCTSConfig(simulations=sims))
    t = time.time()
    res = match(agent, make_opponent(opp_name), games, seed=seed).as_dict()
    res.update(model=model, simulations=sims, opponent=opp_name, seconds=round(time.time() - t, 1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--checkpoints", help="glob of checkpoints, e.g. 'runs/main/iter_*.pt'")
    ap.add_argument("--simulations", type=int, nargs="+", default=[0, 200])
    ap.add_argument(
        "--opponents",
        nargs="+",
        default=["random", "greedy", "minimax-d2", "minimax-d4", "minimax-d6"],
    )
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    models = list(args.model)
    if args.checkpoints:
        models += sorted(glob.glob(args.checkpoints))
    jobs = [(m, s, o, args.games, args.seed) for m in models for s in args.simulations for o in args.opponents]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = []
        for r in pool.imap_unordered(run_job, jobs):
            print(json.dumps(r), flush=True)
            results.append(r)
    results.sort(key=lambda r: (r["model"], r["simulations"], args.opponents.index(r["opponent"])))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
