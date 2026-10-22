"""Exact game-theoretic value of every first move, computed with the C solver.

Early positions are the hardest to solve (no opening book), so each one runs
in its own process with a time limit.

    python scripts/solve_openings.py --out results/opening_solver.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _solve_first_move(col: int):
    from connect4_zero import solver
    from connect4_zero.game import State

    t = time.time()
    score = -solver.solve(State.from_moves([col]))  # from the first player's view
    return {"first_move": col, "score_for_first_player": score,
            "outcome": {1: "win", 0: "draw", -1: "loss"}[(score > 0) - (score < 0)],
            "seconds": round(time.time() - t, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeout", type=float, default=3600, help="seconds per first move")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(args.workers) as pool:
        handles = {c: pool.apply_async(_solve_first_move, (c,)) for c in (3, 2, 4, 1, 5, 0, 6)}
        start = time.time()
        for c, h in handles.items():
            try:
                r = h.get(timeout=max(1.0, args.timeout - (time.time() - start)))
            except mp.TimeoutError:
                r = {"first_move": c, "outcome": "timeout"}
            print(json.dumps(r), flush=True)
            results.append(r)
        pool.terminate()
    results.sort(key=lambda r: r["first_move"])
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
