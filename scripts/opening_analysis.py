"""Does the model rediscover Connect 4 opening theory?

Known result (Allis 1988; Tromp 1988): with perfect play the first player wins,
but only by starting in the centre column. Starting in the columns next to the
centre draws, and any other first move loses.

This script reports, for the empty board and for every first move, the raw
network's policy and value and the MCTS evaluation, next to that theory.

    python scripts/opening_analysis.py --model models/connect4_zero.pt --out results/opening_analysis.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from connect4_zero.game import WIDTH, State  # noqa: E402
from connect4_zero.mcts import MCTSConfig, Node, run_mcts  # noqa: E402
from connect4_zero.model import Evaluator, load_checkpoint  # noqa: E402

# Game-theoretic outcome for the FIRST player after each first move (columns 0..6).
THEORY = [-1, -1, 0, 1, 0, -1, -1]
LABEL = {1: "win", 0: "draw", -1: "loss"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--simulations", type=int, default=1600)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(4)
    net, _ = load_checkpoint(args.model)
    ev = Evaluator(net)
    cfg = MCTSConfig(simulations=args.simulations)

    empty = State()
    priors, value = ev([empty])
    root = Node(empty)
    run_mcts([root], ev, cfg)
    visits = np.array(root.N) / sum(root.N)

    rows = []
    for c in range(WIDTH):
        child = empty.play(c)
        _, v_child = ev([child])
        q = root.W[c] / root.N[c] if root.N[c] else float("nan")
        rows.append(
            {
                "first_move": c,
                "theory_for_first_player": LABEL[THEORY[c]],
                "raw_policy": round(float(priors[0][c]), 4),
                # the child value is from the second player's view: negate it
                "raw_value_for_first_player": round(-float(v_child[0]), 4),
                "mcts_visit_share": round(float(visits[c]), 4),
                "mcts_q_for_first_player": round(float(q), 4),
            }
        )
    out = {
        "model": args.model,
        "simulations": args.simulations,
        "empty_board_raw_value": round(float(value[0]), 4),
        "empty_board_mcts_value": round(root.q(), 4),
        "raw_policy_argmax": int(np.argmax(priors[0])),
        "mcts_choice": int(np.argmax(root.N)),
        "first_moves": rows,
        # Spearman-style check: does the value ordering agree with theory?
        "value_order_agrees_with_theory": bool(
            all(
                rows[a]["raw_value_for_first_player"] > rows[b]["raw_value_for_first_player"]
                for a in range(WIDTH)
                for b in range(WIDTH)
                if THEORY[a] > THEORY[b]
            )
        ),
    }
    print(json.dumps(out, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
