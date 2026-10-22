"""Tactical accuracy: does the model find forced moves?

Builds a fixed set of positions (from games between noisy minimax players) where
the side to move either has an immediate win, or must block the opponent's only
immediate win. Reports how often the raw policy network and MCTS pick a correct
move. This isolates "learned tactics" from overall playing strength.

    python scripts/tactics.py --model models/connect4_zero.pt --out results/tactics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from connect4_zero.baselines import MinimaxPlayer  # noqa: E402
from connect4_zero.game import State  # noqa: E402
from connect4_zero.mcts import MCTSConfig, MCTSPlayer  # noqa: E402
from connect4_zero.model import Evaluator, load_checkpoint  # noqa: E402


def opponent_view(s: State) -> State:
    return State(current=s.current ^ s.mask, mask=s.mask, moves=s.moves)


def build_positions(n: int, seed: int) -> list[tuple[State, str, set[int]]]:
    rng = np.random.default_rng(seed)
    players = [MinimaxPlayer(2), MinimaxPlayer(3)]
    out, seen = [], set()
    while len(out) < n:
        s = State()
        while not s.is_terminal() and len(out) < n:
            legal = s.legal_moves()
            wins = {c for c in legal if s.is_winning_move(c)}
            threats = {c for c in legal if opponent_view(s).is_winning_move(c)}
            if s.key() not in seen:
                if wins:
                    out.append((s, "win", wins))
                    seen.add(s.key())
                elif len(threats) == 1:
                    out.append((s, "block", threats))
                    seen.add(s.key())
            # 30% random moves keep the positions diverse
            move = int(rng.choice(legal)) if rng.random() < 0.3 else players[s.moves % 2](s, rng)
            s = s.play(move)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--positions", type=int, default=500)
    ap.add_argument("--simulations", type=int, nargs="+", default=[0, 50, 200])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    positions = build_positions(args.positions, args.seed)
    kinds = {k: sum(p[1] == k for p in positions) for k in ("win", "block")}
    results = []
    for model in args.model:
        net, _ = load_checkpoint(model)
        ev = Evaluator(net)
        for sims in args.simulations:
            player = MCTSPlayer(ev, MCTSConfig(simulations=sims))
            correct = {"win": 0, "block": 0}
            for s, kind, good in positions:
                pi = player.policy(s)
                if int(np.argmax(pi)) in good:
                    correct[kind] += 1
            rec = {
                "model": model,
                "simulations": sims,
                "win_in_1_accuracy": round(correct["win"] / max(kinds["win"], 1), 4),
                "block_accuracy": round(correct["block"] / max(kinds["block"], 1), 4),
                "positions": kinds,
            }
            print(json.dumps(rec), flush=True)
            results.append(rec)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
