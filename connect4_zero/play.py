"""Play Connect 4 against a trained model in the terminal.

    python -m connect4_zero.play --model models/connect4_zero.pt --simulations 400
    python -m connect4_zero.play --model models/connect4_zero.pt --ai-first
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .game import State
from .mcts import MCTSConfig, Node, best_move, run_mcts
from .model import Evaluator, load_checkpoint


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/connect4_zero.pt", help="path to a trained checkpoint")
    ap.add_argument("--simulations", type=int, default=400, help="MCTS simulations per move (more = stronger, slower)")
    ap.add_argument("--ai-first", action="store_true", help="let the AI make the first move")
    args = ap.parse_args(argv)

    torch.set_num_threads(4)
    net, _ = load_checkpoint(args.model)
    ev = Evaluator(net)
    cfg = MCTSConfig(simulations=args.simulations)
    rng = np.random.default_rng()
    ai_turn = 0 if args.ai_first else 1
    s = State()
    print("You are", "O" if args.ai_first else "X", "- type a column 0-6, q to quit.\n")
    while not s.is_terminal():
        print(s.render(), "\n")
        if s.moves % 2 == ai_turn:
            root = Node(s)
            run_mcts([root], ev, cfg)
            move = best_move(root, rng)
            win_prob = (1 + sum(root.W) / sum(root.N)) / 2  # root value in [-1, 1] -> [0, 1]
            print(f"AI plays {move}  (visits {root.N}, estimated win chance {win_prob:.0%})\n")
        else:
            raw = input("your move: ").strip()
            if raw.lower() == "q":
                return
            if not raw.isdigit() or not s.can_play(int(raw)):
                print("illegal move\n")
                continue
            move = int(raw)
        s = s.play(move)
    print(s.render(), "\n")
    if not s.winner_last:
        print("Draw.")
    else:
        print("AI wins." if (s.moves - 1) % 2 == ai_turn else "You win!")


if __name__ == "__main__":
    main()
