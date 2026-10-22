"""Plot training curves from <run>/metrics.jsonl.

    python scripts/plot_training.py runs/main/metrics.jsonl --out results/training_curves.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.metrics)]
    it = [r["iteration"] + 1 for r in rows]
    trained = [r for r in rows if r["train"]]
    t_it = [r["iteration"] + 1 for r in trained]

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))

    a = ax[0, 0]
    a.plot(t_it, [r["train"]["policy_loss"] for r in trained], label="policy loss (cross-entropy)")
    a.plot(t_it, [r["train"]["value_loss"] for r in trained], label="value loss (MSE)")
    a.plot(t_it, [r["train"]["policy_kl"] for r in trained], label="KL(search policy || network)", ls="--")
    a.set_title("Training losses")
    a.set_xlabel("iteration")
    a.legend(frameon=False, fontsize=8)

    a = ax[0, 1]
    evals = [r for r in rows if "eval" in r]
    keys = list(evals[0]["eval"]) if evals else []
    for k in keys:
        a.plot([r["iteration"] + 1 for r in evals], [r["eval"][k]["score"] for r in evals], marker="o", label=k)
    a.set_ylim(-0.02, 1.02)
    a.set_title("Score vs reference opponents (win=1, draw=0.5)")
    a.set_xlabel("iteration")
    a.legend(frameon=False, fontsize=8)

    a = ax[1, 0]
    g = [r["selfplay"]["games"] for r in rows]
    a.plot(it, [r["selfplay"]["first_player_wins"] / n for r, n in zip(rows, g)], label="first player wins")
    a.plot(it, [r["selfplay"]["second_player_wins"] / n for r, n in zip(rows, g)], label="second player wins")
    a.plot(it, [r["selfplay"]["draws"] / n for r, n in zip(rows, g)], label="draws")
    a.set_title("Self-play outcomes")
    a.set_xlabel("iteration")
    a.legend(frameon=False, fontsize=8)

    a = ax[1, 1]
    a.plot(it, [r["selfplay"]["mean_length"] for r in rows])
    a.set_title("Mean self-play game length (plies)")
    a.set_xlabel("iteration")

    for a in ax.flat:
        a.grid(alpha=0.3)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)


if __name__ == "__main__":
    main()
