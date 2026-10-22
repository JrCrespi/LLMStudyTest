"""Plot training curves and evaluation results.

    python scripts/plot_training.py --metrics runs/main/metrics.jsonl --progress results/progress.json \\
        --scaling results/sim_scaling.json --outdir results
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, GREEN, GREY = "#2a6fdb", "#e8833a", "#3a9a5b", "#888888"


def _style(a):
    a.grid(alpha=0.3)
    a.spines[["top", "right"]].set_visible(False)


def training_curves(metrics: str, progress: str | None, out: Path) -> None:
    rows = [json.loads(line) for line in open(metrics)]
    it = [r["iteration"] + 1 for r in rows]
    trained = [r for r in rows if r["train"]]
    t_it = [r["iteration"] + 1 for r in trained]

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))

    a = ax[0, 0]
    if progress:
        res = json.loads(Path(progress).read_text())
        res = [r for r in res if r["opponent"] == "minimax-d4"]
        xs = [int(re.search(r"iter_(\d+)", r["model"]).group(1)) for r in res]
        order = sorted(range(len(xs)), key=xs.__getitem__)
        xs = [xs[i] for i in order]
        sc = [res[i]["score"] for i in order]
        lo = [res[i]["score_ci95"][0] for i in order]
        hi = [res[i]["score_ci95"][1] for i in order]
        a.fill_between(xs, lo, hi, color=BLUE, alpha=0.2, lw=0, label="95% CI (bootstrap over pairs)")
        a.plot(xs, sc, color=BLUE, marker="o", label="MCTS-200 vs minimax depth 4")
        a.axhline(0.5, color=GREY, lw=1, ls="--")
        n = res[0]["games"]
        a.set_title(f"Strength during training ({n} games per point)")
        a.set_ylabel("score (win 1, draw 0.5)")
        a.set_ylim(-0.02, 1.02)
        a.legend(frameon=False, fontsize=8, loc="lower right")
    a.set_xlabel("iteration")

    a = ax[0, 1]
    a.plot(t_it, [r["train"]["policy_loss"] for r in trained], color=BLUE, label="policy loss (cross-entropy)")
    a.plot(t_it, [r["train"]["value_loss"] for r in trained], color=ORANGE, label="value loss (MSE)")
    a.plot(t_it, [r["train"]["policy_kl"] for r in trained], color=GREEN, ls="--", label="KL(search policy || net)")
    a.set_title("Training losses")
    a.set_xlabel("iteration")
    a.legend(frameon=False, fontsize=8)

    a = ax[1, 0]
    g = [r["selfplay"]["games"] for r in rows]
    a.plot(it, [r["selfplay"]["first_player_wins"] / n for r, n in zip(rows, g)], color=BLUE, label="first player wins")
    a.plot(it, [r["selfplay"]["second_player_wins"] / n for r, n in zip(rows, g)], color=ORANGE, label="second player wins")
    a.plot(it, [r["selfplay"]["draws"] / n for r, n in zip(rows, g)], color=GREY, label="draws")
    a.set_title("Self-play outcomes (perfect play: first player wins)")
    a.set_xlabel("iteration")
    a.legend(frameon=False, fontsize=8)

    a = ax[1, 1]
    a.plot(it, [r["selfplay"]["mean_length"] for r in rows], color=BLUE)
    a.set_title("Mean self-play game length (plies)")
    a.set_xlabel("iteration")

    for a in ax.flat:
        _style(a)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def scaling_curve(scaling: str, out: Path) -> None:
    res = sorted(json.loads(Path(scaling).read_text()), key=lambda r: r["simulations"])
    xs = [max(r["simulations"], 1) for r in res]
    fig, a = plt.subplots(figsize=(6, 4))
    a.fill_between(xs, [r["score_ci95"][0] for r in res], [r["score_ci95"][1] for r in res], color=BLUE, alpha=0.2, lw=0)
    a.plot(xs, [r["score"] for r in res], color=BLUE, marker="o")
    a.axhline(0.5, color=GREY, lw=1, ls="--")
    a.set_xscale("log")
    a.set_xticks(xs, ["raw\nnet" if r["simulations"] == 0 else str(r["simulations"]) for r in res])
    a.set_ylim(-0.02, 1.02)
    a.set_xlabel("MCTS simulations per move")
    a.set_ylabel("score vs minimax depth 4")
    a.set_title(f"Search scaling ({res[0]['games']} games per point, 95% CI)")
    _style(a)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--progress")
    ap.add_argument("--scaling")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    training_curves(args.metrics, args.progress, outdir / "training_curves.png")
    if args.scaling:
        scaling_curve(args.scaling, outdir / "sim_scaling.png")


if __name__ == "__main__":
    main()
