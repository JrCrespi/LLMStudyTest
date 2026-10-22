"""AlphaZero training loop: self-play -> replay buffer -> SGD, with checkpoints.

Strength is measured afterwards on the saved checkpoints (scripts/evaluate.py),
in parallel and with more games than an in-loop evaluation could afford.

Usage:
    python -m connect4_zero.train --out runs/main
    python -m connect4_zero.train --out runs/main --resume
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .game import flip_planes, flip_policy
from .mcts import MCTSConfig
from .model import NetConfig, PolicyValueNet, save_checkpoint
from .selfplay import Sample, SelfPlayConfig, SelfPlayPool


@dataclass
class TrainConfig:
    iterations: int = 100
    games_per_iter: int = 128
    simulations: int = 200  # full (recorded) searches
    fast_simulations: int = 40  # cheap searches that only advance the game
    full_search_prob: float = 0.25
    temperature_moves: int = 12
    # Replay window grows with training so early, near-random games leave quickly.
    window_start: int = 8_000
    window_growth: int = 500  # positions added to the window per iteration
    window_max: int = 30_000
    min_buffer: int = 3_000
    batch_size: int = 256
    sample_reuse: float = 8.0  # each new position is seen ~this many times by SGD
    lr: float = 2e-3
    lr_final: float = 2e-4
    weight_decay: float = 1e-2  # AdamW (decoupled) decay
    # value target = (1 - w) * game outcome z + w * MCTS root Q; w anneals start -> end
    q_weight_start: float = 0.5
    q_weight_end: float = 0.2
    checkpoint_every: int = 10
    seed: int = 0
    threads: int = 4
    workers: int = 4  # self-play processes
    net: NetConfig = field(default_factory=NetConfig)


def _batch(samples: list[Sample], idx: np.ndarray, q_weight: float, rng: np.random.Generator):
    planes = np.stack([samples[i].planes for i in idx])
    pis = np.stack([samples[i].policy for i in idx])
    z = np.array([samples[i].value for i in idx], dtype=np.float32)
    q = np.array([samples[i].root_q for i in idx], dtype=np.float32)
    target_v = (1 - q_weight) * z + q_weight * q
    flip = rng.random(len(idx)) < 0.5  # left/right mirror augmentation
    planes[flip] = flip_planes(planes[flip])
    pis[flip] = flip_policy(pis[flip])
    return torch.from_numpy(planes), torch.from_numpy(pis), torch.from_numpy(target_v)


def train_steps(net, opt, buffer: list[Sample], steps: int, cfg: TrainConfig, q_weight: float, rng) -> dict:
    net.train()
    tot_p = tot_v = tot_ent = 0.0
    for _ in range(steps):
        idx = rng.integers(0, len(buffer), size=cfg.batch_size)
        x, pi, z = _batch(buffer, idx, q_weight, rng)
        logits, v = net(x.contiguous(memory_format=torch.channels_last))
        logp = F.log_softmax(logits, dim=-1)
        loss_p = -(pi * logp).sum(-1).mean()
        loss_v = F.mse_loss(v, z)
        loss = loss_p + loss_v
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tot_p += loss_p.item()
        tot_v += loss_v.item()
        tot_ent += -(pi * torch.log(pi.clamp_min(1e-8))).sum(-1).mean().item()
    n = max(steps, 1)
    # policy loss minus target entropy = KL(pi || p): how far the net is from the search policy
    return {"policy_loss": tot_p / n, "value_loss": tot_v / n, "policy_kl": (tot_p - tot_ent) / n}


def _environment() -> dict:
    import platform
    import subprocess

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = None, None
    return {
        "git_sha": sha,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def _save_buffer(path: Path, buffer) -> None:
    np.savez_compressed(
        path,
        planes=np.stack([s.planes for s in buffer]).astype(np.uint8),
        policy=np.stack([s.policy for s in buffer]),
        value=np.array([s.value for s in buffer], dtype=np.float32),
        root_q=np.array([s.root_q for s in buffer], dtype=np.float32),
    )


def _load_buffer(path: Path) -> list[Sample]:
    d = np.load(path)
    return [
        Sample(p.astype(np.float32), pi, float(v), float(q))
        for p, pi, v, q in zip(d["planes"], d["policy"], d["value"], d["root_q"])
    ]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/latest.pt")
    for f, v in asdict(TrainConfig()).items():
        if f == "net":
            continue
        if isinstance(v, bool):
            raise TypeError(f"add explicit argparse handling for bool field {f}")
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    ap.add_argument("--channels", type=int, default=NetConfig.channels)
    ap.add_argument("--blocks", type=int, default=NetConfig.blocks)
    args = ap.parse_args(argv)

    cfg = TrainConfig(
        **{k: getattr(args, k) for k in asdict(TrainConfig()) if k != "net"},
        net=NetConfig(channels=args.channels, blocks=args.blocks),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    net = PolicyValueNet(cfg.net).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    buffer: deque[Sample] = deque(maxlen=cfg.window_max)
    log_path = out / "metrics.jsonl"
    start_iter = 0
    if args.resume:
        ckpt = torch.load(out / "latest.pt", map_location="cpu", weights_only=True)
        if ckpt["net_config"] != asdict(cfg.net):
            raise ValueError(f"checkpoint net {ckpt['net_config']} != requested {asdict(cfg.net)}")
        net.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        rng.bit_generator.state = json.loads(ckpt["rng_state"])
        buffer.extend(_load_buffer(out / "buffer.npz"))
        start_iter = ckpt["iteration"] + 1
    elif log_path.exists():
        log_path.unlink()
    (out / "config.json").write_text(json.dumps({**asdict(cfg), "environment": _environment()}, indent=2))

    sp_cfg = SelfPlayConfig(
        games=cfg.games_per_iter,
        temperature_moves=cfg.temperature_moves,
        full_search_prob=cfg.full_search_prob,
        fast_simulations=cfg.fast_simulations,
        mcts=MCTSConfig(simulations=cfg.simulations),
    )
    pool = SelfPlayPool(asdict(cfg.net), cfg.workers)
    for it in range(start_iter, cfg.iterations):
        t0 = time.time()
        samples, sp_stats = pool.run(net, sp_cfg, rng)
        buffer.extend(samples)
        t_sp = time.time() - t0

        # cosine learning-rate decay over the whole run
        frac = it / max(cfg.iterations - 1, 1)
        lr = float(cfg.lr_final + 0.5 * (cfg.lr - cfg.lr_final) * (1 + np.cos(np.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr

        t1 = time.time()
        losses = {}
        window = min(cfg.window_start + cfg.window_growth * it, cfg.window_max)
        q_weight = cfg.q_weight_start + (cfg.q_weight_end - cfg.q_weight_start) * frac
        if len(buffer) >= cfg.min_buffer:
            steps = int(np.ceil(cfg.sample_reuse * len(samples) / cfg.batch_size))
            recent = list(itertools.islice(buffer, max(len(buffer) - window, 0), None))
            losses = train_steps(net, opt, recent, steps, cfg, q_weight, rng)
            losses.update(steps=steps, window=len(recent), q_weight=round(q_weight, 3))
        t_tr = time.time() - t1

        rec = {
            "iteration": it,
            "lr": lr,
            "buffer": len(buffer),
            "selfplay": sp_stats,
            "train": losses,
            "time_selfplay_s": round(t_sp, 1),
            "time_train_s": round(t_tr, 1),
        }
        extra = {"iteration": it}
        if (it + 1) % cfg.checkpoint_every == 0 or it == cfg.iterations - 1:
            save_checkpoint(out / f"iter_{it + 1:03d}.pt", net, extra)
        save_checkpoint(
            out / "latest.pt",
            net,
            {**extra, "optimizer": opt.state_dict(), "rng_state": json.dumps(rng.bit_generator.state)},
        )
        _save_buffer(out / "buffer.npz", buffer)
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    pool.close()


if __name__ == "__main__":
    main()
