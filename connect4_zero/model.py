"""Policy-value residual network (AlphaZero style, scaled down for CPU)."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

from .game import HEIGHT, WIDTH, State, encode_batch


@dataclass
class NetConfig:
    channels: int = 64
    blocks: int = 4
    value_hidden: int = 64


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    """Input: (B, 2, 6, 7) planes from the side to move.

    Output: policy logits (B, 7) and value in [-1, 1] (B,) for the side to move.
    """

    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg or NetConfig()
        ch = self.cfg.channels
        self.stem = nn.Sequential(nn.Conv2d(2, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU())
        self.tower = nn.Sequential(*[ResBlock(ch) for _ in range(self.cfg.blocks)])
        self.p_head = nn.Sequential(nn.Conv2d(ch, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU())
        self.p_fc = nn.Linear(2 * HEIGHT * WIDTH, WIDTH)
        self.v_head = nn.Sequential(nn.Conv2d(ch, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU())
        self.v_fc1 = nn.Linear(HEIGHT * WIDTH, self.cfg.value_hidden)
        self.v_fc2 = nn.Linear(self.cfg.value_hidden, 1)

    def forward(self, x):
        h = self.tower(self.stem(x))
        p = self.p_fc(self.p_head(h).flatten(1))
        v = F.relu(self.v_fc1(self.v_head(h).flatten(1)))
        v = torch.tanh(self.v_fc2(v)).squeeze(-1)
        return p, v


def inference_copy(net: PolicyValueNet) -> nn.Module:
    """Eval-mode copy with BatchNorm folded into the preceding convolutions.

    Folding removes the BN ops at inference time (~1.3x faster on CPU) and,
    because it works on a copy, never touches the mode or memory layout of
    the network being trained.
    """
    m = copy.deepcopy(net).eval()
    with torch.no_grad():
        for seq in (m.stem, m.p_head, m.v_head):
            seq[0] = fuse_conv_bn_eval(seq[0], seq[1])
            seq[1] = nn.Identity()
        for blk in m.tower:
            blk.c1 = fuse_conv_bn_eval(blk.c1, blk.b1)
            blk.b1 = nn.Identity()
            blk.c2 = fuse_conv_bn_eval(blk.c2, blk.b2)
            blk.b2 = nn.Identity()
    return m.to(memory_format=torch.channels_last)


class Evaluator:
    """Batched inference wrapper used by MCTS: states -> (priors, values).

    Results are cached by position, so transpositions and the many repeated
    openings in self-play cost one network call. The evaluator snapshots the
    weights when it is built, so build a new one after every update.
    """

    def __init__(self, net: PolicyValueNet, cache_size: int = 100_000):
        self.net = inference_copy(net)
        self.cache_size = cache_size
        self.cache: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
        self.hits = 0
        self.misses = 0

    @torch.inference_mode()
    def _forward(self, states: list[State]) -> tuple[np.ndarray, np.ndarray]:
        planes, legal = encode_batch(states)
        x = torch.from_numpy(planes).contiguous(memory_format=torch.channels_last)
        logits, v = self.net(x)
        logits = logits.masked_fill(~torch.from_numpy(legal), float("-inf"))
        return torch.softmax(logits, dim=-1).numpy(), v.numpy()

    def __call__(self, states: list[State]) -> tuple[np.ndarray, np.ndarray]:
        priors = np.empty((len(states), WIDTH), dtype=np.float32)
        values = np.empty(len(states), dtype=np.float32)
        todo, todo_idx, seen = [], [], {}
        for i, s in enumerate(states):
            key = s.key()
            hit = self.cache.get(key)
            if hit is not None:
                priors[i], values[i] = hit
                self.hits += 1
            elif key in seen:  # duplicate inside this batch
                todo_idx.append((i, seen[key]))
            else:
                seen[key] = len(todo)
                todo.append(s)
                todo_idx.append((i, len(todo) - 1))
        if todo:
            self.misses += len(todo)
            p, v = self._forward(todo)
            for i, j in todo_idx:
                priors[i], values[i] = p[j], v[j]
            if len(self.cache) + len(todo) > self.cache_size:
                self.cache.clear()
            for s, pj, vj in zip(todo, p, v):
                self.cache[s.key()] = (pj, float(vj))
        return priors, values


def save_checkpoint(path, net: PolicyValueNet, extra: dict | None = None) -> None:
    torch.save({"net_config": asdict(net.cfg), "state_dict": net.state_dict(), **(extra or {})}, path)


def load_checkpoint(path) -> tuple[PolicyValueNet, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    net = PolicyValueNet(NetConfig(**ckpt["net_config"]))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt
