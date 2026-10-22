"""Python wrapper around the exact C solver in solver/c4solver.c.

The shared library is compiled on first use (needs a C compiler such as gcc).
Used only for evaluation: it gives ground truth for "is this move optimal?".
"""
from __future__ import annotations

import ctypes
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from .game import N_CELLS, State

_SRC = Path(__file__).resolve().parents[1] / "solver" / "c4solver.c"
_LIB = _SRC.parent / "build" / "libc4solver.so"


@lru_cache(maxsize=1)
def _lib() -> ctypes.CDLL:
    if not _LIB.exists() or _LIB.stat().st_mtime < _SRC.stat().st_mtime:
        cc = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
        if cc is None:
            raise RuntimeError("a C compiler is required to build the solver")
        _LIB.parent.mkdir(exist_ok=True)
        subprocess.run([cc, "-O3", "-shared", "-fPIC", "-o", str(_LIB), str(_SRC)], check=True)
    lib = ctypes.CDLL(str(_LIB))
    lib.c4_solve.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]
    lib.c4_solve.restype = ctypes.c_int
    lib.c4_nodes.restype = ctypes.c_uint64
    return lib


def solve(state: State) -> int:
    """Exact score for the side to move (0 draw, >0 win, <0 loss; larger = faster)."""
    if state.is_terminal():
        return -((N_CELLS + 2 - state.moves) // 2) if state.winner_last else 0
    return _lib().c4_solve(state.current, state.mask, state.moves)


def move_scores(state: State) -> dict[int, int]:
    """Exact score of every legal move, from the point of view of the side to move."""
    out = {}
    for c in state.legal_moves():
        if state.is_winning_move(c):
            out[c] = (N_CELLS + 1 - state.moves) // 2
        else:
            out[c] = -solve(state.play(c))
    return out


def outcome(score: int) -> int:
    return (score > 0) - (score < 0)


def reset() -> None:
    _lib().c4_reset()


class SolverPlayer:
    """Perfect player: picks uniformly among the moves with the best exact score."""

    name = "perfect-solver"

    def __call__(self, state: State, rng) -> int:
        scores = move_scores(state)
        best = max(scores.values())
        return int(rng.choice([c for c, v in scores.items() if v == best]))

