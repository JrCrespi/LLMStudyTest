"""Connect 4 rules on a 64-bit bitboard.

Layout (column-major, one sentinel bit on top of each column):

     6 13 20 27 34 41 48
     5 12 19 26 33 40 47   <- row 5 (top)
     4 11 18 25 32 39 46
     3 10 17 24 31 38 45
     2  9 16 23 30 37 44
     1  8 15 22 29 36 43
     0  7 14 21 28 35 42   <- row 0 (bottom)

The state is always stored from the point of view of the player to move:
``current`` holds that player's stones and ``mask`` holds every stone.
This makes the neural-network input and the value target symmetric for
both players.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WIDTH = 7
HEIGHT = 6
H1 = HEIGHT + 1
N_CELLS = WIDTH * HEIGHT

_BOTTOM = [1 << (c * H1) for c in range(WIDTH)]
_TOP = [1 << (HEIGHT - 1 + c * H1) for c in range(WIDTH)]
_COLUMN = [((1 << HEIGHT) - 1) << (c * H1) for c in range(WIDTH)]
_BOARD = sum(_COLUMN)
_BOTTOM_ROW = sum(_BOTTOM)


def winning_cells(pos: int, mask: int) -> int:
    """Empty cells that would complete four-in-a-row for the stones in ``pos``.

    Bit-parallel version of the classic solver trick (P. Pons): every
    direction is checked for all cells at once with shifts.
    """
    r = (pos << 1) & (pos << 2) & (pos << 3)  # vertical (only upwards is possible)
    for d in (H1, HEIGHT, H1 + 1):  # horizontal and the two diagonals
        p = (pos << d) & (pos << 2 * d)
        r |= p & (pos << 3 * d)
        r |= p & (pos >> d)
        p = (pos >> d) & (pos >> 2 * d)
        r |= p & (pos << d)
        r |= p & (pos >> 3 * d)
    return r & (_BOARD ^ mask)


def cells_to_columns(cells: int) -> list[int]:
    return [c for c in range(WIDTH) if cells & _COLUMN[c]]


def has_four(pos: int) -> bool:
    """True if the bitboard ``pos`` contains four aligned stones."""
    for shift in (1, H1, H1 - 1, H1 + 1):  # vertical, horizontal, two diagonals
        m = pos & (pos >> shift)
        if m & (m >> (2 * shift)):
            return True
    return False


@dataclass(frozen=True, slots=True)
class State:
    current: int = 0  # stones of the player to move
    mask: int = 0  # all stones
    moves: int = 0  # plies played so far
    winner_last: bool = False  # True if the previous move won the game

    # ------------------------------------------------------------------ rules
    def can_play(self, col: int) -> bool:
        return not self.winner_last and (self.mask & _TOP[col]) == 0

    def legal_moves(self) -> list[int]:
        if self.is_terminal():
            return []
        return [c for c in range(WIDTH) if (self.mask & _TOP[c]) == 0]

    def legal_mask(self) -> np.ndarray:
        out = np.zeros(WIDTH, dtype=bool)
        for c in self.legal_moves():
            out[c] = True
        return out

    def play(self, col: int) -> "State":
        if not self.can_play(col):
            raise ValueError(f"illegal move {col}")
        move_bit = (self.mask + _BOTTOM[col]) & _COLUMN[col]
        mover = self.current | move_bit
        new_mask = self.mask | move_bit
        return State(
            current=mover ^ new_mask,  # opponent becomes the player to move
            mask=new_mask,
            moves=self.moves + 1,
            winner_last=has_four(mover),
        )

    def is_winning_move(self, col: int) -> bool:
        if not self.can_play(col):
            return False
        move_bit = (self.mask + _BOTTOM[col]) & _COLUMN[col]
        return has_four(self.current | move_bit)

    def playable_cells(self) -> int:
        """One bit per non-full column: the cell where the next stone would land."""
        return (self.mask + _BOTTOM_ROW) & _BOARD

    def is_terminal(self) -> bool:
        return self.winner_last or self.moves == N_CELLS

    def terminal_value(self) -> float:
        """Outcome for the player to move at a terminal state (-1 loss, 0 draw)."""
        return -1.0 if self.winner_last else 0.0

    # --------------------------------------------------------------- encoding
    def planes(self) -> np.ndarray:
        """Board as (2, HEIGHT, WIDTH) float32: [own stones, opponent stones].

        Row 0 of the array is the top of the board (as it is drawn).
        """
        own = _bits_to_grid(self.current)
        opp = _bits_to_grid(self.current ^ self.mask)
        return np.stack([own, opp]).astype(np.float32)

    def player_to_move(self) -> int:
        """1 for the first player, 2 for the second."""
        return 1 + (self.moves % 2)

    def key(self) -> tuple[int, int]:
        return (self.current, self.mask)

    # ----------------------------------------------------------------- display
    def render(self, first_player_symbol: str = "X", second_player_symbol: str = "O") -> str:
        own = _bits_to_grid(self.current)
        opp = _bits_to_grid(self.current ^ self.mask)
        if self.moves % 2 == 0:
            p1, p2 = own, opp
        else:
            p1, p2 = opp, own
        lines = []
        for r in range(HEIGHT):
            row = []
            for c in range(WIDTH):
                row.append(first_player_symbol if p1[r, c] else second_player_symbol if p2[r, c] else ".")
            lines.append(" ".join(row))
        lines.append(" ".join(str(c) for c in range(WIDTH)))
        return "\n".join(lines)

    @staticmethod
    def from_moves(moves: str | list[int]) -> "State":
        s = State()
        for m in moves:
            s = s.play(int(m))
        return s


def _bits_to_grid(bits: int) -> np.ndarray:
    raw = np.unpackbits(
        np.frombuffer(bits.to_bytes(8, "little"), dtype=np.uint8), bitorder="little"
    )[: WIDTH * H1]
    grid = raw.reshape(WIDTH, H1)[:, :HEIGHT]  # (col, row-from-bottom)
    return grid.T[::-1].copy()  # (row-from-top, col)


_SHIFTS = np.array(
    [[c * H1 + (HEIGHT - 1 - r) for c in range(WIDTH)] for r in range(HEIGHT)], dtype=np.uint64
)  # bit index of (row-from-top, col)
_TOP_ARR = np.array(_TOP, dtype=np.uint64)


def encode_batch(states: list["State"]) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ``planes()`` and ``legal_mask()`` for a list of states."""
    cur = np.array([s.current for s in states], dtype=np.uint64)
    mask = np.array([s.mask for s in states], dtype=np.uint64)
    pair = np.stack([cur, cur ^ mask], axis=1)  # (B, 2)
    planes = ((pair[:, :, None, None] >> _SHIFTS) & np.uint64(1)).astype(np.float32)
    legal = (mask[:, None] & _TOP_ARR) == 0
    return planes, legal


def flip_planes(planes: np.ndarray) -> np.ndarray:
    """Mirror a (..., HEIGHT, WIDTH) array left/right (Connect 4 is symmetric)."""
    return planes[..., ::-1].copy()


def flip_policy(policy: np.ndarray) -> np.ndarray:
    return policy[..., ::-1].copy()
