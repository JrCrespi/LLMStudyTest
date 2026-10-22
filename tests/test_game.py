import numpy as np
import pytest

from connect4_zero.game import HEIGHT, N_CELLS, WIDTH, State, flip_planes, has_four


def test_empty_board():
    s = State()
    assert s.legal_moves() == list(range(WIDTH))
    assert not s.is_terminal()
    assert s.planes().shape == (2, HEIGHT, WIDTH)
    assert s.planes().sum() == 0


def test_stones_fall_to_the_bottom_and_perspective_switches():
    s = State.from_moves([3])
    planes = s.planes()
    # After one move the side to move is the second player: its own plane is empty.
    assert planes[0].sum() == 0
    assert planes[1][HEIGHT - 1, 3] == 1
    s = s.play(3)
    planes = s.planes()
    assert planes[0][HEIGHT - 1, 3] == 1  # first player's stone, first player to move
    assert planes[1][HEIGHT - 2, 3] == 1


@pytest.mark.parametrize(
    "moves",
    [
        "0101010",  # vertical
        "0011223",  # horizontal
        "01122323633",  # diagonal /
        "65544343033",  # diagonal \
    ],
)
def test_wins_in_every_direction(moves):
    s = State.from_moves(moves)
    assert s.is_terminal()
    assert s.terminal_value() == -1.0  # the side to move has lost
    assert s.legal_moves() == []


def test_no_false_win_across_column_wrap():
    # Stones at the top of column 0 and bottom of column 1 must not connect.
    s = State.from_moves("0000001")
    assert not s.is_terminal()


def test_full_column_is_illegal():
    s = State.from_moves("000000")
    assert 0 not in s.legal_moves()
    with pytest.raises(ValueError):
        s.play(0)


def test_full_board_draw():
    # Column pattern that fills the board without four in a row.
    order = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 2, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 2,
             4, 5, 4, 5, 4, 5, 5, 4, 5, 4, 5, 4, 6, 6, 6, 6, 6, 6]
    s = State()
    for m in order:
        assert not s.is_terminal()
        s = s.play(m)
    assert s.moves == N_CELLS
    assert s.is_terminal() and s.terminal_value() == 0.0


def test_is_winning_move_matches_play():
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = State()
        while not s.is_terminal():
            for c in s.legal_moves():
                assert s.is_winning_move(c) == s.play(c).winner_last
            s = s.play(int(rng.choice(s.legal_moves())))


def test_has_four_direct():
    assert has_four(0b1111)
    assert not has_four(0b111)


def test_flip_is_involution():
    p = State.from_moves("0123").planes()
    assert np.array_equal(flip_planes(flip_planes(p)), p)
    assert np.array_equal(flip_planes(p)[..., 0], p[..., WIDTH - 1])
