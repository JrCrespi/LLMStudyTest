import numpy as np
import torch

from connect4_zero.arena import MatchResult, match, play_game
from connect4_zero.baselines import GreedyPlayer, MinimaxPlayer, RandomPlayer
from connect4_zero.game import State
from connect4_zero.mcts import MCTSConfig, MCTSPlayer, Node, run_mcts
from connect4_zero.model import Evaluator, PolicyValueNet, load_checkpoint, save_checkpoint
from connect4_zero.selfplay import SelfPlayConfig, self_play


def uniform(states):
    priors = np.stack([s.legal_mask() / s.legal_mask().sum() for s in states])
    return priors, np.zeros(len(states))


def test_mcts_takes_immediate_win():
    s = State.from_moves("010203")  # first player threatens column 0
    p = MCTSPlayer(uniform, MCTSConfig(simulations=300))
    assert p(s, np.random.default_rng(0)) == 0


def test_mcts_blocks_immediate_loss():
    s = State.from_moves("01020")  # second player must block column 0
    p = MCTSPlayer(uniform, MCTSConfig(simulations=300))
    assert p(s, np.random.default_rng(0)) == 0


def test_visit_counts_are_consistent():
    root = Node(State())
    run_mcts([root], uniform, MCTSConfig(simulations=100))
    # every simulation adds one visit to exactly one root edge
    assert sum(root.N) == 100
    assert root.visits == 101  # + the initial root evaluation


def test_batched_search_equals_independent_roots():
    roots = [Node(State.from_moves("010203")), Node(State.from_moves("01020"))]
    run_mcts(roots, uniform, MCTSConfig(simulations=200))
    assert int(np.argmax(roots[0].N)) == 0
    assert int(np.argmax(roots[1].N)) == 0


def test_minimax_finds_win_and_block():
    rng = np.random.default_rng(0)
    assert MinimaxPlayer(4)(State.from_moves("010203"), rng) == 0
    assert MinimaxPlayer(4)(State.from_moves("01020"), rng) == 0


def test_minimax_ladder_is_ordered():
    # a deeper search should not lose to a shallower one on balance
    r = match(MinimaxPlayer(4), MinimaxPlayer(1), pairs=10, seed=0)
    assert r.score > 0.6
    r = match(GreedyPlayer(), RandomPlayer(), pairs=20, seed=0)
    assert r.score > 0.7
    assert r.games == 40 and len(r.as_first) == len(r.as_second) == 20


def test_play_game_outcome_sign():
    rng = np.random.default_rng(0)
    # minimax as first player against a random opponent should win
    assert play_game(MinimaxPlayer(3), RandomPlayer(), rng) == 1
    assert play_game(RandomPlayer(), MinimaxPlayer(3), rng) == -1


def test_match_result_statistics():
    r = MatchResult(pairs=[(1.0, 1.0)] * 10 + [(1.0, 0.5)] * 5 + [(0.0, 0.5)] * 5)
    assert r.games == 40 and r.wins == 25 and r.draws == 10 and r.losses == 5
    lo, hi = r.score_ci()
    assert lo < r.score < hi
    perfect = MatchResult(pairs=[(1.0, 1.0)] * 49)
    lo, hi = perfect.score_ci()
    assert hi >= 1.0 - 1e-9 and 0.9 < lo < 1.0  # not a degenerate [1, 1]
    assert perfect.as_dict()["elo_ci95"][0] > 0
    # a slightly worse result must not get a better lower bound
    near = MatchResult(pairs=[(1.0, 1.0)] * 47 + [(1.0, 0.5), (1.0, 0.0)])
    assert near.score_ci()[0] < lo
    # all pairs equal at 0.5: still a non-degenerate interval
    lo, hi = MatchResult(pairs=[(1.0, 0.0)] * 30).score_ci()
    assert lo < 0.5 < hi


def test_openings_are_not_repeated():
    seen = []

    def recorder(state, rng):
        if state.moves == 2:
            seen.append(state.key())
        return RandomPlayer()(state, rng)

    match(recorder, RandomPlayer(), pairs=49, seed=3)
    assert len(seen) == 49 and len(set(seen)) == 49  # each opening once (agent moves first once per pair)


def test_network_shapes_and_masking():
    net = PolicyValueNet()
    ev = Evaluator(net)
    states = [State(), State.from_moves("000000")]
    priors, values = ev(states)
    assert priors.shape == (2, 7) and values.shape == (2,)
    assert np.allclose(priors.sum(1), 1, atol=1e-5)
    assert priors[1, 0] == 0  # full column gets no probability
    assert np.all(np.abs(values) <= 1)


def test_evaluator_cache_returns_same_result():
    ev = Evaluator(PolicyValueNet())
    s = State.from_moves("3")
    p1, v1 = ev([s])
    p2, v2 = ev([s, s])
    assert ev.hits >= 2
    assert np.allclose(p1[0], p2[0]) and np.allclose(p2[0], p2[1])
    assert np.isclose(v1[0], v2[1])


def test_self_play_targets():
    samples, stats = self_play(uniform, SelfPlayConfig(games=4, full_search_prob=1.0, mcts=MCTSConfig(simulations=20)), np.random.default_rng(0))
    assert stats["games"] == 4 and len(samples) == stats["positions"]
    for smp in samples:
        assert smp.planes.shape == (2, 6, 7)
        assert abs(smp.policy.sum() - 1) < 1e-5
        assert smp.value in (-1.0, 0.0, 1.0)
    # the last position of a decisive game is the winner to move: value +1
    assert samples[-1].value in (1.0, 0.0)


def test_checkpoint_roundtrip(tmp_path):
    net = PolicyValueNet()
    save_checkpoint(tmp_path / "m.pt", net, {"iteration": 3})
    net2, ckpt = load_checkpoint(tmp_path / "m.pt")
    assert ckpt["iteration"] == 3
    x = torch.from_numpy(State.from_moves("33").planes()[None])
    net.eval()
    with torch.inference_mode():
        assert torch.allclose(net(x)[1], net2(x)[1])


def test_search_runs_all_simulations_after_reaching_terminal_nodes():
    # Regression: a simulation that ends in a terminal node must not stop the search.
    for prior in (True, False):
        root = Node(State.from_moves("010203"))
        run_mcts([root], uniform, MCTSConfig(simulations=150, tactical_prior=prior))
        assert sum(root.N) == 150


def test_pure_search_finds_tactics_without_tactical_prior():
    cfg = MCTSConfig(simulations=400, tactical_prior=False)
    rng = np.random.default_rng(1)
    assert MCTSPlayer(uniform, cfg)(State.from_moves("010203"), rng) == 0
    assert MCTSPlayer(uniform, cfg)(State.from_moves("01020"), rng) == 0
    # same tactics mirrored, so a column-0 tie-break cannot make the test pass by luck
    assert MCTSPlayer(uniform, cfg)(State.from_moves("656463"), rng) == 6
    assert MCTSPlayer(uniform, cfg)(State.from_moves("65646"), rng) == 6


def test_tactical_moves():
    from connect4_zero.mcts import tactical_moves

    assert tactical_moves(State()) is None
    assert tactical_moves(State.from_moves("010203")) == [0]  # win
    assert tactical_moves(State.from_moves("01020")) == [0]  # block
