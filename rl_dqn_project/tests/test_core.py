"""Fast sanity tests (run: python -m pytest -q). Each one checks a claim we make in the defense."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from agents.dqn import DQNAgent, QNetwork, ReplayBuffer
from agents.tabular import TabularAgent
from common.runner import run_single
from common.schedules import LinearSchedule
from envs.discretizers import Discretizer, cartpole_discretizer


def test_discretizer_bins_and_clipping():
    d = Discretizer([0.0], [1.0], 4)  # edges at 0.25, 0.5, 0.75
    assert [d(np.array([x])) for x in (-5, 0.1, 0.3, 0.6, 0.9, 99)] == [0, 0, 1, 2, 3, 3]
    assert d.n_states == 4


def test_discretizer_custom_angle_edges():
    d = cartpole_discretizer(bins=10, angle_edges=[0.0])
    assert d.bins == (10, 10, 2, 10)
    left = d.bin_ids(np.array([0, 0, -0.01, 0]))[2]
    right = d.bin_ids(np.array([0, 0, 0.01, 0]))[2]
    assert (left, right) == (0, 1)


def test_linear_schedule():
    s = LinearSchedule(1.0, 0.1, 100)
    assert s(0) == 1.0 and abs(s(50) - 0.55) < 1e-9 and s(100) == s(10_000) == 0.1


def test_tabular_update_terminal_vs_truncated():
    d = Discretizer([0.0], [1.0], 2)
    ag = TabularAgent(d, 2, seed=0, alpha=0.5, gamma=0.9, init_value=0.0)
    ag.Q[1] = [10.0, 0.0]
    s, s2 = np.array([0.1]), np.array([0.9])
    ag.observe(s, 0, 1.0, s2, terminated=True, truncated=False, t=1)
    assert ag.Q[0, 0] == 0.5 * 1.0                     # no bootstrap after a true terminal
    ag.Q[0, 0] = 0.0
    ag.observe(s, 0, 1.0, s2, terminated=False, truncated=True, t=2)
    assert np.isclose(ag.Q[0, 0], 0.5 * (1.0 + 0.9 * 10.0))  # time-limit cut: still bootstrap


def test_gather_picks_taken_actions():
    q = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    a = torch.tensor([2, 0])
    assert q.gather(1, a.unsqueeze(1)).squeeze(1).tolist() == [3.0, 4.0]


def test_dqn_targets_terminal_and_bootstrap():
    ag = DQNAgent(2, 3, seed=0, gamma=0.9)
    s2 = torch.randn(2, 2)
    r = torch.tensor([-1.0, -1.0])
    done = torch.tensor([1.0, 0.0])
    y = ag.td_targets(r, s2, done)
    assert y[0].item() == -1.0                                   # terminal: y = r exactly, no bootstrap
    expected = -1.0 + 0.9 * ag.q_target(s2[1:]).max().item()
    assert abs(y[1].item() - expected) < 1e-6                    # non-terminal: r + gamma * max Q_target
    dbl = DQNAgent(2, 3, seed=0, gamma=0.9, double=True)
    a_star = dbl.q(s2[1:]).argmax().item()
    assert abs(dbl.td_targets(r, s2, done)[1].item() - (-1.0 + 0.9 * dbl.q_target(s2[1:])[0, a_star].item())) < 1e-6


def test_replay_buffer_wraps():
    buf = ReplayBuffer(3, 1, np.random.default_rng(0))
    for i in range(5):
        buf.add([i], 0, float(i), [i + 1], False)
    assert len(buf) == 3 and sorted(buf.rewards.tolist()) == [2.0, 3.0, 4.0]


def test_hard_and_soft_target_updates():
    ag = DQNAgent(4, 2, seed=0, target_update_freq=10)
    with torch.no_grad():
        for p in ag.q.parameters():
            p.add_(1.0)
    ag._maybe_update_target(5)
    assert not torch.equal(ag.q.head.weight, ag.q_target.head.weight)
    ag._maybe_update_target(10)
    assert torch.equal(ag.q.head.weight, ag.q_target.head.weight)

    soft = DQNAgent(4, 2, seed=0, tau=0.5)
    before = soft.q_target.head.weight.clone()
    with torch.no_grad():
        soft.q.head.weight.add_(2.0)
    soft._maybe_update_target(1)
    assert torch.allclose(soft.q_target.head.weight, before + 1.0)


def test_dueling_mean_advantage_is_zero():
    net = QNetwork(4, 3, (16,), dueling=True)
    x = torch.randn(5, 4)
    z = net.body(x)
    a = net.advantage(z)
    assert torch.allclose(net(x), net.value(z) + a - a.mean(1, keepdim=True))


def _tiny(agent, seed, tmp_path):
    spec = {"env_id": "CartPole-v1", "agent": agent, "total_steps": 1_500, "eval_every": 500, "eval_episodes": 2,
            "seed": seed, "weights_path": str(tmp_path / f"{agent}{seed}")}
    if agent == "tabular":
        spec |= {"discretizer": {"kind": "cartpole", "bins": 6}, "agent_kwargs": {}}
    else:
        spec |= {"agent_kwargs": {"learning_starts": 200}}
    return run_single(spec)


def test_runs_are_reproducible(tmp_path):
    for agent in ("tabular", "dqn"):
        a, b = _tiny(agent, 3, tmp_path), _tiny(agent, 3, tmp_path)
        assert a["train_return"] == b["train_return"] and a["eval_return_mean"] == b["eval_return_mean"]
