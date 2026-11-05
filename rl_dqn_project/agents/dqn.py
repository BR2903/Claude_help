"""Deep Q-Network (Mnih et al., 2015) with switches for the ablation study.

Switches (all are plain constructor arguments, so a config dict fully describes a run):
  use_target_network=False  -> targets are computed with the SAME network being optimized
                               (the "moving target" ablation).
  use_replay_buffer=False   -> "online DQN": one gradient step per environment step on the
                               transition that just happened (batch of 1, strictly sequential,
                               highly correlated data) -- breaks the i.i.d. assumption.
  double=True               -> Double DQN target (van Hasselt et al., 2016): online net SELECTS
                               a' = argmax Q_online(s', .), target net EVALUATES Q_target(s', a').
  dueling=True              -> Dueling head (Wang et al., 2016): Q = V(s) + A(s,a) - mean_a A(s,a).
  tau < 1                   -> soft (Polyak) target update every step instead of a hard copy.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.schedules import LinearSchedule


# =============================================================================
# Network
# =============================================================================
class QNetwork(nn.Module):
    """MLP: obs -> Q(s, a) for every discrete action (one output node per action)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: Sequence[int] = (128, 128), dueling: bool = False):
        super().__init__()
        layers, last = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.body = nn.Sequential(*layers)
        self.dueling = dueling
        if dueling:
            self.value = nn.Linear(last, 1)              # V(s)
            self.advantage = nn.Linear(last, n_actions)  # A(s, a)
        else:
            self.head = nn.Linear(last, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.body(x)
        if self.dueling:
            a = self.advantage(z)
            # subtract the mean advantage so V and A are identifiable
            return self.value(z) + a - a.mean(dim=1, keepdim=True)
        return self.head(z)


# =============================================================================
# Experience replay
# =============================================================================
class ReplayBuffer:
    """Fixed-size circular buffer stored as pre-allocated NumPy arrays.

    Sampling uniformly at random from many past transitions (1) de-correlates consecutive
    samples, making mini-batches closer to i.i.d., and (2) re-uses each transition many times.
    """

    def __init__(self, capacity: int, obs_dim: int, rng: np.random.Generator):
        self.capacity, self.rng = capacity, rng
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)  # 1.0 only for TRUE terminals
        self.ptr, self.size = 0, 0

    def add(self, obs, action, reward, next_obs, terminated: bool) -> None:
        i = self.ptr
        self.obs[i], self.next_obs[i] = obs, next_obs
        self.actions[i], self.rewards[i], self.dones[i] = action, reward, float(terminated)
        self.ptr = (self.ptr + 1) % self.capacity  # overwrite the oldest when full
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (torch.from_numpy(self.obs[idx]), torch.from_numpy(self.actions[idx]),
                torch.from_numpy(self.rewards[idx]), torch.from_numpy(self.next_obs[idx]),
                torch.from_numpy(self.dones[idx]))

    def __len__(self) -> int:
        return self.size


# =============================================================================
# Agent
# =============================================================================
class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        seed: int,
        hidden: Sequence[int] = (128, 128),
        lr: float = 5e-4,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        learning_starts: int = 1_000,
        train_freq: int = 1,
        target_update_freq: int = 500,
        tau: float = 1.0,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 25_000,
        grad_clip: float = 10.0,
        use_target_network: bool = True,
        use_replay_buffer: bool = True,
        double: bool = False,
        dueling: bool = False,
    ):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.n_actions, self.gamma = n_actions, gamma
        self.batch_size, self.learning_starts = batch_size, learning_starts
        self.train_freq, self.target_update_freq, self.tau = train_freq, target_update_freq, tau
        self.grad_clip = grad_clip
        self.use_target, self.use_replay, self.double = use_target_network, use_replay_buffer, double
        self.eps = LinearSchedule(eps_start, eps_end, eps_decay_steps)

        self.q = QNetwork(obs_dim, n_actions, hidden, dueling)
        self.q_target = QNetwork(obs_dim, n_actions, hidden, dueling)
        self.q_target.load_state_dict(self.q.state_dict())  # start identical
        self.q_target.requires_grad_(False)
        self.optim = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, obs_dim, self.rng) if use_replay_buffer else None
        self.n_updates = 0
        self._stats = {"loss": [], "q_mean": [], "q_max": [], "grad_norm": []}

    # ---- acting ---------------------------------------------------------- #
    @torch.no_grad()
    def greedy(self, obs: np.ndarray) -> int:
        """argmax_a Q(s, a); weights are not touched (used for frozen-weight evaluation)."""
        q = self.q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def act(self, obs: np.ndarray, t: int) -> int:
        if self.rng.random() < self.eps(t):
            return int(self.rng.integers(self.n_actions))
        return self.greedy(obs)

    # ---- learning -------------------------------------------------------- #
    def observe(self, obs, action, reward, next_obs, terminated: bool, truncated: bool, t: int) -> None:
        if self.use_replay:
            self.buffer.add(obs, action, reward, next_obs, terminated)
            if t >= self.learning_starts and t % self.train_freq == 0:
                self._update(*self.buffer.sample(self.batch_size))
        else:
            # Online DQN: learn from the single transition that just happened, then forget it.
            if t >= self.learning_starts:
                batch = (torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
                         torch.as_tensor([action]),
                         torch.as_tensor([reward], dtype=torch.float32),
                         torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0),
                         torch.as_tensor([float(terminated)]))
                self._update(*batch)
        self._maybe_update_target(t)

    def _update(self, s, a, r, s2, done) -> None:
        # Q(s, a) for the actions ACTUALLY TAKEN in the batch.
        # self.q(s) has shape [B, n_actions]; gather(1, a[:, None]) picks column a_i in row i -> [B, 1].
        q_all = self.q(s)
        q_sa = q_all.gather(1, a.unsqueeze(1)).squeeze(1)

        y = self.td_targets(r, s2, done)
        loss = F.smooth_l1_loss(q_sa, y)  # Huber loss: quadratic near 0, linear for big TD errors
        self.optim.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.optim.step()
        self.n_updates += 1

        self._stats["loss"].append(loss.item())
        self._stats["q_mean"].append(q_sa.mean().item())
        self._stats["q_max"].append(q_all.max().item())
        self._stats["grad_norm"].append(float(grad_norm))

    @torch.no_grad()  # the target is treated as a constant (semi-gradient TD)
    def td_targets(self, r, s2, done) -> torch.Tensor:
        target_net = self.q_target if self.use_target else self.q
        if self.double:
            a2 = self.q(s2).argmax(dim=1, keepdim=True)          # online net selects
            q_next = target_net(s2).gather(1, a2).squeeze(1)     # target net evaluates
        else:
            q_next = target_net(s2).max(dim=1).values            # max_a' Q_target(s', a')
        return r + self.gamma * (1.0 - done) * q_next            # no bootstrap after a true terminal

    def _maybe_update_target(self, t: int) -> None:
        if not self.use_target:
            return
        if self.tau < 1.0:  # soft / Polyak: theta_target <- tau*theta + (1 - tau)*theta_target
            with torch.no_grad():
                for p_t, p in zip(self.q_target.parameters(), self.q.parameters()):
                    p_t.mul_(1.0 - self.tau).add_(p, alpha=self.tau)
        elif t % self.target_update_freq == 0:  # hard copy every target_update_freq env steps
            self.q_target.load_state_dict(self.q.state_dict())

    def pop_stats(self) -> dict:
        out = {k: (float(np.mean(v)) if v else float("nan")) for k, v in self._stats.items()}
        for v in self._stats.values():
            v.clear()
        return out

    def save(self, path: str) -> None:
        torch.save(self.q.state_dict(), path)
