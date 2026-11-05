"""Tiny sampled-trajectory checks for first-visit returns and TD termination."""
import unittest
from unittest.mock import patch
import numpy as np
import solution
import expanded_study


class Episode:
    def __init__(self, rewards, truncated=False):
        self.rewards = rewards
        self.truncated = truncated

    def reset(self, seed=None):
        self.index = 0
        return 0, {}

    def step(self, action):
        reward = self.rewards[self.index]
        self.index += 1
        done = self.index == len(self.rewards)
        return 0, reward, done and not self.truncated, done and self.truncated, {}

    def close(self):
        pass


class UpdateTests(unittest.TestCase):
    def test_mc_first_visit_discounted_return(self):
        config = {**solution.DEFAULT, "method": "mc", "gamma": 0.995, "mc_rate": "mean"}
        with patch.object(expanded_study, "environment", return_value=Episode([0, 1])), patch.object(expanded_study, "action", return_value=0):
            q, info = expanded_study.train_mc(config, 123, episodes=1)
        self.assertAlmostEqual(q[0, 0], 0.995)
        self.assertEqual(info["transitions"], 2)
        self.assertEqual(info["completed_episodes"], 1)
        self.assertEqual(np.count_nonzero(q), 1)

    def test_td_no_bootstrap_on_ending(self):
        config = {**solution.DEFAULT, "initial_q": 0.5}
        rate = config["alpha"] / (1 + 1 / config["alpha_scale"]) ** config["alpha_power"]
        for truncated in (False, True):
            with self.subTest(truncated=truncated), patch.object(solution, "environment", return_value=Episode([0], truncated)), patch.object(solution, "action", return_value=0):
                q, info = solution.train(config, 123, steps=1)
                self.assertAlmostEqual(q[0, 0], 0.5 - rate * 0.5)
                self.assertEqual(info["transitions"], 1)


if __name__ == "__main__":
    unittest.main()
