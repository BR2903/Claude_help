"""Independent shortest-path checks for the ASCII warehouse layout."""

from collections import deque
import unittest

import numpy as np
import qlearning4warehouseproblem_template01 as warehouse


# Independently transcribed from the ASCII gaps, not from the reward matrix.
PASSAGES = "AB BC BF CG DH EI FJ GH HL IJ JK KL".split()
NEIGHBORS = {name: set() for name in "ABCDEFGHIJKL"}
for left, right in PASSAGES:
    NEIGHBORS[left].add(right)
    NEIGHBORS[right].add(left)


def shortest_distance(start, goal):
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        current, distance = queue.popleft()
        if current == goal:
            return distance
        for neighbor in NEIGHBORS[current] - seen:
            seen.add(neighbor)
            queue.append((neighbor, distance + 1))
    raise AssertionError("Disconnected reference map")


class WarehouseTests(unittest.TestCase):
    def test_matrix_matches_ascii(self):
        for name, state in warehouse.location_to_state.items():
            actual = {warehouse.state_to_location[int(s)]
                      for s in np.flatnonzero(warehouse.R[state])}
            self.assertEqual(actual, NEIGHBORS[name])

    def test_all_144_public_routes_are_shortest_and_legal(self):
        original = warehouse.R.copy()
        for start in NEIGHBORS:
            for goal in NEIGHBORS:
                with self.subTest(start=start, goal=goal):
                    path = warehouse.route(start, goal)
                    self.assertEqual((path[0], path[-1]), (start, goal))
                    self.assertEqual(len(path) - 1, shortest_distance(start, goal))
                    self.assertEqual(len(path), len(set(path)))
                    for left, right in zip(path, path[1:]):
                        self.assertIn(right, NEIGHBORS[left])
        np.testing.assert_array_equal(warehouse.R, original)

    def test_values_across_seeds_match_exact_optimum(self):
        for seed in (0, 7, 42):
            for goal in NEIGHBORS:
                q = warehouse.train_q(goal, seed=seed)
                for start, neighbors in NEIGHBORS.items():
                    s = warehouse.location_to_state[start]
                    if start == goal:
                        np.testing.assert_array_equal(q[s], 0)
                        continue
                    for neighbor in neighbors:
                        a = warehouse.location_to_state[neighbor]
                        distance = shortest_distance(neighbor, goal)
                        expected = 4 + 6 * warehouse.gamma ** distance
                        self.assertAlmostEqual(q[s, a], expected, places=7)
                self.assertTrue(np.all(q[warehouse.R == 0] == 0))

    def test_reproducibility_and_failure_cases(self):
        np.testing.assert_array_equal(warehouse.train_q("G"), warehouse.train_q("G"))
        for bad in ("Z", "a", "", None, 3):
            with self.assertRaises(ValueError):
                warehouse.route(bad, "G")
            with self.assertRaises(ValueError):
                warehouse.route("E", bad)
        with self.assertRaises(RuntimeError):
            warehouse.train_q("G", max_updates=1)
        with self.assertRaises(ValueError):
            warehouse.train_q("G", max_updates=0)
        with self.assertRaises(RuntimeError):
            warehouse._greedy_route(4, 6, np.zeros_like(warehouse.R))


if __name__ == "__main__":
    unittest.main()
