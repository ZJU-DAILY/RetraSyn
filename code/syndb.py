import random

import numpy as np
from grid import Grid, GridMap
import utils
from typing import List, Tuple


class SynDB:
    def __init__(self):
        # All history data
        self.history_data: List[List[Tuple[Grid, int]]] = []
        # Trajectories that haven't terminated
        self.current_data: List[List[Tuple[Grid, int]]] = []
        # Current timestamp
        self.t = -1

    def generate_new_points(self,
                            markov_mat: np.ndarray,
                            grid_map: GridMap,
                            avg_len: float):
        self.t += 1
        for traj in self.current_data:
            prev_grid = traj[-1][0]
            candidates = grid_map.get_adjacent(prev_grid)
            # add self-transition
            candidates.append(prev_grid.index)
            candidate_prob = np.zeros(len(candidates) + 1)
            row = utils.grid_index_map_func(prev_grid, grid_map)

            for k, (i, j) in enumerate(candidates):
                col = utils.grid_index_map_func(grid_map.map[i][j], grid_map)
                prob = markov_mat[row][col]

                if np.isnan(prob):
                    candidate_prob[k] = 0
                else:
                    candidate_prob[k] = prob

            # Quit probability
            col = -1
            prob = markov_mat[row][col]
            prob *= min(1.0, len(traj) / avg_len)
            candidate_prob[-1] = prob

            if candidate_prob.sum() < 0.00001:
                traj.append((prev_grid, self.t))
            else:
                candidate_prob = candidate_prob / candidate_prob.sum()
                sample_id = np.random.choice(np.arange(len(candidate_prob)), p=candidate_prob)

                if sample_id == len(candidate_prob) - 1:
                    # Quitting
                    continue
                i, j = candidates[sample_id]
                traj.append((grid_map.map[i][j], self.t))

        # Move terminated trajectories to history data
        new_curr_data = []
        for traj in self.current_data:
            if traj[-1][1] == self.t:
                new_curr_data.append(traj)
            else:
                self.history_data.append(traj)
        self.current_data = new_curr_data

    def generate_new_points_baseline(self,
                                     markov_mat: np.ndarray,
                                     grid_map: GridMap):
        """
        For baseline, without considering quitting events
        """
        self.t += 1
        for traj in self.current_data:
            prev_grid = traj[-1][0]
            candidates = grid_map.get_adjacent(prev_grid)
            # add self-transition
            candidates.append(prev_grid.index)
            candidate_prob = np.zeros(len(candidates))
            row = utils.grid_index_map_func(prev_grid, grid_map)

            for k, (i, j) in enumerate(candidates):
                col = utils.grid_index_map_func(grid_map.map[i][j], grid_map)
                prob = markov_mat[row][col]

                if np.isnan(prob):
                    candidate_prob[k] = 0
                else:
                    candidate_prob[k] = prob

            if candidate_prob.sum() < 0.00001:
                sample_id = np.random.choice(np.arange(len(candidates)))
                i, j = candidates[sample_id]
                traj.append((grid_map.map[i][j], self.t))
            else:
                candidate_prob = candidate_prob / candidate_prob.sum()
                sample_id = np.random.choice(np.arange(len(candidate_prob)), p=candidate_prob)

                i, j = candidates[sample_id]
                traj.append((grid_map.map[i][j], self.t))

    def adjust_data_size(self,
                         markov_mat: np.ndarray,
                         target_n: int,
                         grid_map: GridMap,
                         quit_distribution: np.ndarray):
        while self.n < target_n:
            # Add new trajectories
            # Get entering distribution
            prob = markov_mat[-1] / markov_mat[-1].sum()
            sample_id = np.random.choice(np.arange(grid_map.size), p=prob[:-1])
            self.current_data.append([(utils.grid_index_inv_func(sample_id, grid_map), self.t)])

        if self.n > target_n:
            if np.sum(quit_distribution) < 1e-5:
                random.shuffle(self.current_data)
                sample_data = self.current_data[target_n:]

                for idx, traj in enumerate(sample_data):
                    sample_data[idx] = traj[:-1]
                self.history_data.extend(sample_data)
                self.current_data = self.current_data[:target_n]
            else:
                # Sampling based on quitting distribution
                prob = np.zeros(self.n)
                for i in range(self.n):
                    row = utils.grid_index_map_func(self.current_data[i][-2][0], grid_map)
                    prob[i] = quit_distribution[row]
                prob += 1e-8
                prob = prob / prob.sum()
                sample_id = np.random.choice(np.arange(self.n), size=self.n - target_n, replace=False, p=prob)
                non_sample_id = list(set(np.arange(self.n)) - set(sample_id))
                new_history_add = [self.current_data[i] for i in sample_id]

                for idx, traj in enumerate(new_history_add):
                    new_history_add[idx] = traj[:-1]
                self.history_data.extend(new_history_add)
                new_curr_data = [self.current_data[i] for i in non_sample_id]
                self.current_data = new_curr_data

    def random_initialize(self,
                          target_n: int,
                          grid_map: GridMap):
        self.t = 0
        while self.n < target_n:
            sample_id = np.random.choice(np.arange(grid_map.size))
            self.current_data.append([(utils.grid_index_inv_func(sample_id, grid_map), self.t)])

    @property
    def n(self):
        return len(self.current_data)

    @property
    def all_data(self):
        d = self.history_data.copy()
        d.extend(self.current_data)
        return d


class Users:
    """
    User status:
    1: active(available), 0: inactive(not recycled), 2: sampled for reporting, -1: quitted
    """

    def __init__(self):
        self.users = {}

    def register(self, uid):
        try:
            self.users[uid]
        except KeyError:
            self.users[uid] = 1

    def sample(self, p):
        available_users = self.available_users
        sampled_users = random.sample(available_users, int(p * len(available_users)))
        for uid in sampled_users:
            self.users[uid] = 2
        return sampled_users

    def deactivate(self, uid):
        self.users[uid] = 0

    def remove(self, uid):
        self.users[uid] = -1

    def recycle(self, uid):
        if self.users[uid] != -1:
            self.users[uid] = 1

    @property
    def available_users(self):
        a_u = []
        for (uid, state) in self.users.items():
            if state == 1:
                a_u.append(uid)
        return a_u
