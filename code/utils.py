from typing import List, Tuple
from grid import Grid, GridMap
import numpy as np
import json
import math


def list_to_dict(l: List):
    d = {}
    for (index, val) in enumerate(l):
        d[val] = index
    return d


def t_dataset_stats(dataset: List[List[Tuple[float, float, float, float, int]]], stats_name: str):
    """
    Used in budget-division strategy
    Get statistics of the transition-formed dataset, the name of the data file
    should be '{dataset_name}_transition.pkl'
    dataset: [[(x0, y0, x1, y1, flag), ...], ...]
    """
    xs, ys = [], []
    for t_l in dataset:
        for (x0, y0, x1, y1, flag) in t_l:
            if flag:
                continue
            xs.extend([x0, x1])
            ys.extend([y0, y1])
    stats = {'min_x': min(xs), 'min_y': min(ys), 'max_x': max(xs), 'max_y': max(ys)}
    with open(stats_name, 'w') as f:
        json.dump(stats, f)
    return stats


def tid_dataset_stats(dataset: List[List[Tuple[float, float, float, float, int, int]]], stats_name: str):
    """
    Used in population-division strategy
    Get statistics of the transition-formed dataset, the name of the data file
    should be '{dataset_name}_transition_id.pkl'
    dataset: [[(x0, y0, x1, y1, flag, uid), ...], ...]
    """
    xs, ys = [], []
    for t_l in dataset:
        for (x0, y0, x1, y1, flag, uid) in t_l:
            if flag:
                continue
            xs.extend([x0, x1])
            ys.extend([y0, y1])
    stats = {'min_x': min(xs), 'min_y': min(ys), 'max_x': max(xs), 'max_y': max(ys)}
    with open(stats_name, 'w') as f:
        json.dump(stats, f)
    return stats


def xy2grid(xy_list: List[Tuple[float, float]], grid_map: GridMap):
    grid_list = []
    for pos in xy_list:
        found = False
        for i in range(len(grid_map.map)):
            for j in range(len(grid_map.map[i])):
                if grid_map.map[i][j].in_cell(pos):
                    grid_list.append(grid_map.map[i][j])
                    found = True
                    break
            if found:
                break

    return grid_list


def xyt2grid(xy_list: List[Tuple[float, float, int]], grid_map: GridMap):
    grid_list = []
    for (x, y, t) in xy_list:
        found = False
        for i in range(len(grid_map.map)):
            for j in range(len(grid_map.map[i])):
                if grid_map.map[i][j].in_cell((x, y)):
                    grid_list.append((grid_map.map[i][j], t))
                    found = True
                    break
            if found:
                break

    return grid_list


def grid_index_map_func(g: Grid, grid_map: GridMap):
    """
    Map a grid to its index: (i, j) => int
    return: i*|column|+j
    """
    i, j = g.index
    return i * len(grid_map.map[0]) + j


def grid_index_inv_func(index: int, grid_map: GridMap):
    """
    Inverse function of grid_index_map_func
    """
    i = index // len(grid_map.map[0])
    j = index % len(grid_map.map[0])
    return grid_map.map[i][j]


def pair_grid_index_map_func(grid_pair: Tuple[Grid, Grid], grid_map: GridMap):
    """
    Map a pair of grid to index: (g1, g2) => (i1, i2) => int
    Firstly map (g1, g2) to a matrix of [N x N], where N is
    the total number of grids
    return: i1 * N + i2
    """
    g1, g2 = grid_pair
    index1 = grid_index_map_func(g1, grid_map)
    index2 = grid_index_map_func(g2, grid_map)

    return index1 * grid_map.size + index2


def kl_divergence(prob1, prob2):
    prob1 = np.asarray(prob1)
    prob2 = np.asarray(prob2)

    kl = np.log((prob1 + 1e-8) / (prob2 + 1e-8)) * prob1

    return np.sum(kl)


def js_divergence(prob1, prob2):
    prob1 = np.asarray(prob1)
    prob2 = np.asarray(prob2)

    avg_prob = (prob1 + prob2) / 2

    return 0.5 * kl_divergence(prob1, avg_prob) + 0.5 * kl_divergence(prob2, avg_prob)


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def allocation_p(e, w, alpha=1):
    return alpha / w * math.log(e + 1)


def get_travel_distance(t: List[Tuple[float, float, int]]):
    dist = 0
    for i in range(len(t) - 1):
        curr_p = (t[i][0], t[i][1])
        next_p = (t[i + 1][0], t[i + 1][1])
        dist += euclidean_distance(curr_p, next_p)

    return dist


def get_diameter(t: List[Tuple[float, float,int]]):
    max_d = 0
    for i in range(len(t)):
        for j in range(i+1, len(t)):
            max_d = max(max_d, euclidean_distance((t[i][0],t[i][1]), (t[j][0],t[j][1])))

    return max_d


def pass_through(t: List[Tuple[Grid, int]], g: Grid):
    for t_g in t:
        if t_g[0].index == g.index:
            return True

    return False


