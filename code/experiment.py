import numpy as np
from grid import Grid, GridMap, Transition
from typing import List, Tuple
import utils
import random
import math
import multiprocessing

CORES = multiprocessing.cpu_count() // 2


class Query:
    def __init__(self):
        pass

    def point_query(self, db):
        raise NotImplementedError

    def point_query_t(self, db):
        print('Temporal query is not supported!')


class SquareQuery(Query):
    def __init__(self,
                 min_x: float,
                 min_y: float,
                 max_x: float,
                 max_y: float,
                 max_time: int,
                 time_range=5,
                 size_factor=9.0):
        super().__init__()
        self.edge = math.sqrt((max_x - min_x) * (max_y - min_y) / size_factor)
        # Randomly select center
        center_x = random.random() * (max_x - min_x - self.edge) + min_x + self.edge / 2
        center_y = random.random() * (max_y - min_y - self.edge) + min_y + self.edge / 2
        self.center = (center_x, center_y)

        self.left_x = center_x - self.edge / 2
        self.up_y = center_y + self.edge / 2
        self.right_x = center_x + self.edge / 2
        self.down_y = center_y - self.edge / 2

        self.min_t = random.randint(0, max_time - time_range)
        self.max_t = self.min_t + time_range - 1

    def in_square(self, point: Tuple[float, float]):
        return self.left_x <= point[0] <= self.right_x and self.down_y <= point[1] <= self.up_y

    def in_square_t(self, point: Tuple[float, float, int]):
        return self.min_t <= point[2] <= self.max_t and self.in_square((point[0], point[1]))

    def point_query(self, db: List[List[Tuple[float, float]]]):
        count = 0
        for t in db:
            for p in t:
                if self.in_square(p):
                    count += 1

        return count

    def point_query_t(self, db: List[List[Tuple[float, float, int]]]):
        count = 0
        for t in db:
            for p in t:
                if self.in_square_t(p):
                    count += 1
        return count


class Pattern:
    def __init__(self, grids: List[Grid]):
        self.grids = grids

    @property
    def size(self):
        return len(self.grids)

    def __eq__(self, other):
        if other is None:
            return False
        if not type(other) == Pattern:
            return False
        if not other.size == self.size:
            return False

        for i in range(self.size):
            if not self.grids[i].index == other.grids[i].index:
                return False

        return True

    def __hash__(self):
        prime = 31
        result = 1
        for g in self.grids:
            result = result * prime + g.__hash__()

        return result

    def check_pattern(self):
        if self.size <= 2:
            return True
        for i in range(self.size - 1):
            if self.grids[i].equal(self.grids[i + 1]):
                return False
        return True


def eval_st_query_error(orig_db, syn_db, queries: List[SquareQuery], sanity_bound=0.01, upt=34000):
    actual_ans = list()
    syn_ans = list()

    average_total_points = upt * (queries[0].max_t - queries[0].min_t + 1)

    for q in queries:
        actual_ans.append(q.point_query_t(orig_db))
        syn_ans.append(q.point_query_t(syn_db))

    actual_ans = np.asarray(actual_ans)
    syn_ans = np.asarray(syn_ans)
    numerator = np.abs(actual_ans - syn_ans)
    # use sanity bound to mitigate the effect of extremely small actual_ans
    denominator = np.asarray([max(actual_ans[i], average_total_points * sanity_bound) for i in range(len(actual_ans))])

    error = numerator / denominator

    return np.mean(error)


def eval_jsd(true, release):
    results = []
    for i in range(len(true)):
        results.append(utils.js_divergence(true[i], release[i]))
    return np.mean(results)


def mine_patterns(db: List[List[Tuple[Grid, int]]], min_time, max_time, min_size=2, max_size=5):
    pattern_dict = {}
    for curr_size in range(min_size, max_size + 1):
        if curr_size > max_time - min_time:
            break
        for traj in db:
            for i in range(0, len(traj) - curr_size + 1):
                if traj[i][1] < min_time or traj[i + curr_size - 1][1] > max_time:
                    continue
                p = Pattern([g_t[0] for g_t in traj[i:i + curr_size]])
                if not p.check_pattern():
                    continue
                try:
                    pattern_dict[p] += 1
                except KeyError:
                    pattern_dict[p] = 1

    return pattern_dict


def calculate_pattern_f1(orig_pattern,
                         syn_pattern,
                         k=100):
    sorted_orig = sorted(orig_pattern.items(), key=lambda x: x[1], reverse=True)
    sorted_syn = sorted(syn_pattern.items(), key=lambda x: x[1], reverse=True)

    orig_top_k = [x[0] for x in sorted_orig][:k]
    syn_top_k = [x[0] for x in sorted_syn][:k]

    count = 0
    for p1 in syn_top_k:
        if p1 in orig_top_k:
            count += 1

    precision = count / k
    recall = count / k

    return 2 * precision * recall / (precision + recall) if precision else 0


def get_grid_count(grid_db: List[List[Tuple[Grid, int]]], domain: List[Grid], max_time, min_time=0):
    """
    Return a list of grid counts for each timestamp
    """
    domain_map = utils.list_to_dict(domain)
    grid_counts = [np.zeros(len(domain)) for _ in range(max_time)]
    for traj in grid_db:
        for (g, t) in traj:
            if t < min_time or t >= max_time:
                continue
            grid_counts[t][domain_map[g]] += 1

    return grid_counts


def get_transition_count(grid_db: List[List[Tuple[Grid, int]]], domain: List[Transition], max_time, min_time=0):
    domain_map = utils.list_to_dict(domain)
    trans_counts = [np.zeros(len(domain)) for _ in range(max_time - 1)]
    for traj in grid_db:
        for i in range(len(traj) - 1):
            t = traj[i][1]
            if t < min_time or t >= max_time - 1:
                continue
            curr_grid = traj[i][0]
            next_grid = traj[i + 1][0]
            trans = Transition(curr_grid, next_grid)
            trans_counts[t][domain_map[trans]] += 1

    return trans_counts


def eval_hotspot_ndcg(orig_counts, syn_counts, k=10):
    orig_density = orig_counts / orig_counts.sum()
    syn_density = syn_counts / syn_counts.sum()
    sorted_orig = sorted(enumerate(orig_density), key=lambda x: x[1], reverse=True)
    sorted_syn = sorted(enumerate(syn_density), key=lambda x: x[1], reverse=True)

    orig_top_k = [x[0] for x in sorted_orig][:k]
    syn_top_k = [x[0] for x in sorted_syn][:k]

    r = np.zeros(k)

    for i, p1 in enumerate(syn_top_k):
        if p1 in orig_top_k:
            r[i] = 1 / (orig_top_k.index(p1) + 1)

    idcg = np.sum((np.ones(k) / np.arange(1, k + 1)) * 1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(r * 1. / np.log2(np.arange(2, k + 2)))

    return dcg / idcg if idcg else 0


def calculate_coverage_kendall_tau(orig_db: List[List[Tuple[Grid, int]]],
                                   syn_db: List[List[Tuple[Grid, int]]],
                                   grid_map: GridMap):
    actual_counts = np.zeros(grid_map.size)
    syn_counts = np.zeros(grid_map.size)

    # For each grid, find how many trajectories pass through it
    for i in range(len(grid_map.map)):
        for j in range(len(grid_map.map[0])):
            g = grid_map.map[i][j]
            index = utils.grid_index_map_func(g, grid_map)
            for t in orig_db:
                actual_counts[index] += utils.pass_through(t, g)
            for t in syn_db:
                syn_counts[index] += utils.pass_through(t, g)

    concordant_pairs = 0
    reversed_pairs = 0
    for i in range(grid_map.size):
        for j in range(i + 1, grid_map.size):
            if actual_counts[i] > actual_counts[j]:
                if syn_counts[i] > syn_counts[j]:
                    concordant_pairs += 1
                else:
                    reversed_pairs += 1
            if actual_counts[i] < actual_counts[j]:
                if syn_counts[i] < syn_counts[j]:
                    concordant_pairs += 1
                else:
                    reversed_pairs += 1

    denominator = grid_map.size * (grid_map.size - 1) / 2
    return (concordant_pairs - reversed_pairs) / denominator


def calculate_length_error(orig_db: List[List[Tuple[float, float, int]]],
                           syn_db: List[List[Tuple[float, float, int]]],
                           bucket_num=20):
    orig_length = [utils.get_travel_distance(t) for t in orig_db]
    syn_length = [utils.get_travel_distance(t) for t in syn_db]

    bucket_size = (max(max(orig_length), max(syn_length)) - min(min(orig_length), min(syn_length))) / bucket_num

    orig_count = np.zeros(bucket_num)
    syn_count = np.zeros(bucket_num)
    for i in range(bucket_num):
        start = i * bucket_size
        end = start + bucket_size

        for d in orig_length:
            if start <= d <= end:
                orig_count[i] += 1
        for d in syn_length:
            if start <= d <= end:
                syn_count[i] += 1

    # Normalization
    orig_count /= np.sum(orig_count)
    syn_count /= np.sum(syn_count)

    return utils.js_divergence(orig_count, syn_count)


def get_trip_distribution(grid_db: List[List[Tuple[Grid, int]]], grid_map: GridMap):
    dist = np.zeros(grid_map.size * grid_map.size)

    for g_t in grid_db:
        start = g_t[0][0]
        end = g_t[-1][0]

        index = utils.pair_grid_index_map_func((start, end), grid_map)
        dist[index] += 1

    return dist


def calculate_trip_error(orig_db: List[List[Tuple[Grid, int]]],
                         syn_db: List[List[Tuple[Grid, int]]],
                         grid_map: GridMap):
    orig_trip = get_trip_distribution(orig_db, grid_map)
    syn_trip = get_trip_distribution(syn_db, grid_map)

    orig_trip /= orig_trip.sum()
    syn_trip /= syn_trip.sum()

    return utils.js_divergence(orig_trip, syn_trip)
