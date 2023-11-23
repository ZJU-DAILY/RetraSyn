import pickle

import utils
from grid import Grid, GridMap
from parse import args
import multiprocessing
import random
import numpy as np
from typing import List, Tuple
import json
import experiment
from logger.logger import ConfigParser
import lzma

config = ConfigParser(name='evaluation', save_dir='./')
logger = config.get_logger(config.exper_name)
CORES = multiprocessing.cpu_count() // 2
random.seed(2023)
np.random.seed(2023)


def spatial_decomposition(db: List[List[Tuple[float, float, int]]], gm: GridMap, multi=False):
    if multi:
        def decomp_multi(xy_l: List[Tuple[float, float, int]]):
            return utils.xyt2grid(xy_l, gm)

        pool = multiprocessing.Pool(CORES)
        grid_db = pool.map(decomp_multi, db)
        pool.close()
    else:
        grid_db = [utils.xyt2grid(traj, gm) for traj in db]

    return grid_db


def split_traj_db(grid_db: List[List[Tuple[Grid, int]]], gm: GridMap):
    def split_traj(grid_t: List[Tuple[Grid, int]]):
        new_trajs = []
        split_id = []
        for i in range(len(grid_t) - 1):
            curr_grid = grid_t[i][0]
            next_grid = grid_t[i + 1][0]
            if not (curr_grid.equal(next_grid) or gm.is_adjacent_grids(curr_grid, next_grid)):
                split_id.append(i + 1)
        if not len(split_id):
            return [grid_t]

        start_id = 0
        for sid in split_id:
            new_trajs.append(grid_t[start_id:sid])
            start_id = sid
        new_trajs.append(grid_t[sid:])
        return new_trajs

    new_grid_db = []
    for traj in grid_db:
        new_grid_db.extend(split_traj(traj))

    return new_grid_db


logger.info(args)
orig_file = f'../data/{args.dataset}.xz'
syn_file = f'../data/syn_data/{args.dataset}/{args.method}_{args.epsilon}_g{args.grid_num}_w{args.w}.pkl'

orig_db: List[List[Tuple[float, float, int]]]
syn_db: List[List[Tuple[float, float, int]]]

with lzma.open(orig_file, 'rb') as f:
    orig_db = pickle.load(f)
with open(syn_file, 'rb') as f:
    syn_db = pickle.load(f)

with open(f'../data/{args.dataset}_stats.json', 'r') as f:
    stats = json.load(f)

grid_map = GridMap(args.grid_num,
                   stats['min_x'],
                   stats['min_y'],
                   stats['max_x'],
                   stats['max_y'])
logger.info('Spatial decomposition...')
if args.multiprocessing:
    def decomp_multi(xy_l: List[Tuple[float, float, int]]):
        return utils.xyt2grid(xy_l, grid_map)


    if args.dataset == 'sanjoaquin':
        # dataset is too large, use smaller cores to avoid memory error
        CORES = 5

    pool = multiprocessing.Pool(CORES)
    orig_grid_db = pool.map(decomp_multi, orig_db)
    pool.close()
    pool = multiprocessing.Pool(CORES)
    syn_grid_db = pool.map(decomp_multi, syn_db)
    pool.close()
else:
    orig_grid_db = [utils.xyt2grid(traj, grid_map) for traj in orig_db]
    syn_grid_db = [utils.xyt2grid(traj, grid_map) for traj in syn_db]

orig_grid_db = split_traj_db(orig_grid_db, grid_map)

if args.dataset == 'oldenburg':
    max_time = 500
    # average user per timestamp
    upt = 34000
elif args.dataset == 'tdrive':
    max_time = 886
    upt = 3821
elif args.dataset == 'sanjoaquin':
    max_time = 1000
    upt = 56749

logger.info('Experiment: Density')
orig_counts = experiment.get_grid_count(orig_grid_db, grid_map.get_list_map(), max_time=max_time)
syn_counts = experiment.get_grid_count(syn_grid_db, grid_map.get_list_map(), max_time=max_time)

orig_density = [counts / (counts.sum() + 1e-10) for counts in orig_counts]
syn_density = [counts / (counts.sum() + 1e-10) for counts in syn_counts]
density_results = experiment.eval_jsd(orig_density, syn_density)
logger.info(f'Density Error={density_results}')

logger.info('Experiment: Transition')
orig_trans = experiment.get_transition_count(orig_grid_db, grid_map.get_normal_transition(), max_time=max_time)
syn_trans = experiment.get_transition_count(syn_grid_db, grid_map.get_normal_transition(), max_time=max_time)
orig_distribution = [counts / (counts.sum() + 1e-10) for counts in orig_trans]
syn_distribution = [counts / (counts.sum() + 1e-10) for counts in syn_trans]
transition_results = experiment.eval_jsd(orig_distribution, syn_distribution)
logger.info(f'Transition Error={transition_results}')

logger.info('Experiment: Spatial-Temporal Query Error...')
st_queries = [experiment.SquareQuery(grid_map.min_x, grid_map.min_y, grid_map.max_x, grid_map.max_y, max_time,
                                         time_range=args.phi) for _ in
                  range(100)]
if args.multiprocessing:

    average_total_points = upt * (st_queries[0].max_t - st_queries[0].min_t + 1)


    def orig_query_multi(query):
        return query.point_query_t(orig_db)


    def syn_query_multi(query):
        return query.point_query_t(syn_db)


    if args.dataset == 'sanjoaquin':
        CORES = 3

    pool = multiprocessing.Pool(CORES)
    actual_ans = pool.map(orig_query_multi, st_queries)
    pool.close()

    pool = multiprocessing.Pool(CORES)
    syn_ans = pool.map(syn_query_multi, st_queries)
    pool.close()

    actual_ans = np.asarray(actual_ans)
    syn_ans = np.asarray(syn_ans)
    numerator = np.abs(actual_ans - syn_ans)
    denominator = np.asarray([max(actual_ans[i], average_total_points * 0.01) for i in range(len(actual_ans))])

    st_query_error = numerator / denominator
else:
    st_query_error = experiment.eval_st_query_error(orig_db, syn_db, queries=st_queries, upt=upt)


logger.info(f'Spatial-Temporal Query Error: {np.mean(st_query_error)}')

random.seed(2023)
np.random.seed(2023)
logger.info('Experiment: Pattern Errors')
min_times = [random.randint(0, max_time - args.phi) for _ in range(100)]
max_times = [m_t + args.phi - 1 for m_t in min_times]
if args.multiprocessing:
    pattern_queries = [(min_times[i], max_times[i]) for i in range(len(min_times))]


    def pattern_error_multi(query: Tuple[int, int]):
        orig_pattern = experiment.mine_patterns(orig_grid_db, query[0], query[1])
        syn_pattern = experiment.mine_patterns(syn_grid_db, query[0], query[1])
        return experiment.calculate_pattern_f1(orig_pattern, syn_pattern)

    pool = multiprocessing.Pool(CORES)
    pattern_errors = pool.map(pattern_error_multi, pattern_queries)
    pool.close()
else:
    pattern_errors = []
    for i in range(len(min_times)):
        orig_pattern = experiment.mine_patterns(orig_grid_db, min_times[i], max_times[i])
        syn_pattern = experiment.mine_patterns(syn_grid_db, min_times[i], max_times[i])

        pattern_errors.append(experiment.calculate_pattern_f1(orig_pattern, syn_pattern))
logger.info(f'Pattern F1 Error: {np.mean(pattern_errors)}')

logger.info('Experiment: Kendall-tau Coefficient')
k_t = experiment.calculate_coverage_kendall_tau(orig_grid_db, syn_grid_db, grid_map)
logger.info(f'Kendall-tau Coefficient : {k_t}')

logger.info('Experiment: Length Error...')
length_err = experiment.calculate_length_error(orig_db, syn_db)
logger.info(f'Length Error : {length_err}')

logger.info('Experiment: Trip Error...')
trip_err = experiment.calculate_trip_error(orig_grid_db, syn_grid_db, grid_map)
logger.info(f'Trip Error : {trip_err}')

random.seed(2023)
np.random.seed(2023)
logger.info('Experiment: Hotspot NDCG')
min_times = [random.randint(0, max_time - args.phi) for _ in range(100)]
max_times = [m_t + args.phi - 1 for m_t in min_times]
if args.multiprocessing:
    hotspot_queries = [(min_times[i], max_times[i]) for i in range(len(min_times))]


    def hotspot_multi(query: Tuple[int, int]):
        orig_total_counts = np.zeros_like(orig_counts[query[0]])
        syn_total_counts = np.zeros_like(syn_counts[query[0]])
        orig_total_counts += orig_counts[query[0]]
        syn_total_counts += syn_counts[query[0]]
        for i in range(query[0] + 1, query[1] + 1):
            orig_total_counts += orig_counts[i]
            syn_total_counts += syn_counts[i]
        return experiment.eval_hotspot_ndcg(orig_total_counts, syn_total_counts)


    pool = multiprocessing.Pool(CORES)
    hotspot_errors = pool.map(hotspot_multi, hotspot_queries)
    pool.close()
else:
    hotspot_errors = []
    for i in range(len(min_times)):
        orig_total_counts = np.zeros_like(orig_counts[min_times[i]])
        syn_total_counts = np.zeros_like(syn_counts[min_times[i]])
        orig_total_counts += orig_counts[min_times[i]]
        syn_total_counts += syn_counts[min_times[i]]
        for j in range(min_times[i] + 1, max_times[i] + 1):
            orig_total_counts += orig_counts[j]
            syn_total_counts += syn_counts[j]
        hotspot_errors.append(experiment.eval_hotspot_ndcg(orig_total_counts, syn_total_counts))
logger.info(f'Hotspot NDCG : {np.mean(hotspot_errors)}')
