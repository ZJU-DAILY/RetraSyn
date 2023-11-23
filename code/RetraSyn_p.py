import pickle

from ldp import OUE
from grid import Grid, GridMap, Transition
from typing import List,Tuple
import utils
import numpy as np
import math
from parse import args
import multiprocessing
import random
from syndb import SynDB, Users
from logger.logger import ConfigParser
import lzma

config = ConfigParser(name='RetraSyn', save_dir='./')
logger = config.get_logger(config.exper_name)

CORES = multiprocessing.cpu_count() // 2
random.seed(2023)
np.random.seed(2023)

logger.info(args)


def spatial_decomposition(xy_l: List[Tuple[float, float, float, float, int, int]], gm: GridMap):
    grid_list = []
    for (x0, y0, x1, y1, flag, uid) in xy_l:
        if flag == 0:
            g0, g1 = utils.xy2grid([(x0, y0), (x1, y1)], gm)
            grid_list.append((g0, g1, flag, uid))
        elif flag == 1:
            g1 = utils.xy2grid([(x1, y1)], gm)[0]
            grid_list.append((g1, g1, flag, uid))
        else:
            g0 = utils.xy2grid([(x0, y0)], gm)[0]
            grid_list.append((g0, g0, flag, uid))
    return grid_list


def split_traj(traj_stream: List[List[Tuple[Grid, Grid, int, int]]], gm: GridMap):
    """
    Deal with non-adjacent transitions;
    If (G1, G2, flag) is not adjacent, split it into (G1, end, 2) at t and (start, G2, 1) at t + 1
    """
    new_stream = []
    while len(new_stream) <= len(traj_stream):
        new_stream.append([])
    for t in range(len(traj_stream)):
        for g1, g2, flag, uid in traj_stream[t]:
            if flag:
                new_stream[t].append((g1, g2, flag, uid))
                continue
            if not g1.equal(g2) and not gm.is_adjacent_grids(g1, g2):
                new_stream[t].append((g1, g1, 2, uid))
                new_stream[t + 1].append((g2, g2, 1, uid))
            else:
                new_stream[t].append((g1, g2, flag, uid))
    return new_stream


def generate_markov_matrix(markov_vec: np.ndarray, trans_domain: List[Transition]):
    n = grid_map.size + 1  # with virtual start & end point
    markov_mat = np.zeros((n, n), dtype=float)
    end_distribution = np.zeros(n - 1)
    for k in range(len(markov_vec)):
        if markov_vec[k] <= 0:
            continue

        # find index in matrix
        trans = trans_domain[k]
        if not trans.flag:
            i = utils.grid_index_map_func(trans.g1, grid_map)
            j = utils.grid_index_map_func(trans.g2, grid_map)
        elif trans.flag == 1:
            # start transition, located in last row of the matrix
            i = -1
            j = utils.grid_index_map_func(trans.g2, grid_map)
        else:
            # end transition, located in last column of the matrix
            i = utils.grid_index_map_func(trans.g1, grid_map)
            j = -1
            end_distribution[i] = markov_vec[k]
        markov_mat[i][j] = markov_vec[k]

    # Normalize probabilities by each ROW
    markov_mat = markov_mat / (markov_mat.sum(axis=1).reshape((-1, 1)) + 1e-8)
    end_distribution = end_distribution / (end_distribution.sum() + 1e-8)
    return markov_mat, end_distribution


def convert_grid_to_raw(grid_db: List[List[Tuple[Grid, int]]]):
    def traj_grid_to_raw(traj: List[Tuple[Grid, int]]):
        xy_traj = []
        for (g, t) in traj:
            x, y = g.sample_point()
            xy_traj.append((x, y, t))
        return xy_traj

    raw_db = [traj_grid_to_raw(traj) for traj in grid_db]

    return raw_db


def RetraSyn(traj_stream, w: int, eps, trans_domain: List[Transition]):
    trans_domain_map = utils.list_to_dict(trans_domain)

    synthetic_db = SynDB()
    trans_distribution = []
    used_budget = []
    release = []
    N_sp = []
    users = Users()
    quitted_users = []

    for t in range(2):
        # warm-up stage
        oue = OUE(eps, len(trans_domain), lambda x: trans_domain_map[x])

        # remove quitted users
        for uid in quitted_users:
            users.remove(uid)
        quitted_users = []

        # add new users
        for (_, _, flag, uid) in traj_stream[t]:
            users.register(uid)
            if flag == 2:
                quitted_users.append(uid)

        sampled_users = users.sample(1 / w)

        # update transition distribution
        for (g1, g2, flag, uid) in traj_stream[t]:
            if not users.users[uid] == 2:
                # user not sampled
                continue
            trans = Transition(g1, g2, flag)
            oue.privatise(trans)
        oue.adjust()

        est_counts = oue.non_negative_data / oue.n

        # generate Markov matrix
        markov_mat, end_distribution = generate_markov_matrix(est_counts,
                                                              trans_domain)
        trans_distribution.append(markov_mat)
        release.append(est_counts / est_counts.sum())

        # generate new points in synthetic data based on current distribution
        synthetic_db.generate_new_points(markov_mat, grid_map, avg_lens[args.dataset])

        # adjust size of synthetic database
        synthetic_db.adjust_data_size(markov_mat, len(traj_stream[t]), grid_map, end_distribution)

        used_budget.append(sampled_users)
        for uid in sampled_users:
            users.deactivate(uid)
        N_sp.append(0)

    for t in range(2, len(traj_stream)):
        if not len(traj_stream[t]):
            continue

        # user recycling
        if t >= w:
            for uid in used_budget[t - w]:
                users.recycle(uid)

        # remove quited users
        for uid in quitted_users:
            users.remove(uid)
        quitted_users = []

        for (g1, g2, flag, uid) in traj_stream[t]:
            users.register(uid)
            if flag == 2:
                quitted_users.append(uid)

        dev = np.abs(np.array(release[-1]) - np.average(release[max(0, t - 5):t], axis=0)).sum()

        cr = max(0.5, 1 - np.average(N_sp[max(0, t - 5):t]) / len(trans_domain))
        p = utils.allocation_p(dev, w, alpha=8)
        p = min(p * cr, 0.6)
        sampled_users = users.sample(p)

        oue = OUE(eps, len(trans_domain), lambda x: trans_domain_map[x])

        for (g1, g2, flag, uid) in traj_stream[t]:
            if not users.users[uid] == 2:
                continue
            trans = Transition(g1, g2, flag)
            oue.privatise(trans)
        oue.adjust()

        if oue.n == 0:
            used_budget.append([])

            N_sp.append(0)
            counts = release[-1]
        else:
            f_hat = oue.non_negative_data / oue.n
            f_tilde = release[-1]

            # select significant patterns
            variance = 4 * math.exp(eps) / (oue.n * (math.exp(eps) - 1) ** 2)
            select = (f_tilde - f_hat) ** 2 > variance

            # merge significant patterns and other patterns
            counts = np.zeros(len(trans_domain))
            sig_counts = oue.non_negative_data / oue.n

            for i in range(len(select)):
                if select[i]:
                    counts[i] = sig_counts[i]
                else:
                    counts[i] = f_tilde[i] * sig_counts.sum()

            for uid in sampled_users:
                users.deactivate(uid)

            used_budget.append(sampled_users)
            N_sp.append(np.sum(select))
        # generate Markov matrix
        markov_mat, end_distribution = generate_markov_matrix(counts, trans_domain)

        # generate new points in synthetic data based on current distribution
        synthetic_db.generate_new_points(markov_mat, grid_map, avg_lens[args.dataset])

        # check entering distribution
        if markov_mat[-1].sum() == 0:
            for i in range(t):
                if not trans_distribution[t - i - 1][-1].sum() == 0:
                    markov_mat[-1] = trans_distribution[t - i - 1][-1]
                    break

        # adjust size of synthetic database
        synthetic_db.adjust_data_size(markov_mat, len(traj_stream[t]), grid_map, end_distribution)

        trans_distribution.append(markov_mat)
        release.append(counts / counts.sum())

        if (t + 1) % 100 == 0:
            logger.info(f'{t + 1} timestamps processed')

    return synthetic_db


def lpd(traj_stream, w: int, eps: float, trans_domain: List[Transition]):
    trans_domain_map = utils.list_to_dict(trans_domain)
    release = []
    used_budget_1 = []
    used_budget_2 = []

    synthetic_db = SynDB()
    trans_distribution = []
    users = Users()

    # randomly initialize synthetic database
    synthetic_db.random_initialize(len(traj_stream[0]), grid_map)
    # first timestamp
    # add new users
    for (g1, g2, flag, uid) in traj_stream[1]:
        users.register(uid)

    sampled_users = users.sample(1/4)
    oue = OUE(eps, len(trans_domain), lambda x: trans_domain_map[x])

    # update transition distribution
    for (g1, g2, flag, uid) in traj_stream[1]:
        if not users.users[uid] == 2:
            # user not sampled
            continue
        trans = Transition(g1, g2, flag)
        oue.privatise(trans)
    oue.adjust()
    est_counts = oue.non_negative_data / oue.n
    release.append(est_counts / est_counts.sum())

    # generate Markov matrix
    markov_mat, end_distribution = generate_markov_matrix(est_counts,
                                                          trans_domain)
    trans_distribution.append(markov_mat)

    # generate new points in synthetic data based on current distribution
    synthetic_db.generate_new_points_baseline(markov_mat, grid_map)

    used_budget_2.append(sampled_users)
    for uid in sampled_users:
        users.deactivate(uid)
    used_budget_1.append([])
    quitted_users = []

    for t in range(2, len(traj_stream)-1):
        if not len(traj_stream[t]):
            continue
        # user recycling

        if t >= w:
            for uid in used_budget_1[t - w]:
                users.recycle(uid)
            for uid in used_budget_2[t-w]:
                users.recycle(uid)

        # remove quited users
        for uid in quitted_users:
            users.remove(uid)
        quitted_users = []

        for (g1, g2, flag, uid) in traj_stream[t]:
            users.register(uid)
            if flag == 2:
                quitted_users.append(uid)

        # set dissimilarity budget
        sampled_users = users.sample(1/(2*w))

        if len(sampled_users) == 0:
            release.append(release[-1])
            used_budget_2.append([])
            markov_mat, end_distribution = generate_markov_matrix(release[-1], trans_domain)
            synthetic_db.generate_new_points_baseline(markov_mat, grid_map)

            trans_distribution.append(markov_mat)
            continue

        # estimate c_t
        oue = OUE(eps, len(trans_domain), lambda x: trans_domain_map[x])

        for (g1, g2, flag, uid) in traj_stream[t]:
            if not users.users[uid] == 2:
                continue
            trans = Transition(g1, g2, flag)
            oue.privatise(trans)
        oue.adjust()

        for uid in sampled_users:
            users.deactivate(uid)
        used_budget_1.append(sampled_users)

        c_bar = oue.non_negative_data / oue.n

        # calculate dissimilarity
        dis = np.mean((c_bar - release[-1]) ** 2)
        dis -= 4 * math.exp(eps) / (oue.n * (math.exp(eps) - 1) ** 2)

        users_rm = len(users.available_users)//2
        err = 4 * math.exp(eps) / (int(1/2*users_rm) * (math.exp(eps) - 1) ** 2)

        if dis > err and users_rm > 0:
            sampled_users = users.sample(users_rm/2/len(users.available_users))
            # perturbation
            oue = OUE(eps, len(trans_domain), lambda x: trans_domain_map[x])
            used_users = []
            for (g1, g2, flag, uid) in traj_stream[t]:
                if not users.users[uid] == 2:
                    continue
                used_users.append(uid)
                trans = Transition(g1, g2, flag)
                oue.privatise(trans)
            oue.adjust()
            est_counts = oue.non_negative_data / oue.n
            release.append(est_counts / est_counts.sum())
            used_budget_2.append(sampled_users)
            for uid in sampled_users:
                users.deactivate(uid)
        else:
            # approximation
            release.append(release[-1])
            used_budget_2.append([])

        markov_mat, end_distribution = generate_markov_matrix(release[-1], trans_domain)
        synthetic_db.generate_new_points_baseline(markov_mat, grid_map)

        trans_distribution.append(markov_mat)

        if (t + 1) % 100 == 0:
            logger.info(f'{t + 1} timestamps processed')
    return synthetic_db


def lpa(traj_stream, w: int, eps: float, trans_domain: List[Transition]):
    trans_domain_map = utils.list_to_dict(trans_domain)
    l: int = 0
    eps_l2 = 0

    release = []
    used_budget_1 = []
    used_budget_2 = []

    synthetic_db = SynDB()
    trans_distribution = []
    users = Users()

    # randomly initialize synthetic database
    synthetic_db.random_initialize(len(traj_stream[0]), grid_map)
    # first timestamp
    # add new users
    for (g1, g2, flag, uid) in traj_stream[1]:
        users.register(uid)
    # first timestamp
    sampled_users = users.sample(1/(2*w))
    oue = OUE(eps, len(trans_domain), lambda x: trans_domain_map[x])

    for (g1, g2, flag, uid) in traj_stream[1]:
        if not users.users[uid] == 2:
            # user not sampled
            continue
        trans = Transition(g1, g2, flag)
        oue.privatise(trans)
    oue.adjust()
    est_counts = oue.non_negative_data / oue.n
    release.append(est_counts/est_counts.sum())

    # generate Markov matrix
    markov_mat, end_distribution = generate_markov_matrix(est_counts,
                                                          trans_domain)
    trans_distribution.append(markov_mat)

    # generate new points in synthetic data based on current distribution
    synthetic_db.generate_new_points_baseline(markov_mat, grid_map)
    used_budget_2.append(sampled_users)
    for uid in sampled_users:
        users.deactivate(uid)
    used_budget_1.append([])
    quited_users = []

    l = 1
    eps_l2 = int(len(traj_stream[1])/(w*2))

    for t in range(2, len(traj_stream)-1):
        if not len(traj_stream[t]):
            continue
        # user recycling
        if t >= w:
            for uid in used_budget_1[t - w]:
                users.recycle(uid)
            for uid in used_budget_2[t - w]:
                users.recycle(uid)

        # remove quited users
        for uid in quited_users:
            users.remove(uid)
        quited_users = []

        for (g1, g2, flag, uid) in traj_stream[t]:
            users.register(uid)
            if flag == 2:
                quited_users.append(uid)

        # set dissimilarity budget
        sampled_users = users.sample(1 / (2 * w))

        if len(sampled_users) == 0:
            release.append(release[-1])
            used_budget_2.append([])
            markov_mat, end_distribution = generate_markov_matrix(release[-1], trans_domain)
            synthetic_db.generate_new_points_baseline(markov_mat, grid_map)

            trans_distribution.append(markov_mat)
            continue

        # estimate c_t
        oue = OUE(eps, len(trans_domain), lambda x: trans_domain_map[x])

        for (g1, g2, flag, uid) in traj_stream[t]:
            if not users.users[uid] == 2:
                continue
            trans = Transition(g1, g2, flag)
            oue.privatise(trans)
        oue.adjust()

        for uid in sampled_users:
            users.deactivate(uid)
        used_budget_1.append(sampled_users)
        c_bar = oue.non_negative_data / oue.n

        # calculate dissimilarity

        dis = np.mean((c_bar - release[-1]) ** 2)
        dis -= 4 * math.exp(eps) / (oue.n * (math.exp(eps) - 1) ** 2)

        # calculate nullified timestamps
        t_N = eps_l2 / (len(traj_stream[t]) / (2 * w)) - 1

        if t - l <= t_N:
            # nullified timestamp
            release.append(release[-1])
            used_budget_2.append([])
        else:
            # calculate absorbed timestamps
            t_A = t - (l + t_N)
            N_pp = int(len(traj_stream[t])/(w*2)) * min(t_A, w)
            err = 4 * math.exp(eps) / (N_pp * (math.exp(eps) - 1) ** 2)

            if dis > err:
                sampled_users = users.sample(N_pp/len(users.available_users))
                # perturbation
                oue = OUE(eps, len(trans_domain), lambda x: trans_domain_map[x])

                for (g1, g2, flag, uid) in traj_stream[t]:
                    if not users.users[uid] == 2:
                        continue
                    trans = Transition(g1, g2, flag)
                    oue.privatise(trans)
                oue.adjust()
                est_counts = oue.non_negative_data / oue.n
                release.append(est_counts/est_counts.sum())
                used_budget_2.append(sampled_users)
                for uid in sampled_users:
                    users.deactivate(uid)
                l = t
                eps_l2 = N_pp
            else:
                # approximation
                release.append(release[-1])
                used_budget_2.append([])

        markov_mat, end_distribution = generate_markov_matrix(release[-1], trans_domain)
        synthetic_db.generate_new_points_baseline(markov_mat, grid_map)

        trans_distribution.append(markov_mat)

        if (t + 1) % 100 == 0:
            logger.info(f'{t + 1} timestamps processed')

    return synthetic_db


avg_lens = {
    'tdrive': 13.61,
    'porto_center': 12.00,
    'oldenburg': 59.98,
    'sanjoaquin': 55.3
}

timestamps = {
    'tdrive': 886,
    'oldenburg': 500,
    'sanjoaquin': 1000}

logger.info('Reading dataset...')
with lzma.open(f'../data/{args.dataset}_transition_id.xz', 'rb') as f:
    dataset = pickle.load(f)[:timestamps[args.dataset]]

stats = utils.tid_dataset_stats(dataset, f'../data/{args.dataset}_stats.json')
grid_map = GridMap(args.grid_num,
                   stats['min_x'],
                   stats['min_y'],
                   stats['max_x'],
                   stats['max_y'])

logger.info('Spatial decomposition...')
if args.multiprocessing:
    def decomp_multi(xy_l):
        return spatial_decomposition(xy_l, grid_map)


    pool = multiprocessing.Pool(CORES)
    grid_db = pool.map(decomp_multi, dataset)
    pool.close()
else:
    grid_db = [spatial_decomposition(xy_l, grid_map) for xy_l in dataset]

grid_db = split_traj(grid_db, grid_map)


if args.method == 'retrasyn':
    logger.info('RetraSyn...')
    syn_grid_db = RetraSyn(grid_db, args.w, args.epsilon,
                           grid_map.get_all_transition())
elif args.method == 'lpd':
    logger.info('LPD...')
    syn_grid_db = lpd(grid_db, args.w, args.epsilon, grid_map.get_all_transition())
elif args.method == 'lpa':
    logger.info('LPA...')
    syn_grid_db = lpa(grid_db, args.w, args.epsilon, grid_map.get_all_transition())
else:
    logger.info('Invalid method name!')
    exit()

syn_xy_db = convert_grid_to_raw(syn_grid_db.all_data)
with open(
        f'../data/syn_data/{args.dataset}/{args.method}_{args.epsilon}_g{args.grid_num}_w{args.w}_p.pkl',
        'wb') as f:
    pickle.dump(syn_xy_db, f)
