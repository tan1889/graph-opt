"""
Implementation of the hill climbing for fixed budget stochastic graph function optimisation

Algorithm: hill climbing with multiple restarts for non-concave graph, see our paper for details

Please quote our paper in your work: T Nguyen, Y Abbasi-Yadkori, B Kveton, A Shameli, A Rao.
"Sample Efficient Graph-Based Optimization with Noisy Observations."
Artificial Intelligence and Statistics. 2019.

"""
import pickle
import random
import math
import time
from datetime import datetime
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='FIXED BUDGET HILL CLIMBING SUCCESSIVE REJECT WITH RESTARTS')
parser.add_argument('--filename', type=str, default='synthetic_graph_r10.pkl', metavar='F',
                    help='Dataset filename. One of the files in \data directory. Default: synthetic_graph_r10.pkl')
parser.add_argument('--max-path-len', type=int, default=10, metavar='E',
                    help='Max path length from any point to the global maximum. Default: 10')
parser.add_argument('--n-trials', type=int, default=10, metavar='S',
                    help='Number of trials to run. Default: 10')
parser.add_argument('--budget', type=int, default=1000, metavar='S',
                    help='Total budget in number of samples. Default: 1000')
parser.add_argument('--verbose', type=int, default=2, metavar='V',
                    help='Level of verbose (1 - 3). Default: 2')
args = parser.parse_args()

args.n_restarts = math.ceil(args.budget / 1000)

# read data
t0 = time.time()
random.seed(1)      # fixed random seed
with open('data/' + args.filename, "rb") as f_in:
    f, A = pickle.load(f_in)

# find global maximum and print out (only for information)
n_nodes = len(f)
rank = sorted(range(n_nodes), key=lambda x: f[x], reverse=True)  # rank[0] = max_node, rank[1] = second max etc.

print('\ndataset={}, n_nodes={}, global maximum f_max={:.4f} is at node {}'.format(
    args.filename, n_nodes, f[rank[0]], rank[0]))
print("HILL CLIMBING SUCCESSIVE REJECT: max_path_len={}, n_trials={}, budget={}\n".format(
    args.max_path_len, args.n_trials, args.budget))


def sample_f(x):  # return a sample of f for node i, increase n_samples
    global f, n_samples
    f_val = 1. if random.uniform(0., 1.) <= f[x] else 0.  # Bernoulli with prob = f[x]
    n_samples += 1
    return f_val


def pull(x, n_pulls=1):  # pull node x n_pulls times and update mean, variance, confidence bound
    for _ in range(n_pulls):
        fx = sample_f(x)
        nf[x] += 1
        mfx0 = mf[x]
        # update mean incrementally
        mf[x] = (fx + (nf[x] - 1) * mfx0) / nf[x]


def get_arm_budget(phase, total_budget, n_arms):
    """get the number of pulls (budget) per arm in a given phase for a given total budget and n_arms"""
    assert 0 < phase < n_arms

    if total_budget <= n_arms:  # budget is smaller than number of arms -> use all budget in the first round
        return 1 if phase == 1 else 0

    logK = 0.5 + sum([1./i for i in range(2, n_arms + 1)])
    nk = lambda k: 0 if k <= 0 else math.ceil((total_budget - n_arms) / logK / (n_arms + 1 - k))
    return nk(phase) - nk(phase - 1)


def best_arm_sreject(arms, budget):  # return the best arm given budget using successive reject
    arms = list(arms)
    random.shuffle(arms)
    n_arms = len(arms)
    n_samples0 = n_samples
    if args.verbose > 2:
        print('  Identifying best arm by successive reject:  budget={},\n   arms: {}\n   {}'.format(
            budget, arms, top_arms(arms)))

    for k in range(1, n_arms):  # phase k <- 1 ... n_arms - 1
        pull_budget = get_arm_budget(k, budget, n_arms)

        if pull_budget > 0:
            max_n_pull = (budget - n_samples + n_samples0) // pull_budget
            if max_n_pull < len(arms):
                pull_list = random.sample(arms, max_n_pull)
            else:
                pull_list = arms
            for x in pull_list:
                pull(x, n_pulls=pull_budget)

        reject_node = min(arms, key=lambda x: mf[x])
        arms.remove(reject_node)

        if args.verbose > 2:  # print log
            print('  Phase #{}: n_samples={}, pull_budget_per_arm={} reject_node={}, active_nodes: len={}'.format(
                k, n_samples, pull_budget, reject_node, len(arms)))
            if len(arms) < 10:
                print('    {' + ''.join(['{}->f{}m{}n{} '.format(
                    x, round(f[x],4), round(mf[x],4), nf[x]) for x in arms]) + '}')

    return arms.pop()


def climb(budget):
    """given the budget, climb up the graph from a random node and return the path"""
    n_samples0 = n_samples
    x = random.randint(0, len(A) - 1)  # random starting node

    if args.verbose > 1:
        print('Started hill climbing:  random_start_node={},  budget={}'.format(x, budget))

    # This lines are for debugging: start from a troublesome node
    # x = 5100

    path = [x]
    for p in range(args.max_path_len):
        arms = set(A[x])
        arms.add(x)
        best_arm_budget = (budget - n_samples + n_samples0) // (args.max_path_len - p)
        z = best_arm_sreject(arms, best_arm_budget)
        if z != x:
            path.append(z)
            x = z
    return path


def list2string(l, decimal_places=5):  # return a string of decimal number in list l formatted nicely
    formatter = '{{:0.{}f}}'.format(decimal_places)
    s = '[' + ', '.join([formatter.format(x) for x in l]) + ']'
    return s


def top_arms(arms, top=8):
    arms = list(arms)
    ids = sorted(range(len(arms)), key=lambda i: f[arms[i]], reverse=True)
    s = 'tops: '
    for i in range(min(top, len(ids))):
        x = arms[ids[i]]
        s += '{}->f{}m{}n{}  '.format(x, round(f[x], 3), round(mf[x], 3), nf[x])
    return s


# ---------------- MAIN ----------------
results = []  # list of results for each trial

for r in range(args.n_trials):

    # reset the sampler for each trial
    n_samples = 0                   # total number of used queries (samples)
    nf = [0 for i in range(n_nodes)]      # nf[x] number of samples at node x, sum(nf) = n_samples
    mf = [0. for i in range(n_nodes)]     # mf[x] empirical mean of samples of f at node x

    maxima = []
    paths = []
    for _ in range(args.n_restarts):
        path = climb(int(0.95 * args.budget / args.n_restarts))
        paths.append(path)
        maxima.append(path[-1])

        if args.verbose > 1:
            print('Finished hill-climbing:  path_len={}\n   path={}\n     mf={}\n      f={}'.format(
                    len(path), path, list2string([mf[x] for x in path]), list2string([f[x] for x in path])))

    max_node = best_arm_sreject(maxima, args.budget - n_samples)  # max(maxima, key=lambda x: mf[x])
    max_node_rank = rank.index(max_node) + 1
    results.append({'max_node': max_node, 'max_node_value': f[max_node], 'max_node_rank': max_node_rank,
                    'true_max_node': rank[0], 'true_max_node_value': f[rank[0]]})

    if args.verbose > 0:
        print('Trial #{}:  n_samples = {},  f^ = {:.4f} @ node = {},  rank = {},  f* - f^ = {:.4f}'.format(
            r + 1, n_samples, f[max_node], max_node, max_node_rank, f[rank[0]] - f[max_node]))
        if args.verbose > 2:
            m = sorted(range(n_nodes), key=lambda z: -mf[z])
            for p in range(min(n_nodes, 10)):
                i = m[p]
                print('    node={}, mean={:.5f}\tn_samples={}\tf={:.5f}'.format(i, mf[i], nf[i], f[i]))
            print('')


print('HILL CLIMBING SUCCESSIVE REJECT WITH RESTARTS STATISTICS:   budget = {}   n_trials = {}'.format(
    args.budget, args.n_trials))

gaps = np.array([r['true_max_node_value'] - r['max_node_value'] for r in results])
counts = np.bincount(np.array([r['max_node_rank'] for r in results]))
rank_stats = ''
for i in range(counts.shape[0]):
    if counts[i] > 0:
        rank_stats += '{}={:.1f}%  '.format(i, 100. * counts[i] / args.n_trials)

print('    f* - f^ : average={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}'.format(
    gaps.mean(), gaps.std(), gaps.min(), gaps.max()))
print('    f^ ranks:', rank_stats)

elapsed_time = time.time() - t0
print('\nELAPSED TIME: {:.1f} SECONDS  (per trial: {:.2f}s)'.format(elapsed_time, elapsed_time/args.n_trials))


result = {'algorithm': 'hill-climb-restart', 'args': args, 'run-time': elapsed_time,
          'subopt-gaps-mean': gaps.mean(), 'subopt-gaps-std': gaps.std(), 'rank-stats': rank_stats}

f = 'output/{}_budget.{}_hill_climb_restart_{:%Y%m%d.%H%M}.pkl'.format(
    args.filename.replace('.pkl', ''), args.budget, datetime.now())

pickle.dump(result, open(f, "wb"))
print('\nRESULT SAVED:', f)
