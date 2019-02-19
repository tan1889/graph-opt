"""
Implementation of the simulated annealing for fixed budget stochastic graph function optimisation

Algorithm: simulated annealing, see our paper for details

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


parser = argparse.ArgumentParser(description='FIXED BUDGET SIMULATED ANNEALING')
parser.add_argument('--filename', type=str, default='synthetic_graph_r10.pkl', metavar='F',
                    help='Dataset filename. One of the files in \data directory. Default: synthetic_graph_r10.pkl')
parser.add_argument('--min-sampling', type=int, default=1, metavar='S',
                    help='Minimum number of samples each time a node is sampled. Default: 1')
parser.add_argument('--temperature', type=float, default=250., metavar='S',
                    help='Simulated annealing temperature. Default: 250')
parser.add_argument('--n-trials', type=int, default=10, metavar='S',
                    help='Number of trials to run and output average result. Default: 10')
parser.add_argument('--budget', type=int, default=1000, metavar='S',
                    help='Total budget in number of samples. Default: 1000')
parser.add_argument('--verbose', type=int, default=2, metavar='V',
                    help='Level of verbose (1 - 3). Default: 2')
args = parser.parse_args()


# parameters
LAMBDA = args.temperature
MIN_SAMPLING = args.min_sampling

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
print("SIMULATED ANNEALING params: n_trials={}, budget={}\n".format(args.n_trials, args.budget))


def sample_f(i):  # return a sample of f for node i, increase n_samples
    global f, n_samples
    f_val = 1.0 if random.uniform(0, 1) < f[i] else 0.0
    n_samples += 1
    return f_val


def pull(x, n_pulls=1):  # pull node x n_pulls times and return mean, std, number of samples
    if nf[x] + n_pulls < MIN_SAMPLING:
        n_pulls = MIN_SAMPLING - nf[x]
    for _ in range(n_pulls):
        fx = sample_f(x)
        nf[x] += 1
        mfx0 = mf[x]
        # update mean incrementally
        mf[x] = (fx + (nf[x] - 1) * mfx0) / nf[x]
        # update variance incrementally
        # vf[x] = (nf[x] - 1) * vf[x] / nf[x] + (nf[x] - 1) * (mf[x] - mfx0) ** 2 / nf[x] + (fx - mf[x]) ** 2 / nf[x]


def walk(budget):
    path = []
    x = random.randint(0, len(A) - 1)  # random starting node
    while True:
        path.append(x)
        while True:
            pull(x)
            z = random.choice(A[x])
            pull(z)
            if n_samples >= budget:
                return path
            p_move = min(1.0, math.exp(LAMBDA * (mf[z] - mf[x])))
            if random.uniform(0, 1) <= p_move:
                x = z
                break


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
        s += '{}->{:.4f}/{:.4f}/{}  '.format(x, f[x], mf[x], nf[x])
    return s


# ---------------- MAIN ----------------

results = []  # list of results for each trial

for r in range(args.n_trials):

    # reset the sampler for each trial
    n_samples = 0                   # total number of queries to obtain function values
    nf = [0 for _ in range(n_nodes)]      # nf[i] number of samples at node i, sum(nf) = n_samples
    mf = [0. for _ in range(n_nodes)]     # mf[i] empirical mean of samples of f at node i
    vf = [0. for _ in range(n_nodes)]     # vf[i] variance of samples of f at node i

    path = walk(args.budget)

    if args.verbose > 1:
        print('Finished Metropolis Walk:  path_len = {}\n   path={}\n   mf={}\n    f={}'.format(
            len(path), path, list2string([mf[x] for x in path]), list2string([f[x] for x in path])))

    top_nodes = path if len(path) <= 10 else path[-10:]
    max_node = max(top_nodes, key=lambda z: mf[z])
    max_node_rank = rank.index(max_node) + 1
    results.append({'max_node': max_node, 'max_node_value': f[max_node], 'max_node_rank': max_node_rank,
                    'true_max_node': rank[0], 'true_max_node_value': f[rank[0]]})

    if args.verbose > 0:
        print('Trial #{}:  n_samples = {},  f^ = {:.4f} @ node = {},  rank = {},  f* - f^ = {:.4f}'.format(
            r + 1, n_samples, f[max_node], max_node, max_node_rank, f[rank[0]] - f[max_node]))
        if args.verbose > 2:
            m = sorted(range(n_nodes), key=lambda z: -nf[z])
            for p in range(min(n_nodes, 10)):
                i = m[p]
                print('    node={}, mean={:.5f}\tn_samples={}\tf={:.5f}'.format(i, mf[i], nf[i], f[i]))
            print('')


print('\nFIXED BUDGET SIMULATED ANNEALING STATISTICS:   budget={}   n_trials={}'.format(
    args.budget, args.n_trials))

gaps = np.array([r['true_max_node_value'] - r['max_node_value'] for r in results])
counts = np.bincount(np.array([r['max_node_rank'] for r in results]))
rank_stats = ''
for i in range(counts.shape[0]):
    if counts[i] > 0:
        rank_stats += '{}->{:.1f}%  '.format(i, 100. * counts[i] / args.n_trials)

print('    f* - f^ :  average={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}'.format(
    gaps.mean(), gaps.std(), gaps.min(), gaps.max()))
print('    f^ ranks: ', rank_stats)

elapsed_time = time.time() - t0
print('\nELAPSED TIME: {:.1f} SECONDS  (per trial: {:.2f}s)'.format(elapsed_time, elapsed_time/args.n_trials))

result = {'algorithm': 'simanneal_{}'.format(MIN_SAMPLING), 'args': args, 'run-time': elapsed_time,
          'subopt-gaps-mean': gaps.mean(), 'subopt-gaps-std': gaps.std(), 'rank-stats': rank_stats}

f = 'output/{}_budget.{}_sim_anneal_{}_{:%Y%m%d.%H%M}.pkl'.format(
    args.filename.replace('.pkl', ''), args.budget, MIN_SAMPLING, datetime.now())

pickle.dump(result, open(f, "wb"))
print('\nRESULT SAVED:', f)
