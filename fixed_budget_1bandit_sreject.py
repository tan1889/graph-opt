"""
Implementation of the successive reject algorithm for fixed budget stochastic graph function optimisation

Algorithm: successive reject by Jean-Yves Audibert et. al. "Best Arm Identification in Multi-Armed Bandits"

Adaptation for fixed budget stochastic graph function optimisation: Consider each node in the graph as
one arm of a big bandit problem (ignoring the graph structure), apply the successive reject algorithm
on this bandit problem to approximate the best node given a fixed budget.

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


parser = argparse.ArgumentParser(description='FIXED BUDGET 1BANDIT SUCCESSIVE REJECT')
parser.add_argument('--filename', type=str, default='synthetic_graph_r10.pkl', metavar='F',
                    help='Dataset filename. One of the files in .\data directory. Default: synthetic_graph_r10.pkl')
parser.add_argument('--n-trials', type=int, default=10, metavar='S',
                    help='Number of trials to run. Program returns average result. Default: 10')
parser.add_argument('--budget', type=int, default=1000, metavar='S',
                    help='Total budget (number of samples). Default: 1000')
parser.add_argument('--verbose', type=int, default=2, metavar='V',
                    help='Level of verbose (1 - 3). Default: 2')
args = parser.parse_args()


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
print("1BANDIT SUCCESSIVE REJECT params: n_trials={}, budget={}\n".format(args.n_trials, args.budget))


def sample_f(x):  # return a sample of f for node x, increase n_samples
    global f, n_samples
    f_val = 1. if random.uniform(0., 1.) <= f[x] else 0.  # Bernoulli with prob = f[x]
    n_samples += 1
    return f_val


def pull(x, n_pulls=1):  # pull node x n_pulls times and update mean mf[x]
    global nf, mf
    for _ in range(n_pulls):
        fx = sample_f(x)
        nf[x] += 1
        mfx0 = mf[x]
        # update mean incrementally
        mf[x] = (fx + (nf[x] - 1) * mfx0) / nf[x]


# definition as in the original paper
logK = 0.5 + sum([1. / i for i in range(2, n_nodes + 1)])
nk = lambda k: 0 if k <= 0 else math.ceil((args.budget - n_nodes) / logK / (n_nodes + 1 - k))


def successive_reject():

    active_nodes = [x for x in range(n_nodes)]
    random.shuffle(active_nodes)

    # if budget is smaller than number of arms
    # -> use all budget in the first round, return arms with empirical max
    if args.budget <= n_nodes:
        pull_list = random.sample(active_nodes, args.budget)
        for x in pull_list:
            pull(x)
        return max(pull_list, key=lambda x: mf[x])

    k = 1  # phase k <- 1 ... n_nodes - 1
    while True:
        k0 = k
        pull_budget_per_arm = 0
        while pull_budget_per_arm == 0 and k < n_nodes:
            pull_budget_per_arm = nk(k) - nk(k - 1)
            k += 1

        n_no_pulls = k - k0 - 1
        if n_no_pulls > 0:
            # for n_no_pulls rounds there were no pull action, just rejects,
            # so we combine all reject rounds into one below to save time
            active_nodes.sort(key=lambda x: mf[x])
            active_nodes = active_nodes[n_no_pulls:]

        if pull_budget_per_arm > 0:
            max_n_pull = (args.budget - n_samples) // pull_budget_per_arm
            if max_n_pull < len(active_nodes):
                pull_list = random.sample(active_nodes, max_n_pull)
            else:
                pull_list = active_nodes
            for x in pull_list:
                pull(x, n_pulls=pull_budget_per_arm)

            reject_node = min(active_nodes, key=lambda x: mf[x])
            active_nodes.remove(reject_node)  # reject the smallest mean node

        if args.verbose > 2:  # print log
            print('Phase #{}: n_samples={}, pull_budget_per_arm={}, active_nodes: len={} {}'.format(
                k-1, n_samples, pull_budget_per_arm, len(active_nodes), '' if len(active_nodes) > 10 else active_nodes))

        if k == n_nodes or n_samples >= args.budget:
            return max(active_nodes, key=lambda x: mf[x])


# ---------------- MAIN ----------------

results = []  # list of results for each trial

for r in range(args.n_trials):

    # reset the sampler for each trial
    n_samples = 0                   # total number of queries to obtain function values
    nf = [0 for _ in range(n_nodes)]      # nf[i] number of samples at node i, sum(nf) = n_samples
    mf = [0. for _ in range(n_nodes)]     # mf[i] empirical mean of samples of f at node i

    max_node = successive_reject()
    max_node_rank = rank.index(max_node) + 1
    results.append({'max_node': max_node, 'max_node_value': f[max_node], 'max_node_rank': max_node_rank,
                    'true_max_node': rank[0], 'true_max_node_value': f[rank[0]]})
    if args.verbose > 1:
        print('Trial #{}:  n_samples = {},  f^ = {:.4f} @ node = {},  rank = {},  f* - f^ = {:.4f}'.format(
            r + 1, n_samples, f[max_node], max_node, max_node_rank, f[rank[0]] - f[max_node]))
        if args.verbose > 2:
            m = sorted(range(n_nodes), key=lambda z: -mf[z])
            for p in range(min(n_nodes, 10)):
                i = m[p]
                print('    node={}, mean={:.5f}\tn_samples={}\tf={:.5f}'.format(i, mf[i], nf[i], f[i]))
            print('')


print('\nFIXED BUDGET 1BANDIT SUCCESSIVE REJECT STATISTICS: budget={} n_trials={}'.format(
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

result = {'algorithm': '1bandit_sreject_fast', 'args': args, 'run-time': elapsed_time,
          'subopt-gaps-mean': gaps.mean(), 'subopt-gaps-std': gaps.std(), 'rank-stats': rank_stats}

f = 'output/{}_budget.{}_1bandit_sreject_fast_{:%Y%m%d.%H%M}.pkl'.format(
    args.filename.replace('.pkl', ''), args.budget, datetime.now())

pickle.dump(result, open(f, "wb"))
print('\nRESULT SAVED:', f)
