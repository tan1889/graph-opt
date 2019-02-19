"""
Implementation of the spectral bandit algorithm for fixed budget stochastic graph function optimisation

Algorithm: spectral bandit by Michal Valko et. al. "Spectral Bandits for Smooth Graph Functions"

Adaptation for fixed budget stochastic graph function optimisation: Consider each node in the graph as
one arm of a big bandit problem (ignoring the graph structure), apply the spectral bandit algorithm
on this bandit problem to approximate the best node given a fixed budget.

Please quote our paper in your work: T Nguyen, Y Abbasi-Yadkori, B Kveton, A Shameli, A Rao.
"Sample Efficient Graph-Based Optimization with Noisy Observations."
Artificial Intelligence and Statistics. 2019.

"""
import math
import time
import pickle
import random
import argparse
import numpy as np
from datetime import datetime


parser = argparse.ArgumentParser(description='FIXED BUDGET 1BANDIT SPECTRAL BANDIT')
parser.add_argument('--filename', type=str, default='synthetic_graph_r10.pkl', metavar='F',
                    help='Dataset filename. One of the files in \data directory. Default: synthetic_graph_r10.pkl')
parser.add_argument('--n-trials', type=int, default=10, metavar='S',
                    help='Number of trials to run and output average result. Default: 10')
parser.add_argument('--budget', type=int, default=1000, metavar='S',
                    help='Total budget in number of samples. Default: 1000')
parser.add_argument('--verbose', type=int, default=2, metavar='V',
                    help='Level of verbose (1 - 3). Default: 2')
args = parser.parse_args()

# read data
t0 = time.time()
random.seed(1)      # fixed random seed
with open('data/' + args.filename, "rb") as f_in:
    f, AL = pickle.load(f_in)

n_nodes = len(f)
rank = sorted(range(n_nodes), key=lambda x: f[x], reverse=True)  # rank[0] = max_node, rank[1] = second max etc.

print('\ndataset={}, n_nodes={}, global maximum f_max={:.4f} is at node {}'.format(
    args.filename, n_nodes, f[rank[0]], rank[0]))
print("1BANDIT SPECTRAL params: n_trials={}, budget={}\n".format(args.n_trials, args.budget))


class GraphBandit:

    def __init__(self, A, f):
        self.f = f
        self.A = A
        self.K = len(f)  # number of arms = nodes in the graph

        L = np.diag(A.sum(axis=0)) - A
        self.Lambda, U = np.linalg.eigh(L)
        self.Lambda = self.Lambda[:: -1]
        self.X = U[:, :: -1].T

        self.best_arm = np.argmax(f)
        self.nf = self.mf = self.n_samples = None
        self.reset()

    def reset(self):
        self.n_samples = 0  # total number of queries to obtain function values
        self.nf = [0 for _ in range(self.K)]     # nf[i] number of samples at node i, sum(nf) = n_samples
        self.mf = [0. for _ in range(self.K)]     # mf[i] empirical mean of samples of f at node i

    def sample_f(self, x):
        fx = 1. if random.uniform(0., 1.) <= self.f[x] else 0.  # Bernoulli with prob = f[x]
        self.n_samples += 1
        self.nf[x] += 1
        mfx0 = self.mf[x]
        # update mean incrementally
        self.mf[x] = (fx + (self.nf[x] - 1) * mfx0) / self.nf[x]
        return fx

    def pull(self, arm):
        """pull arm & returns reward """
        return self.sample_f(arm)

    def print_top_mf(self):
        m = sorted(range(n_nodes), key=lambda z: -self.nf[z])
        for p in range(min(n_nodes, 10)):
            i = m[p]
            print('    node={}, mean={:.5f}\tn_samples={}  \tf={:.5f}'.format(i, self.mf[i], self.nf[i], self.f[i]))
        print('')


class SpectralTS:
    def __init__(self, env):
        self.X = env.X
        self.K = self.X.shape[0]
        self.d = self.X.shape[1]
        self.sigma0 = 1
        self.sigma = 0.5  # env.sigma  why the algorithm needs to know the environment sigma?

        # sufficient statistics
        self.Gram = np.diag(env.Lambda) + np.eye(self.d) / (self.sigma0 * self.sigma0)
        self.B = np.zeros(self.d)

        self.mu = np.zeros(self.K)  # posterior sample

    def update(self, arm, reward):
        self.Gram += np.outer(self.X[arm, :], self.X[arm, :]) / (self.sigma * self.sigma)
        self.B += self.X[arm, :] * reward

        # posterior sampling
        Gram_inv = np.linalg.inv(self.Gram)
        theta_bar = np.linalg.inv(self.Gram).dot(self.B) / (self.sigma * self.sigma)
        theta = np.random.multivariate_normal(theta_bar, Gram_inv)
        self.mu = self.X.dot(theta)

    def get_arm(self):
        arm = np.argmax(self.mu)
        return arm


# make adjacency matrix A
A = np.zeros((n_nodes, n_nodes), int)
for i in range(n_nodes):
    for j in AL[i]:
        A[i, j] = 1

env = GraphBandit(A, f)
results = []  # list of results for each trial

for r in range(args.n_trials):
    env.reset()  # reset the environment for each trial

    alg = SpectralTS(env)

    for t in range(1, args.budget+1):
        arm = alg.get_arm()
        reward = env.pull(arm)
        alg.update(arm, reward)  # update model and regret
        if t % 100 == 0 and args.verbose > 2:
            print('Total n_samples={}. Nodes with top empirical means:'.format(env.n_samples))
            env.print_top_mf()

    # max_node = alg.get_arm()  # get the arm with highest empirical mean
    max_node = max(range(n_nodes), key=lambda z: env.nf[z])  # get the arm pulled the most
    max_node_rank = rank.index(max_node) + 1
    results.append({'max_node': max_node, 'max_node_value': f[max_node], 'max_node_rank': max_node_rank,
                    'true_max_node': rank[0], 'true_max_node_value': f[rank[0]]})

    if args.verbose > 1:
        print('Trial #{}:  n_samples = {},  f^ = {:.4f} @ node = {},  rank = {},  f* - f^ = {:.4f}'.format(
            r + 1, env.n_samples, f[max_node], max_node, max_node_rank, f[rank[0]] - f[max_node]))
        if args.verbose > 2:
            env.print_top_mf()


print('\nFIXED BUDGET 1BANDIT SPECTRAL STATISTICS: budget={} n_trials={}'.format(
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

result = {'algorithm': '1bandit_spectral', 'args': args, 'run-time': elapsed_time,
          'subopt-gaps-mean': gaps.mean(), 'subopt-gaps-std': gaps.std(), 'rank-stats': rank_stats}

f = 'output/{}_budget.{}_1bandit_spectral_{:%Y%m%d.%H%M}.pkl'.format(
    args.filename.replace('.pkl', ''), args.budget, datetime.now())

pickle.dump(result, open(f, "wb"))
print('\nRESULT SAVED:', f)
