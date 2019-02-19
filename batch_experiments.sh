#!/bin/sh

# sh script to run all experiments, 1000 trials for each experiment
# 4 algorithms are (1) hill climbing, (2) simulated annealing, (3) successive reject, (4) spectral bandit
# Execute this file by running this command: sh batch_experiments.sh

# =============================================================
# data/graph_q8107157_n3992.pkl: 3992 nodes. Only test algorithms (1)..(3) as (4) can not handle large graph
# Test at budget = 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000
# =============================================================

python fixed_budget_hill_climbing_sreject_non_convex.py --filename 'graph_q8107157_n3992.pkl' --budget 500  --n-trials 1000 --max-path-len 8 --verbose 1
python fixed_budget_hill_climbing_sreject_non_convex.py --filename 'graph_q8107157_n3992.pkl' --budget 1000  --n-trials 1000 --max-path-len 8 --verbose 1
python fixed_budget_hill_climbing_sreject_non_convex.py --filename 'graph_q8107157_n3992.pkl' --budget 2000  --n-trials 1000 --max-path-len 8 --verbose 1
python fixed_budget_hill_climbing_sreject_non_convex.py --filename 'graph_q8107157_n3992.pkl' --budget 3000  --n-trials 1000 --max-path-len 8 --verbose 1
python fixed_budget_hill_climbing_sreject_non_convex.py --filename 'graph_q8107157_n3992.pkl' --budget 4000  --n-trials 1000 --max-path-len 8 --verbose 1
python fixed_budget_hill_climbing_sreject_non_convex.py --filename 'graph_q8107157_n3992.pkl' --budget 4500  --n-trials 1000 --max-path-len 8 --verbose 1
python fixed_budget_hill_climbing_sreject_non_convex.py --filename 'graph_q8107157_n3992.pkl' --budget 5000  --n-trials 1000 --max-path-len 8 --verbose 1
python fixed_budget_hill_climbing_sreject_non_convex.py --filename 'graph_q8107157_n3992.pkl' --budget 6000  --n-trials 1000 --max-path-len 8 --verbose 1
python fixed_budget_hill_climbing_sreject_non_convex.py --filename 'graph_q8107157_n3992.pkl' --budget 7000  --n-trials 1000 --max-path-len 8 --verbose 1

python fixed_budget_simulated_annealing.py --filename 'graph_q8107157_n3992.pkl' --budget 500  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'graph_q8107157_n3992.pkl' --budget 1000  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'graph_q8107157_n3992.pkl' --budget 2000  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'graph_q8107157_n3992.pkl' --budget 3000  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'graph_q8107157_n3992.pkl' --budget 4000  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'graph_q8107157_n3992.pkl' --budget 4500  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'graph_q8107157_n3992.pkl' --budget 5000  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'graph_q8107157_n3992.pkl' --budget 6000  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'graph_q8107157_n3992.pkl' --budget 7000  --n-trials 1000 --min-sampling 1 --verbose 1

python fixed_budget_simulated_annealing.py --filename 'graph_q8107157_n3992.pkl' --budget 500  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'graph_q8107157_n3992.pkl' --budget 1000  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'graph_q8107157_n3992.pkl' --budget 2000  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'graph_q8107157_n3992.pkl' --budget 3000  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'graph_q8107157_n3992.pkl' --budget 4000  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'graph_q8107157_n3992.pkl' --budget 4500  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'graph_q8107157_n3992.pkl' --budget 5000  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'graph_q8107157_n3992.pkl' --budget 6000  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'graph_q8107157_n3992.pkl' --budget 7000  --n-trials 1000 --min-sampling 5 --verbose 1

python fixed_budget_1bandit_sreject.py --filename 'graph_q8107157_n3992.pkl' --budget 500  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'graph_q8107157_n3992.pkl' --budget 1000  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'graph_q8107157_n3992.pkl' --budget 2000  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'graph_q8107157_n3992.pkl' --budget 3000  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'graph_q8107157_n3992.pkl' --budget 4000  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'graph_q8107157_n3992.pkl' --budget 4500  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'graph_q8107157_n3992.pkl' --budget 5000  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'graph_q8107157_n3992.pkl' --budget 6000  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'graph_q8107157_n3992.pkl' --budget 7000  --n-trials 1000 --verbose 1


# =============================================================
# data/synthetic_graph_r100.pkl: 40401 nodes. Only test algorithms (1)..(3) as (4) can not handle large graph
# Test at budget = 1000, 5000, 10000, 15000, ..., 60000
# =============================================================

python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 1000  --n-trials 1000 --max-path-len 20 --verbose 1
python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 5000  --n-trials 1000 --max-path-len 20 --verbose 1
python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 10000  --n-trials 1000 --max-path-len 20 --verbose 1
python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 15000  --n-trials 1000 --max-path-len 20 --verbose 1
python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 20000  --n-trials 1000 --max-path-len 20 --verbose 1
python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 25000  --n-trials 1000 --max-path-len 20 --verbose 1
python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 30000  --n-trials 1000 --max-path-len 20 --verbose 1
python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 35000  --n-trials 1000 --max-path-len 20 --verbose 1
python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 40000  --n-trials 1000 --max-path-len 20 --verbose 1
python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 42000  --n-trials 1000 --max-path-len 20 --verbose 1
python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 45000  --n-trials 1000 --max-path-len 20 --verbose 1
python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 50000  --n-trials 1000 --max-path-len 20 --verbose 1
python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 55000  --n-trials 1000 --max-path-len 20 --verbose 1
python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 60000  --n-trials 1000 --max-path-len 20 --verbose 1
python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 65000  --n-trials 1000 --max-path-len 20 --verbose 1
python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 70000  --n-trials 1000 --max-path-len 20 --verbose 1

python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 1000  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 5000  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 10000  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 15000  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 20000  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 25000  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 30000  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 35000  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 40000  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 42000  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 45000  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 50000  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 55000  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 60000  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 65000  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 70000  --n-trials 1000 --min-sampling 1 --verbose 1

python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 1000  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 5000  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 10000  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 15000  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 20000  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 25000  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 30000  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 35000  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 40000  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 42000  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 45000  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 50000  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 55000  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 60000  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 65000  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r100.pkl' --budget 70000  --n-trials 1000 --min-sampling 5 --verbose 1

python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 1000  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 5000  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 10000  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 1500  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 20000  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 25000  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 30000  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 35000  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 40000  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 42000  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 45000  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 50000  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 55000  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 60000  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 65000  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r100.pkl' --budget 70000  --n-trials 1000 --verbose 1


# =============================================================
# data/synthetic_graph_r10.pkl: 441 nodes. Test all 4 algorithms.
# Test at budget = 100, 200, ..., 800
# =============================================================

python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r10.pkl' --budget 100  --n-trials 1000 --max-path-len 4 --verbose 1
python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r10.pkl' --budget 200  --n-trials 1000 --max-path-len 4 --verbose 1
python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r10.pkl' --budget 300  --n-trials 1000 --max-path-len 4 --verbose 1
python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r10.pkl' --budget 400  --n-trials 1000 --max-path-len 4 --verbose 1
python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r10.pkl' --budget 450  --n-trials 1000 --max-path-len 4 --verbose 1
python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r10.pkl' --budget 500  --n-trials 1000 --max-path-len 4 --verbose 1
python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r10.pkl' --budget 600  --n-trials 1000 --max-path-len 4 --verbose 1
python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r10.pkl' --budget 700  --n-trials 1000 --max-path-len 4 --verbose 1
python fixed_budget_hill_climbing_sreject.py --filename 'synthetic_graph_r10.pkl' --budget 800  --n-trials 1000 --max-path-len 4 --verbose 1

python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r10.pkl' --budget 100  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r10.pkl' --budget 200  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r10.pkl' --budget 300  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r10.pkl' --budget 400  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r10.pkl' --budget 450  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r10.pkl' --budget 500  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r10.pkl' --budget 600  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r10.pkl' --budget 700  --n-trials 1000 --min-sampling 1 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r10.pkl' --budget 800  --n-trials 1000 --min-sampling 1 --verbose 1

python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r10.pkl' --budget 100  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r10.pkl' --budget 200  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r10.pkl' --budget 300  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r10.pkl' --budget 400  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r10.pkl' --budget 450  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r10.pkl' --budget 500  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r10.pkl' --budget 600  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r10.pkl' --budget 700  --n-trials 1000 --min-sampling 5 --verbose 1
python fixed_budget_simulated_annealing.py --filename 'synthetic_graph_r10.pkl' --budget 800  --n-trials 1000 --min-sampling 5 --verbose 1

python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r10.pkl' --budget 100  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r10.pkl' --budget 200  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r10.pkl' --budget 300  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r10.pkl' --budget 400  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r10.pkl' --budget 450  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r10.pkl' --budget 500  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r10.pkl' --budget 600  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r10.pkl' --budget 700  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_sreject.py --filename 'synthetic_graph_r10.pkl' --budget 800  --n-trials 1000 --verbose 1

python fixed_budget_1bandit_spectral.py --filename 'synthetic_graph_r10.pkl' --budget 100  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_spectral.py --filename 'synthetic_graph_r10.pkl' --budget 200  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_spectral.py --filename 'synthetic_graph_r10.pkl' --budget 300  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_spectral.py --filename 'synthetic_graph_r10.pkl' --budget 400  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_spectral.py --filename 'synthetic_graph_r10.pkl' --budget 450  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_spectral.py --filename 'synthetic_graph_r10.pkl' --budget 500  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_spectral.py --filename 'synthetic_graph_r10.pkl' --budget 600  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_spectral.py --filename 'synthetic_graph_r10.pkl' --budget 700  --n-trials 1000 --verbose 1
python fixed_budget_1bandit_spectral.py --filename 'synthetic_graph_r10.pkl' --budget 800  --n-trials 1000 --verbose 1
