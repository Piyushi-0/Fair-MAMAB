import numpy as np
import argparse
import random
import pickle
import os
import cvxpy as cp
from utils import load_i_instance_nk, createLogHandler

parser = argparse.ArgumentParser(description='fmamab')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--c', type=float, default=0.2)
parser.add_argument('--T', type=int, default=10000)
parser.add_argument('--distr', type=str, default='binomial')
parser.add_argument('--save_as', type=str, default='SWF')
parser.add_argument('--A_choice', type=int, default=0)
parser.add_argument('--SEED', type=int, default=0)
parser.add_argument('--cwarn', type=int, default=1)
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--n', type=int, default=4)
parser.add_argument('--c_idx', type=str, default='0')
args = parser.parse_args()

SEED = args.SEED

"""
Loading hyperparameters
"""
n = args.n
k = args.k
arms = np.arange(k)
alpha = args.alpha
T = args.T
distr = args.distr
A_choice = args.A_choice

if args.c_idx == '0':
    c = np.array(n*[args.c])
    if args.cwarn:
        assert (c <= 1/min(n, k)).all(), "This c could make it infesible"
else:
    # NOTE: c shouldn't change with SEED. For a given A of a given n, it should be the same.
    np.random.seed(n)
    random.seed(n)
    
    c = np.random.rand(n)
    c = c/c.sum()

"""
SEEDing
"""
np.random.seed(SEED)
random.seed(SEED)

fldr = f'{args.save_as}/{n}_{k}/{args.c_idx}'
if args.c_idx == '0':
    fldr = f'{fldr}/{args.c}'
save_as = f'{fldr}/{alpha}_{T}_{A_choice}_{SEED}'
os.makedirs(save_as, exist_ok=True)

T_alpha = int(T**alpha)
T_alpha = T_alpha + 1 if T_alpha%2 else T_alpha
cnt_empty = 0 # used while computing the ratios

fname = "logs"

logger = createLogHandler(f'{save_as}/{fname}.csv', str(os.getpid()))

if not os.path.exists(f'{save_as}/{fname}.csv'):
    logger.info(", SEED, SWR, FR") # we will write the header

def get_samples(A, chosen_j, distr=distr):
    means = A[:, chosen_j]
    if distr == 'binomial':
        return np.random.binomial(1, p=means)
    elif distr == "beta":
        return np.random.beta(means, 1-means)

def pull_arm(arms, A, p, distr=distr):
    chosen_j = np.random.choice(a=arms, p=p)
    sample_reward = get_samples(A, chosen_j, distr=distr)
    return chosen_j, sample_reward

def get_fair_regret(A, pi_t, c_mu_star):
    return np.sum(np.clip(c_mu_star - A@pi_t, a_min=0, a_max=np.inf))

def normalize_p(p):
    if p is None:
        return p
    p = np.clip(p, 0, 1)
    p = p / np.sum(p)
    return p

def solve_LP(A, c_mu_star, k=2):
    def get_obj(p):
        sw = cp.sum(cp.matmul(A, p))
        return sw
    
    p = cp.Variable(k)
    objective = cp.Maximize(get_obj(p))
    constraints = [cp.sum(p) == 1, p >= 0, cp.matmul(A, p) >= c_mu_star]
    problem = cp.Problem(objective, constraints)
    result = problem.solve()
    return normalize_p(p.value)

"""
Loading A and assigning quantities
"""
if A_choice == 0 and n ==4 and k == 3:
    A = np.array([[0.1/0.33, 0.2/0.33, 0], [0.3/0.33, 0, 1], [0.15/0.33, 0.1/0.33, 0.1/0.33], [0, 0.2/0.33, 0.1/0.33]])
else:
    A = load_i_instance_nk(n, k, A_choice)

c_mu_star = c*A.max(1)
pi_star = solve_LP(A, c_mu_star, k=k)
A_pi_star = A@pi_star

"""
Starting the algorithm
"""
A_hat = np.zeros_like(A)
n_arms = np.zeros(k)

# NOTE: EXPLORATION PHASE where each has equal prob of pulling

pi_t = np.ones(k)/k
swr = np.sum(A_pi_star - A@pi_t)
fr = get_fair_regret(A, pi_t, c_mu_star)

logger.info(f"{T_alpha}")
logger.info(f"{swr}, {fr}")

for t in range(T_alpha):
    chosen_j, sample_reward = pull_arm(arms, A, pi_t)
    A_hat[:, chosen_j] += sample_reward
    n_arms[chosen_j] += 1

A_hat[:, n_arms>0] = A_hat[:, n_arms>0]/n_arms[n_arms>0]

# NOTE: EXPLOITATION PHASE
pi_t = solve_LP(A_hat, c*A_hat.max(1), k=k)
swr =  np.sum(A_pi_star - A@pi_t)
fr = get_fair_regret(A, pi_t, c_mu_star)
logger.info(f"{swr}, {fr}")
