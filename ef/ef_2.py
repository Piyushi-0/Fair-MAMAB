import numpy as np
import argparse
import random
import pickle
import os
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
parser.add_argument('--c_idx', type=str, default='0')
parser.add_argument('--n', type=int, default=3)
args = parser.parse_args()

SEED = args.SEED

"""
Loading hyperparameters
"""
n = args.n
k = 2
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
    logger.info(", SEED, SWR, FR")

def min_ratio(A_mat, c_vec, j_star, j_dash):
    """
    ratio that decides policy
    """
    global cnt_empty
    agent_idx = np.where(A_mat[:, j_star] < A_mat[:, j_dash])[0]
    ratio_means = A_mat[agent_idx, j_star] / A_mat[agent_idx, j_dash]
    if not agent_idx.size:
        cnt_empty += 1
        r = np.inf
    else:
        r = min((1-c_vec[agent_idx])/(1-ratio_means))
    return r

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

"""
Loading A and assigning quantities
"""

if A_choice == 0 and n == 3 and k == 2:
    A = np.array([[0.7, 0.3], [0.1, 0.5], [1, 0.6]])
else:
    A = load_i_instance_nk(n, k, A_choice)

def get_pi(A_tilde):
    sum_means = A_tilde.sum(0)
    if sum_means[0] == sum_means[1]:
        j_star = np.random.choice(a=arms, p=[0.5, 0.5])
    else:
        j_star = np.argmax(sum_means)
    j_dash = 1-j_star
    Delta = sum_means[j_star] - sum_means[j_dash]
    x_star = min(1, min_ratio(A_tilde, c, j_star, j_dash))
    pi_star = np.array([x_star, 1-x_star])
    if j_star == 1:
        pi_star = pi_star[::-1]
    return pi_star, j_star, j_dash, Delta

pi_star, j_star, j_dash, Delta = get_pi(A)
arms = np.array([j_star, j_dash])

A_pi_star = A@pi_star
c_mu_star = c*A.max(1) # for fairness

"""
Starting the algorithm
"""
A_hat = np.zeros_like(A)
n_arms = np.zeros(k)

pi_t = np.ones(k)/k
# NOTE: Exploration phase regret. Logging only once. While plotting, we do cumulative sum till T_alpha.
swr = np.sum(A_pi_star - A@pi_t)
fr = get_fair_regret(A, pi_t, c_mu_star)

logger.info(f"{T_alpha}")
logger.info(f"{swr}, {fr}")

for t in range(T_alpha): # Exploration phase.
    chosen_j, sample_reward = pull_arm(arms, A, pi_t)
    A_hat[:, chosen_j] += sample_reward
    n_arms[chosen_j] += 1

A_hat[:, n_arms>0] = A_hat[:, n_arms>0]/n_arms[n_arms>0]

# NOTE: Exploitation phase regret. Logging only once. While plotting, we do cumulative sum till T.
pi_t = get_pi(A_hat)[0]
swr = np.sum(A_pi_star - A@pi_t)
fr = get_fair_regret(A, pi_t, c_mu_star)
logger.info(f"{swr}, {fr}")
