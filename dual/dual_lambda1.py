import argparse
import numpy as np
import pickle
import random, os
import cvxpy as cp
from cvxpy import Maximize, Problem
import logging
from tqdm import tqdm
import warnings
import pickle
from utils import load_i_instance_nk
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def createLogHandler(log_file, job_name="_"):
    logger = logging.getLogger(job_name)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log_file, mode='a')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s; , %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

parser = argparse.ArgumentParser(description='dual')
parser.add_argument('--n', type=int, default=3, help='number of arms')
parser.add_argument('--k', type=int, default=2, help='number of arms')
parser.add_argument('--T', type=int, default=1000, help='number of rounds')
parser.add_argument('--c', type=float, default=0.5, help='for fairness')
parser.add_argument('--SEED', type=int, default=0, help='seed')
parser.add_argument('--ns', type=int, default=100, help='no. samples')
parser.add_argument('--save_as', default='prp')
parser.add_argument('--A_choice', type=int, default=0)
args = parser.parse_args()

SEED = args.SEED
np.random.seed(SEED)
random.seed(SEED)

T = args.T
c = args.c
n = args.n
k = args.k
ns = args.ns
A_choice = args.A_choice
L = int(np.ceil(np.sqrt(T)))
Lk = L*k
arms = np.arange(k)

if A_choice == 0:
    if n == 3 and k == 2:
        A = np.array([[0.7, 0.3], [0.1, 0.5], [1, 0.6]])
    elif n == 4 and k == 3:    
        A = np.array([[0.1/0.33, 0.2/0.33, 0], [0.3/0.33, 0, 1], [0.15/0.33, 0.1/0.33, 0.1/0.33], [0, 0.2/0.33, 0.1/0.33]])
else:
    A = load_i_instance_nk(n, k, A_choice)

def normalize_p(p):
    if p is None:
        return p
    p = np.clip(p, 0, 1)
    p = p / np.sum(p)
    return p

def solve_LP(A_over, c_mu_star_under, k=k):
    def get_obj(p):
        sw = cp.sum(cp.matmul(A_over, p))
        return sw
    
    p = cp.Variable(k)
    objective = cp.Maximize(get_obj(p))
    constraints = [cp.sum(p) == 1, p >= 0, cp.matmul(A_over, p) >= c_mu_star_under]
    problem = cp.Problem(objective, constraints)
    result = problem.solve()
    return normalize_p(p.value)

c_mu_star = c*A.max(1)
pi_star = solve_LP(A, c_mu_star)
A_pi_star = A@pi_star
A_hat = np.zeros((n, k), dtype=float)

A_over = np.ones((n, k), dtype=float)
A_under = A_hat.copy()

fldr = f'{args.save_as}/{c}'
save_as = f'{fldr}/n{n}_k{k}_{T}_{A_choice}'
os.makedirs(save_as, exist_ok=True)
fname = f"{save_as}/logs_{SEED}"
logger = createLogHandler(f"{fname}.csv", str(os.getpid()))

def get_fair_regret(A, pi_t, c_mu_star):
    return np.sum(np.clip(c_mu_star - A@pi_t, a_min=0, a_max=np.inf))

def get_samples(A, chosen_j, ns=1, distr='binomial'):
    means = A[:, chosen_j]
    if distr == 'binomial':
        samples = np.random.binomial(ns, p=means)
    elif distr == "beta":
        samples = np.random.beta(means, 1-means)
    return samples

def pull_arm(arms, A, p, distr='binomial'):
    chosen_j = np.random.choice(a=arms, p=p)
    sample_reward = get_samples(A, chosen_j, distr=distr)
    return chosen_j, sample_reward

Nr = np.sqrt(np.log(2*n*k*(T+1)))  # Should've been just T

tot_SWR = np.zeros(T, dtype=float)
tot_FR = np.zeros(T, dtype=float)

n_arms = np.zeros(k)

for t in range(Lk):
    chosen_j = t%k
    sample_reward = get_samples(A, chosen_j)
    A_hat[:, chosen_j] = (A_hat[:, chosen_j]*n_arms[chosen_j] + sample_reward)/(n_arms[chosen_j] + 1)
    n_arms[chosen_j] += 1
    
    if t<k-1:
        pi_t = solve_LP(A_hat, c*A_hat.max(1))
    else:
        if t == k-1:
            eps = Nr/np.sqrt(n_arms)
            A_under = np.clip(A_hat - eps, 0, 1)
            A_over = np.clip(A_hat + eps, 0, 1)
        else:
            eps[chosen_j] = Nr/np.sqrt(n_arms[chosen_j])
            A_under[:, chosen_j] = np.clip(A_hat[:, chosen_j] - eps[chosen_j], 0, 1)
            A_over[:, chosen_j] = np.clip(A_hat[:, chosen_j] + eps[chosen_j], 0, 1)

        pi_t = solve_LP(A_over, c*A_under.max(1))
    
    swr = np.sum(A_pi_star - A@pi_t)
    fr = get_fair_regret(A, pi_t, c_mu_star)
    
    tot_SWR[t] = swr if not t else tot_SWR[t-1] + swr
    tot_FR[t] = fr if not t else tot_FR[t-1] + fr
    logger.info(f", {tot_SWR[t]}, {tot_FR[t]}")

hat_A_ki = np.max(A_hat, 1)
lambda_cp = cp.Variable(n)
obj =  -cp.norm_inf(cp.sum(cp.multiply(1+cp.reshape(lambda_cp, (n, 1)), A_hat), 0)) + c*cp.sum(cp.multiply(lambda_cp, hat_A_ki))
constraints = [lambda_cp >= 0]
problem = Problem(Maximize(obj), constraints)
problem.solve()
lambda_hat = lambda_cp.value

for t in range(Lk, T):
    UCB_indices = (np.diag(1+lambda_hat)@A_hat).sum(0) + Nr/np.sqrt(n_arms)
    chosen_j = np.argmax(UCB_indices)
    
    samples_for_chosen = get_samples(A, chosen_j, ns=1)
    A_hat[:, chosen_j] = (A_hat[:, chosen_j]*n_arms[chosen_j] + samples_for_chosen)/(n_arms[chosen_j] + 1)
    n_arms[chosen_j] += 1
    
    pi_t = solve_LP(A_hat, c*A_hat.max(1))
    swr = np.sum(A_pi_star - A@pi_t)
    fr = get_fair_regret(A, pi_t, c_mu_star)

    tot_SWR[t] = tot_SWR[t-1] + swr
    tot_FR[t] = tot_FR[t-1] + fr
    logger.info(f", {tot_SWR[t]}, {tot_FR[t]}")
