import logging
import pickle

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

def load_all_instance_nk(n, k):
    fname = "../mu_instance.pkl"
    with open(fname, 'rb') as f:
        all_dict = pickle.load(f)
    key = "n_"+str(n)+"k_"+str(k)
    return all_dict[key]

def load_i_instance_nk(n, k, i):
    all_instance = load_all_instance_nk(n, k)
    return all_instance[i]
