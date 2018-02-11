
from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np
import os, json, glob, multiprocessing
from pomdp_runner import PomdpRunner
from util import RunnerParams

########################
# Experiment Functions #
########################
MAX = np.inf
# CPUs = int(multiprocessing.cpu_count()*2/3)
CPUs = 2


def to_path(*args):
    return os.path.join(*args)

def execute(params):
    np.random.seed()
    params = RunnerParams(**params)

    with open(params.algo_config) as algo_config:
        algo_params = json.load(algo_config)
        runner = PomdpRunner(params)
        runner.run(**algo_params)


def main():
    def create_params():
        ret = [] 
        for i in range(num_tasks):
            base_params = {
                'config': 'pomcp',
                'env': 'GridWorld-2D.POMDP',
                'snapshot': False,
                'random_prior': False,
                'max_play': MAX,
                'budget': MAX
            }
            base_params.update({k: task_params[k][i] for k in task_params})
            ret.append(base_params)
        return ret

    for i in range(20, 120, 20):
        num_tasks = 10
        logfolder = './dev/logs/'
        logfiles = []
        for j in range(num_tasks):
            _dir, file = to_path(logfolder, 'budget={}'.format(i)), '{}.log'.format(j)
            if not os.path.exists(_dir):
                os.makedirs(_dir)
            if not os.path.exists(file):
                open(to_path(_dir, file), 'a').close()

            logfiles.append(to_path(_dir, file))

        task_params = {
            'budget': [i] * num_tasks,
            # 'max_play': [i] * num_tasks,
            'logfile': logfiles
        }

        num_workers = min(num_tasks, CPUs)
        pool = multiprocessing.Pool(num_workers)
        pool.map(execute, create_params())

        pool.close()
        pool.join()
        print('done')

############################
# Experiment Result Reader #
############################
def read_results(exp,subfolder):
    # logfolder = './dev/logs/'
    logfolder = '../experiment_results/pomcp_vs_bpomcp/'
    files = glob.glob(to_path(logfolder, subfolder, '*.log'))
    rewards = []
    for file in files:
        with open(file) as source:
            lines = source.read().splitlines()
            rewards.append(lines[-1].split()[-1])

    rewards = list(map(float, rewards))
    return rewards


def read_path(subfolder):
    logfolder = '../experiment_results/pomcp_vs_bpomcp/'
    files = glob.glob(to_path(logfolder, subfolder, '*.log'))
    board = np.zeros((7, 7))
    action_map = lambda action, i,j : {
        'up': (i - 1, j),
        'down': (i + 1, j),
        'left': (i, j - 1),
        'right': (i, j + 1),
        'halt': (i, j)
    }.get(action)

    for file in files:
        with open(file) as source:
            lines = [line for line in source.read().splitlines() if line.startswith('Taking action')]
            curr = (3, 3)
            board[curr[0], curr[1]] += 1
            for line in lines:
                action = line[15:]
                curr = action_map(action, *curr)
                board[curr[0], curr[1]] += 1

    return board



if __name__ == '__main__':
    pass
    # data = []
    # for i in [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]:
    # for i in range(1, 8, 1):
        # print '======= budget is {} ======='.format(i)
        # print read_path('budget={}'.format(i))
        # data.append(read_results('budget={}'.format(i)))
    
    # data = np.array(data)
    # print np.mean(data, axis=1)
