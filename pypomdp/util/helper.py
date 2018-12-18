import numpy as np
import math, random, time, sys

from numba import jit, jitclass
from collections import Counter
from functools import wraps


np.random.seed()
random.seed()
MAX = np.inf

def timeit(comment=None):
    """
        auto establish and close mongo connection
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            begintime = time.time()
            ret = func(*args,**kwargs)
            duration = (time.time() - begintime).total_seconds()/60
            
            if comment is not None:
                msg = '[{}] {}: duration (min) = {}'.format(begintime.strftime('%Y-%m-%d'), func.__name__, duration)
                print('<{}>'.format(comment))
                print(msg)
            return ret
        return wrapper
    return decorator


def gen_distribution(n):
    rand_nums = np.random.randint(0, 100, size=n)
    base = sum(rand_nums)*1.0
    return [x/base for x in rand_nums]


def draw_arg(probs):
    assert(abs(sum(probs) - 1.0) < 0.00000001)
    probs = np.array(probs)
    # Do a second normalisation to avoid the problem described here: https://stackoverflow.com/questions/46539431/np-random-choice-probabilities-do-not-sum-to-1
    return np.random.choice(list(range(len(probs))), p=probs/probs.sum())


def elem_distribution(arr):
    cnt = Counter(arr)
    _sum = sum(cnt.values())
    return {k: v / _sum for k, v in cnt.items()}

######################################
# High performance utility functions #
######################################s
@jit
def round(num, dec_places=2):
    return float('%.{}f'.format(dec_places) % num)


@jit
def rand(n=1, seed=None):
    if seed:
        np.random.seed(seed)
    return np.random.rand() * n


@jit
def rand_choice(candidates):
    return random.choice(candidates)


@jit
def randint(low, high, seed=None):
    if seed:
        np.random.seed(seed)
    return np.random.randint(low, high)


@jit(nopython=True)
def ucb(N_h, N_ha):
    if N_h == 0:
        return 0.0
    if N_ha == 0:
        return MAX
    return np.sqrt(np.log(N_h) / N_ha)  # Upper-Confidence-Bound

