import numpy as np
import math
from collections import defaultdict


#quantile estimation functions       
def unboundedQuantile(data, l, q, eps, b=1.001, noise = 'exponential'):
    eps_1 = eps/2
    eps_2 = eps/2
    d = defaultdict(int)
    for x in data:
        i = math.log(x-l+1,b) // 1
        d[i] += 1

    if noise == 'exponential':
        t = q * len(data) + np.random.exponential(1/eps_1)
    elif noise == 'laplace':
        t = q * len(data) + np.random.laplace(loc=0, scale=1, size=1)/eps_1

    cur, i = 0, 0
    while True:
        cur += d[i]
        i += 1
        if noise == 'exponential':
            if cur + np.random.exponential(1/eps_2) > t:
                break
        elif noise == 'laplace':
            if cur + (np.random.laplace(loc=0, scale=1, size=1)/eps_2) > t:
                break
    return b**i - 1 + l