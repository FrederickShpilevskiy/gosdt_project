import numpy as np
import pandas as pd



def sample_weights(dist, N, *kwargs):
    if dist == 'exponential':
        return np.random.exponential(scale=1/float(kwargs[0]), size=N)
    if dist == 'binary':
        weights = np.ones(N)
        weights[:int(float(kwargs[0])*N)] = float(kwargs[1])
        weights[int(float(kwargs[0])*N):] = float(kwargs[2])
        np.random.shuffle(weights)
        return weights
    if dist == 'none':
        return np.ones(N)
    else:
        raise RuntimeError(f'Distribution of type {dist} cannot be handled')