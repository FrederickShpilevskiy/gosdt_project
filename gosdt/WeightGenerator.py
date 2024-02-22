import numpy as np
import pandas as pd
import random as rand


def adversarial_single_point(N, p, a, k):
    weights = np.ones(N)
    point = rand.randrange(N)
    weights[point] = k*(2*N*p - a*N + a)
    return weights

# percent_keep = % of samples from other class label to retain after weighted duplication
# to keep points, make them have 1.25 * weight of non-kept points
def adversarial_class_bias(N, is_selected, percent_keep):
    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[is_selected] = True 

    other_label_N = N - is_selected.size
    false_idxs = np.where(selected_mask == False)[0]
    flip_idxs = np.random.choice(false_idxs, size=int(other_label_N*percent_keep), replace=False)
    selected_mask[flip_idxs] = True

    weights = np.ones(N)
    weights[selected_mask] = 1.25
    return weights

# is selected used if you want to sample based on a particular label's entries, see main for selection
def sample_weights(dist, N, p, is_selected, *kwargs):
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
    if dist == 'adversarial-single-point':
        return adversarial_single_point(N, p, kwargs[0], kwargs[1])
    if dist == "adversarial-class-bias":
        return adversarial_class_bias(N, is_selected, kwargs[0])
    if dist == 'bias_one_class':
        weights = np.ones(N)
        weights[is_selected] = kwargs[0]
        weights[~is_selected] = kwargs[1]
        return weights
    else:
        raise RuntimeError(f'Distribution of type {dist} cannot be handled')