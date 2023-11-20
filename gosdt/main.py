import pandas as pd
import numpy as np
import time as t
import pathlib
import argparse
from sklearn.ensemble import GradientBoostingClassifier
import numpy.random as random
from gosdt.model.threshold_guess import compute_thresholds
from gosdt.model.gosdt import GOSDT

SAMPLE_TYPES = ['sampling', 'deterministic', 'mathias', 'baseline']
WEIGHTING_TYPES = ['exponential']

# TODO:
# - change weights
# - make sure samplers are correct

def weighted_loss(model, X_train_dup, y_train_dup, X_train, y_train, weights):
    regularizer = model.tree.loss() - model.error(X_train_dup, y_train_dup)
    return model.error(X_train, y_train, weight=weights) + regularizer


# returns model
def preprocess_dataset(dataset):
    X, y = dataset.iloc[:,:-1].values, dataset.iloc[:,-1].values
    n_est = 40
    max_depth = 1

    # guess thresholds
    X = pd.DataFrame(X, columns=dataset.columns[:-1])
    # print("X:", X.shape)
    # print("y:",y.shape)
    X_train, thresholds, header, threshold_guess_time = compute_thresholds(X, y, n_est, max_depth)
    y_train = pd.DataFrame(y)

    # guess lower bound
    clf = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train.values.flatten())
    warm_labels = clf.predict(X_train)

    # save the labels as a tmp file and return the path to it.
    labelsdir = pathlib.Path('/tmp/warm_lb_labels')
    labelsdir.mkdir(exist_ok=True, parents=True)

    labelpath = labelsdir / 'warm_label.tmp'
    labelpath = str(labelpath)
    pd.DataFrame(warm_labels, columns=["class_labels"]).to_csv(labelpath, header="class_labels",index=None)

    # train GOSDT model
    config = {
                "regularization": 0.001,
                "depth_budget": 5,
                "time_limit": 60,
                "warm_LB": True,
                "path_to_labels": labelpath,
                "similar_support": False,
            }

    return GOSDT(config), pd.concat((X_train, y_train), axis=1)


def perform_tree_fitting(model, data_dup, data, weights):
    X_dup, y_dup = pd.DataFrame(data_dup.iloc[:,:-1].values), pd.DataFrame(data_dup.iloc[:,-1].values)
    X, y = pd.DataFrame(data.iloc[:,:-1].values), pd.DataFrame(data.iloc[:,-1].values)

    model.fit(X_dup, y_dup)

    print("evaluate the model, extracting tree and scores") 

    # get the results
    train_loss = weighted_loss(model, X_dup, y_dup, X, y, weights)

    print(f"Training loss: {train_loss}")
    return train_loss


def baseline(model, data, weights):
    return perform_tree_fitting(model, data, data, weights)


def gosdtDeterministic(model, data, weights, p):
    N = data.shape[0]
    dups = np.round(weights * N * p)
    duped_dataset = data.loc[data.index.repeat(dups)]
    # print(duped_dataset.shape[0], N * p)
    duped_dataset = duped_dataset.reset_index(drop=True)
    return perform_tree_fitting(model, duped_dataset, data, weights)


def gosdtSampling(model, data, weights, p):
    N = data.shape[0]
    sampled_data = data.sample(n=int(N * p), replace=True, weights=weights, ignore_index=True)
    return perform_tree_fitting(model, sampled_data, data, weights)


def mathiasSampling(model, data, weights, p):
    N = data.shape[0]
    deter_count = np.floor(weights * N * p) # determinisitc part of duplication
    # print("disc\n", deter_count[:5])
    # print("p\n", (weights*N*p - deter_count)[:5])
    stoch_count = (np.random.rand(weights.shape[0]) < (weights * N * p - deter_count)).astype(int) # stochastic part
    # print("stoch\n", stoch_count[:5])
    sampled_dups = deter_count + stoch_count # combine to get the samples that should be duplicated
    # print("dups\n", sampled_dups[:5])
    duped_dataset = data.loc[data.index.repeat(sampled_dups)]
    duped_dataset = duped_dataset.reset_index(drop=True)
    return perform_tree_fitting(model, duped_dataset, data, weights)


def sample_weights(dist, N, *kwargs):
    if dist == 'exponential':
        return random.exponential(scale=1/float(kwargs[0]), size=N)
    else:
        raise RuntimeError(f'Distribution of type {dist} cannot be handled')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_dist', type=str, choices=WEIGHTING_TYPES, help='Weighting distribution')
    parser.add_argument('--weight_args', nargs='*')
    parser.add_argument('--sampling_method', type=str, choices=SAMPLE_TYPES, help='Type of sampling method')
    parser.add_argument('-p', type=float, help='Dataset multiplier')
    parser.add_argument('--out', type=str, help='Where results are written to')
    args = parser.parse_args()

    # Load dataset
    data = pd.read_csv('datasets/fico.csv')
    N = data.shape[0]

    # Preporcess dataset and get model
    model, data = preprocess_dataset(data)

    # Sample weights from distribution
    weights = sample_weights(args.weight_dist, N, *args.weight_args)
    weights = weights / weights.sum() # Normalize weights
    
    # Dup dataset and fit model
    print(f'Weight distribution {args.weight_dist}({", ".join(map(str, args.weight_args))}), \tp={args.p}')
    accuracy, loss, time = 0, 0, 0
    if args.sampling_method == 'mathias':
        loss = mathiasSampling(model, data, weights, args.p)
    elif args.sampling_method == 'sampling':
        loss = gosdtSampling(model, data, weights, args.p)
    elif args.sampling_method == 'deterministic':
        loss = gosdtDeterministic(model, data, weights, args.p)
    elif args.sampling_method == 'baseline':
        loss = baseline(model, data, weights)
    else:
        raise RuntimeError(f'Sampling of type {args.sampling_method} cannot be handled')
    
    # Write to file
    if args.out is not None:
        import os.path
        add_header = not os.path.exists(args.out) 
        with open(args.out, 'a+') as file:
            if add_header:
                file.write('sampling_method,distribution,p,loss\n')
            file.write(f'{args.sampling_method}, {args.weight_dist}({",".join(map(str, args.weight_args))}), {args.p}, {loss}\n')
            file.close()
    
            





