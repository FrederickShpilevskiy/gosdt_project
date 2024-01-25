import numpy as np
import pandas as pd
import argparse
import pathlib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from model.threshold_guess import compute_thresholds, cut
from gosdt.model.gosdt import GOSDT


WEIGHTING_TYPES = ['exponential', 'binary', 'none']
DATA_GENERATION_TYPES = ['deterministic', 'sampling']
EXPERIMENT_TYPE = ['gosdt', 'scikit', 'gosdt-fit-without-weights', 'scikit-fit-without-weights']

# hyperparams
TREE_DEPTH = 5

# y should be numpy 
# len(weights) == len(X) == len(y)
def weighted_loss(model, X, y, weights):
    preds = model.predict(X)
    loss = (y.reshape(-1) != preds).astype(float)
    return (loss * weights).sum()


def sample_weights(dist, N, *kwargs):
    if dist == 'exponential':
        return random.exponential(scale=1/float(kwargs[0]), size=N)
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


def generate_data(sampling_method, df, weights):
    data = df.copy()
    data_allowance = data.shape[0] * args.p

    if sampling_method == 'deterministic':
        dups = np.round(weights * data_allowance)
        duped_dataset = data.loc[data.index.repeat(dups)]
        return duped_dataset.reset_index(drop=True)

    if sampling_method == 'sampling':
        return data.sample(n=int(data_allowance), replace=True, weights=weights, ignore_index=True)
    
    else: 
        print("Sampling method does not exist")


def preprocess_dataset(dataset, n_est=40, max_depth=1):
    X, y = dataset.iloc[:,:-1].values, dataset.iloc[:,-1].values

    # guess thresholds
    X = pd.DataFrame(X, columns=dataset.columns[:-1])
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
                "depth_budget": TREE_DEPTH,
                "time_limit": 60,
                "warm_LB": True,
                "path_to_labels": labelpath,
                "similar_support": False,
            }
    return GOSDT(config), X_train, y_train, thresholds, header

def apply_thresholds(dataset, thresholds, header):
    X, y = dataset.iloc[:,:-1].values, dataset.iloc[:,-1].values
    X = pd.DataFrame(X, columns=dataset.columns[:-1])
    # apply thresholds
    X_new = cut(X, thresholds)
    return X_new[list(header)], pd.DataFrame(y)

def run_gosdt(args, df, weights):
    sampled_df = generate_data(args.data_gen, df, weights)
    model, X, y, thresholds, header = preprocess_dataset(sampled_df)

    model.fit(X, y)

     # not real "test" set, we are just interested in performance on all data
     # need to change the data to have the right features for classifying
    X_test, y_test = apply_thresholds(df, thresholds, header)
    return weighted_loss(model, X_test, y_test.to_numpy(), weights)

def run_scikit(args, df, weights):
    sampled_df = generate_data(args.data_gen, df, weights)
    X, y = sampled_df.iloc[:,:-1].values, sampled_df.iloc[:,-1].values

    clf = DecisionTreeClassifier(max_depth=TREE_DEPTH, random_state=42)
    clf.fit(X, y)

    # not real "test" set, we are just interested in performance on all data
    X_test = df.iloc[:,:-1].values
    y_test = df.iloc[:,-1].values
    return weighted_loss(clf, X_test, y_test, weights)
    

def run_gosdt_fit_without_weights(args, df, weights):
    model, X, y, thresholds, header = preprocess_dataset(df)

    model.fit(X, y)

     # not real "test" set, we are just interested in performance on all data
     # need to change the data to have the right features for classifying
    X_test, y_test = apply_thresholds(df, thresholds, header)
    return weighted_loss(model, X_test, y_test.to_numpy(), weights)


def run_scikit_fit_without_weights(args, df, weights):
    X, y = df.iloc[:,:-1].values, df.iloc[:,-1].values
    clf = DecisionTreeClassifier(max_depth=TREE_DEPTH, random_state=42)
    clf.fit(X,y)

    # not real "test" set, we are just interested in performance on all data
    X_test = df.iloc[:,:-1].values
    y_test = df.iloc[:,-1].values
    return weighted_loss(clf, X_test, y_test, weights)


def collect_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to data as csv')
    parser.add_argument('--weight_dist', type=str, choices=WEIGHTING_TYPES, help='Weighting distribution')
    parser.add_argument('--weight_args', nargs='*', type=float)
    parser.add_argument('--data_gen', type=str, choices=DATA_GENERATION_TYPES, help='Type of method to apply to weights')
    parser.add_argument('--experiment', type=str, choices=EXPERIMENT_TYPE, help='Type of tree to fit')
    parser.add_argument('-p', type=float, help='Dataset multiplier')
    parser.add_argument('--out', type=str, help='Where results are written to')
    parser.add_argument('--logs', action='store_true', help="Toggle for logged messages")
    return parser.parse_args()

if __name__ == '__main__':
    np.random.seed(42)

    args = collect_arguments()
    df = pd.read_csv(args.path)

    weights = sample_weights(args.weight_dist, df.shape[0], *args.weight_args)
    weights = weights / weights.sum()

    if args.experiment == "gosdt":
        loss = run_gosdt(args, df, weights)
    elif args.experiment == "gosdt-fit-without-weights":
        loss = run_gosdt_fit_without_weights(args, df, weights)
    elif args.experiment == "scikit":
        loss = run_scikit(args, df, weights)
        if args.logs:
            print(loss)
    elif args.experiment == "scikit-fit-without-weights":
        loss = run_scikit_fit_without_weights(args, df, weights)
        if args.logs:
            print(loss)
    else:
        print("Error: invalid experiment type")
        exit()

    if args.out is not None:
        import os.path
        add_header = not os.path.exists(args.out)
        with open(args.out, 'a+') as file:
            if add_header:
                file.write('sampling_method,distribution,p,experiment,loss\n')
            file.write(f'{args.data_gen}, {args.weight_dist}({"-".join(map(str, args.weight_args))}), {args.p}, {args.experiment}, {loss}\n')
            file.close()
        

