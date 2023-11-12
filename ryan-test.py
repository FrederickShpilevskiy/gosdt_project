import pandas as pd
import numpy as np
import time as t
import pathlib
import random
import argparse
from sklearn.ensemble import GradientBoostingClassifier
from model.threshold_guess import compute_thresholds
from model.gosdt import GOSDT

SAMPLE_TYPES = ["gosdtwG", "mathias"]

def perform_tree_fitting(df):
    X, y = df.iloc[:,:-1].values, df.iloc[:,-1].values
    h = df.columns[:-1]
    n_est = 40
    max_depth = 1

    # guess thresholds
    X = pd.DataFrame(X, columns=h)
    # print("X:", X.shape)
    # print("y:",y.shape)
    X_train, thresholds, header, threshold_guess_time = compute_thresholds(X, y, n_est, max_depth)
    y_train = pd.DataFrame(y)

    # guess lower bound
    start_time = t.perf_counter()
    clf = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train.values.flatten())
    warm_labels = clf.predict(X_train)

    elapsed_time = t.perf_counter() - start_time

    lb_time = elapsed_time

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

    model = GOSDT(config)

    model.fit(X_train, y_train)

    print("evaluate the model, extracting tree and scores", flush=True) 

    # get the results
    train_acc = model.score(X_train, y_train)
    n_leaves = model.leaves()
    n_nodes = model.nodes()
    time = model.utime

    print("Model training time: {}".format(time))
    print("Training accuracy: {}".format(train_acc))
    # print("# of leaves: {}".format(n_leaves))
    # print(model.tree)

def GOSDTwG_sampling(data, weights, r):
    S = r * data.shape[0]
    # pandas sample w weight
    sampled_data = data.sample(n=int(S), replace=True, weights=weights, ignore_index=True)
    perform_tree_fitting(sampled_data)


def MathiasSampling(data, weights, p):
    N = data.shape[0]
    dup_counts = [int(w * N * p) for w in weights] # "duplicate" each row i, dup_counts[i] times

    sampled_dups = []
    for i, w in enumerate(weights):
        # sample each point with prob w*N*p - floor(w*N*p)
        num_duplicates_kept = 0
        prob = w*N*p - int(w*N*p)
        for _ in range(dup_counts[i]):
            num_duplicates_kept += 1 if random.random() < prob else 0

        sampled_dups.append(num_duplicates_kept)

    duped_dataset = data.loc[data.index.repeat(sampled_dups)]
    dataset = duped_dataset.reset_index(drop=True)
    perform_tree_fitting(dataset)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", type=float, nargs='+', help="% of weights to double")
    parser.add_argument("--sampling_method", type=str, help="Type of sampling method")
    parser.add_argument("-r", type=float, help="hyperparam for gosdtwG sampling")
    parser.add_argument("-p", type=float, help="hyperparam for mathias sampling")
    args = parser.parse_args()

    data = pd.read_csv("experiments/datasets/fico.csv")
    h = data.columns
    data = pd.DataFrame(data, columns=h)
    # print(data.head())
    # print("data:", data.shape)
    if args.sampling_method not in SAMPLE_TYPES:
        print(f"Sampling method must be one of: {SAMPLE_TYPES}")

    # From the paper, they introduced weights by randomly selected q% of the data points and doubled their weights
    q_values = args.q
    N = data.shape[0]
    for q in q_values:
        weights = [1] * N # initialize weights to 1 for each data point

        # randomly double weights N*q times
        for i in range(int(N * q)):
            idx = random.randrange(N)
            weights[idx] *= 2
        
        # normalize weights
        w_total = sum(weights)
        weights = [w/w_total for w in weights]
        if args.sampling_method == "mathias":
            print(f"--- Sampling q={q}, \tp={args.p} ---")
            MathiasSampling(data, weights, args.p)
        elif args.sampling_method == "gosdtwG":
            print(f"--- Sampling q={q}, \tr={args.r} ---")
            GOSDTwG_sampling(data, weights, args.r)
        else:
            print("Invalid sampling method")
            exit()
            





