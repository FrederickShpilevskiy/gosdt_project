import numpy as np
import pandas as pd 
import pathlib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from model.threshold_guess import compute_thresholds, cut
from gosdt.model.gosdt import GOSDT
from DataSampler import generate_data

# hyperparams
TREE_DEPTH = None

# y should be numpy 
# len(weights) == len(X) == len(y)
def weighted_loss(model, X, y, weights):
    preds = model.predict(X)
    loss = (y.reshape(-1) != preds).astype(float)
    return (loss * weights).sum()

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
    sampled_df = generate_data(args, df, weights)
    model, X, y, thresholds, header = preprocess_dataset(sampled_df)

    if args.logs:
        print(df.shape)
        print(X.shape)

    print(X.head())
    model.fit(X, y)
    
     # not real "test" set, we are just interested in performance on all data
     # need to change the data to have the right features for classifying
    X_test, y_test = apply_thresholds(df, thresholds, header)
    return weighted_loss(model, X_test, y_test.to_numpy(), weights)

def run_scikit(args, df, weights):
    sampled_df = generate_data(args, df, weights)
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
    wrong = model.predict(X_test) != y_test.to_numpy().reshape(-1)
    return weighted_loss(model, X_test, y_test.to_numpy(), weights), wrong


def run_scikit_fit_without_weights(args, df, weights):
    X, y = df.iloc[:,:-1].values, df.iloc[:,-1].values
    clf = DecisionTreeClassifier(max_depth=TREE_DEPTH, random_state=42)
    clf.fit(X,y)

    # not real "test" set, we are just interested in performance on all data
    X_test = df.iloc[:,:-1].values
    y_test = df.iloc[:,-1].values
    wrong = clf.predict(X_test) != y_test.reshape(-1)
    return weighted_loss(clf, X_test, y_test, weights), wrong

def save_results(args, loss_arg, override_experiment=None):
    if args.file is not None:
        data_source = args.file
    else:
        data_source = f"{args.data_gen_type}({'-'.join(map(str, args.data_gen_args))})"

    if args.out is not None:
        import os.path
        add_header = not os.path.exists(args.out)
        with open(args.out, 'a+') as file:
            if add_header:
                file.write('seed,sampling_method,data_gen,distribution,p,experiment,tree_depth,loss\n')
            file.write(f'{args.seed}, {args.data_dup}, {data_source}, {args.weight_dist}({"-".join(map(str, args.weight_args))}), {args.p}, {args.experiment if override_experiment is None else override_experiment}, {args.tree_depth}, {loss_arg}\n')

def run_experiment(args, df, weights):
    global TREE_DEPTH
    TREE_DEPTH = args.tree_depth

    # Set up the print stuff and 

    experiment = args.experiment
    if experiment == "gosdt":
        loss = run_gosdt(args, df, weights)
        save_results(loss)
    elif experiment == "gosdt-fit-without-weights":
        loss, _ = run_gosdt_fit_without_weights(args, df, weights)
        save_results(loss)
    elif experiment == "scikit":
        loss = run_scikit(args, df, weights)
        save_results(loss)
    elif experiment == "scikit-fit-without-weights":
        loss = run_scikit_fit_without_weights(args, df, weights)
        save_results(loss)
    elif experiment == 'gosdt-bias-to-errors':
       # weights don't matter, create weights based on errors in unweighted tree
       init_loss, wrong = run_gosdt_fit_without_weights(args, df, weights)

       N = df.shape[0]
       bias_value = args.exp_params[0]
       bias_weights = np.ones(N)
       bias_weights[wrong] = bias_value
       bias_weights = bias_weights / np.sum(bias_weights)

       print("start")
       unweighted_tree_loss, _ = run_gosdt_fit_without_weights(args, df, bias_weights)
       print("after refit unweighted")
       weighted_tree_loss = run_gosdt(args, df, bias_weights)
       print("after weighted")
       save_results(args, unweighted_tree_loss, override_experiment="unweighted_tree")
       save_results(args, weighted_tree_loss, override_experiment="weighted_tree")

       if args.logs:
        print(f"init: {init_loss}\t after:{unweighted_tree_loss}\t weighted:{weighted_tree_loss}")

    elif experiment == 'scikit-bias-to-errors':
        # weights don't matter, create weights based on errors in unweighted tree 
        _, wrong = run_scikit_fit_without_weights(args, df, weights)
        
        N = df.shape[0]
        bias_value = args.exp_params[0]
        bias_weights = np.ones(N)
        bias_weights[wrong] = bias_value
        bias_weights = bias_weights / np.sum(bias_weights)

        unweighted_tree_loss, _ = run_scikit_fit_without_weights(args, df, bias_weights)
        weighted_tree_loss = run_scikit(args, df, bias_weights)
        save_results(args, unweighted_tree_loss, override_experiment="unweighted_tree")
        save_results(args, weighted_tree_loss, override_experiment="weighted_tree")

