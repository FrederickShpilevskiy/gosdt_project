import numpy as np
import pandas as pd 
import pathlib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from gosdt import ThresholdGuessBinarizer, GOSDTClassifier
from DataSampler import generate_data

# hyperparams
GBDT_N_EST = 40
GBDT_MAX_DEPTH = 1
REGULARIZATION = 0.001
SIMILAR_SUPPORT = False
DEPTH_BUDGET = None
TIME_LIMIT = 60
VERBOSE = False 

# y should be numpy 
# len(weights) == len(X) == len(y)
def weighted_loss(model, X, y, weights):
    preds = model.predict(X)
    loss = (y.reshape(-1) != preds).astype(float)
    return (loss * weights).sum()

def gosdt_experiment(args, df, weights, should_dup=False):
    if args.logs:
        print("Gosdt Experiment")

    if should_dup:
        sampled_df = generate_data(args, df, weights)
    else:
        sampled_df = df.copy()
    
    X, y = sampled_df.iloc[:, :-1], sampled_df.iloc[:, -1]

    if args.logs:
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

    threshold_enc = ThresholdGuessBinarizer(n_estimators=GBDT_N_EST, max_depth=GBDT_MAX_DEPTH, random_state=args.seed)
    threshold_enc.set_output(transform="pandas")
    X_guessed = threshold_enc.fit_transform(X, y)

    clf = GOSDTClassifier(regularization=REGULARIZATION, similar_support=SIMILAR_SUPPORT, time_limit=TIME_LIMIT, depth_budget=DEPTH_BUDGET, verbose=VERBOSE) 
    clf.fit(X_guessed, y)
    
    wrong = clf.predict(X_guessed) != y.to_numpy().reshape(-1) 

    # Not a real test set, but need to apply thresholds to original data
    X_test, y_test = df.iloc[:, :-1], df.iloc[:, -1]
    X_test_enc = threshold_enc.transform(X_test)

    return weighted_loss(clf, X_test_enc, y_test.to_numpy(), weights), wrong 


def scikit_experiment(args, df, weights, should_dup=False):
    if args.logs:
        print("Scikit Experiment")

    if should_dup:
        sampled_df = generate_data(args, df, weights)
    else:
        sampled_df = df.copy()

    X, y = sampled_df.iloc[:,:-1].values, sampled_df.iloc[:,-1].values
    clf = DecisionTreeClassifier(max_depth=DEPTH_BUDGET, random_state=args.seed)
    clf.fit(X,y)

    # not real "test" set, we are just interested in performance on all data
    X_test = df.iloc[:,:-1].values
    y_test = df.iloc[:,-1].values
    wrong = clf.predict(X_test) != y_test.reshape(-1)
    return weighted_loss(clf, X_test, y_test, weights), wrong

def gosdt_bias_experiment(args, df, weights):
    # weights don't matter, create weights based on errors in unweighted tree
    if args.logs:
        print("Performing fit without weights")

    init_loss, wrong = gosdt_experiment(args, df, weights)
    N = df.shape[0]

    if args.logs:
        print(f"Got {np.sum(wrong)} wrong out of {N} points")

    bias_value = args.exp_params[0]
    bias_weights = np.ones(N)
    bias_weights[wrong] = bias_value
    bias_weights = bias_weights / np.sum(bias_weights)
    if args.logs:
        print("Made weights")
        print(bias_weights[:10])

    weighted_tree_loss, _ = gosdt_experiment(args, df, bias_weights, should_dup=True)
    unweighted_tree_loss, _ = gosdt_experiment(args, df, bias_weights)
    if args.logs:
        print("done weighted tree")
    save_results(args, unweighted_tree_loss, override_experiment="unweighted_tree")
    save_results(args, weighted_tree_loss, override_experiment="weighted_tree")

def scikit_bias_experiment(args, df, weights):
    # weights don't matter, create weights based on errors in unweighted tree 
    _, wrong = scikit_experiment(args, df, weights)
    
    N = df.shape[0]
    bias_value = args.exp_params[0]
    bias_weights = np.ones(N)
    bias_weights[wrong] = bias_value
    bias_weights = bias_weights / np.sum(bias_weights)

    weighted_tree_loss, _ = scikit_experiment(args, df, bias_weights, should_dup=True)
    unweighted_tree_loss, _ = scikit_experiment(args, df, bias_weights)
    save_results(args, unweighted_tree_loss, override_experiment="unweighted_tree")
    save_results(args, weighted_tree_loss, override_experiment="weighted_tree")


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
                file.write('seed,sampling_method,data_gen,distribution,p,experiment,exp_params,tree_depth,loss\n')
            file.write(f'{args.seed}, {args.data_dup}, {data_source}, {args.weight_dist}({"-".join(map(str, args.weight_args))}), {args.p}, {args.experiment if override_experiment is None else override_experiment}, {args.exp_params}, {args.tree_depth}, {loss_arg}\n')
    
    else:
        print(f"loss: {loss_arg}")

def run_experiment(args, df, weights):
    global DEPTH_BUDGET
    DEPTH_BUDGET = args.tree_depth
    np.random.seed(args.seed)

    experiment = args.experiment
    if experiment == "gosdt":
        loss, _ = gosdt_experiment(args, df, weights, should_dup=True)
        save_results(args, loss)
    elif experiment == "gosdt-fit-without-weights":
        loss, _ = gosdt_experiment(args, df, weights)
        save_results(args, loss)
    elif experiment == "scikit":
        loss, _ = scikit_experiment(args, df, weights, should_dup=True)
        save_results(args, loss)
    elif experiment == "scikit-fit-without-weights":
        loss, _ = scikit_experiment(args, df, weights)
        save_results(args, loss)
    elif experiment == 'gosdt-bias-to-errors':
        gosdt_bias_experiment(args, df, weights)
    elif experiment == 'scikit-bias-to-errors':
        scikit_bias_experiment(args, df, weights)


