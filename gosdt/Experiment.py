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

def gosdt_experiment(args, df, weights, should_dup=False, plot_loc=None):
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

    if plot_loc is not None:
        plot_2d_separator(args, clf, X_test.to_numpy(), y_test.to_numpy(), plot_loc, sampled_df, encoder=threshold_enc) 

    return weighted_loss(clf, X_test_enc, y_test.to_numpy(), weights), wrong 


def scikit_experiment(args, df, weights, should_dup=False, plot_loc=None):
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
    X_test = df.iloc[:,:-1].to_numpy()
    y_test = df.iloc[:,-1].to_numpy()
    wrong = clf.predict(X_test) != y_test.reshape(-1)

    if plot_loc is not None:
        plot_2d_separator(args, clf, X_test, y_test, plot_loc, sampled_df) 

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

    # Just make plot for the weighted tree
    weighted_tree_loss, _ = gosdt_experiment(args, df, bias_weights, should_dup=True, plot_loc=args.plot_loc)
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

    # Just make plot for weighted tree
    weighted_tree_loss, _ = scikit_experiment(args, df, bias_weights, should_dup=True, plot_loc=args.plot_loc)
    unweighted_tree_loss, _ = scikit_experiment(args, df, bias_weights)
    save_results(args, unweighted_tree_loss, override_experiment="unweighted_tree")
    save_results(args, weighted_tree_loss, override_experiment="weighted_tree")


def save_results(args, loss_arg, override_experiment=None):
    if args.file is not None:
        data_source = args.file
        data_gen_params = list()
    else:
        data_source = args.data_gen_type
        data_gen_params = args.data_gen_args
    data_gen_params = str(data_gen_params).replace(", ", "-")

    if override_experiment is None:
        experiment = args.experiment
    else:
        experiment = override_experiment

    weight_args = str(args.weight_args).replace(", ", "-")
    exp_args = str(args.exp_params).replace(", ", "-")

    if args.out is not None:
        import os.path
        add_header = not os.path.exists(args.out)
        with open(args.out, 'a+') as file:
            if add_header:
                file.write('seed,sampling_method,data_source,data_args,distribution,dist_args,p,experiment,exp_params,tree_depth,loss\n')
            file.write(f'{args.seed}, {args.data_dup}, {data_source}, {data_gen_params}, {args.weight_dist}, {weight_args}, {args.p}, {experiment}, {exp_args}, {args.tree_depth}, {loss_arg}\n')
    
    else:
        print(f"loss: {loss_arg}")

# Adapted from UBC CPSC 330 Course
# GOSDT transforms the features so we need a copy of the encoder too
def plot_2d_separator(args, clf, X, y, plot_loc, sampled_df, encoder=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.colors as mcolors

    ax = plt.gca()
    eps = X.std()/2

    x_min, x_max = X[:, 0].min() - eps, X[:, 0].max() + eps
    y_min, y_max = X[:, 1].min() - eps, X[:, 1].max() + eps
    xx = np.linspace(x_min, x_max, 1000)
    yy = np.linspace(y_min, y_max, 1000)

    X1, X2 = np.meshgrid(xx, yy)
    X_grid = np.c_[X1.ravel(), X2.ravel()]

    cmap_scatter = sns.color_palette("hls", len(np.unique(y))) # "hsv" is just an example, you can use other categorical color palettes like "Set2", "tab10" etc.
    sns.set_palette(cmap_scatter)

    # Define a colormap for contourf plot
    cont_colors = sns.color_palette("husl", len(np.unique(y)))
    cmap_contourf = mcolors.ListedColormap(cont_colors.as_hex())

    if encoder is not None:
        X_grid_enc = encoder.transform(X_grid)
        y_hat = clf.predict(X_grid_enc)
    else:
        y_hat = clf.predict(X_grid)

    contr = y_hat.reshape(X1.shape)
    plt.contourf(xx, yy, contr, cmap=cmap_contourf)

    df = pd.DataFrame(data=np.c_[X, y], columns=["x1", "x2", "y"])

    if args.plot_type == "undup":
        sns.scatterplot(data=df, x="x1", y="x2", hue=y, ax=ax, edgecolor='black')
    elif args.plot_type == "resamp":
        X, y = sampled_df.iloc[:, :-1], sampled_df.iloc[:, -1]
        sampled_df = pd.DataFrame(data=np.c_[X,y], columns=["x1", "x2", "y"])
        sns.scatterplot(data=sampled_df, x="x1", y="x2", hue=y, ax=ax, edgecolor='black')
    else:
        print("Must specify plot_type and plot_loc")
        exit()
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.linspace(x_min, x_max, num=10))
    ax.set_yticks(np.linspace(y_min, y_max, num=10))
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")

    title = ""
    if args.file is not None:
        title += args.file
    else:
        title += f"{args.data_gen_type}({args.data_gen_args})"
    
    title += f" {args.experiment} using {args.data_dup}"

    ax.set_title(title)

    plt.tight_layout()

    plt.savefig(plot_loc)

def run_experiment(args, df, weights):
    global DEPTH_BUDGET
    DEPTH_BUDGET = args.tree_depth
    np.random.seed(args.seed)

    experiment = args.experiment
    if experiment == "gosdt":
        loss, _ = gosdt_experiment(args, df, weights, should_dup=True, plot_loc=args.plot_loc)
        save_results(args, loss)
    elif experiment == "gosdt-fit-without-weights":
        loss, _ = gosdt_experiment(args, df, weights, plot_loc=args.plot_loc)
        save_results(args, loss)
    elif experiment == "scikit":
        loss, _ = scikit_experiment(args, df, weights, should_dup=True, plot_loc=args.plot_loc)
        save_results(args, loss)
    elif experiment == "scikit-fit-without-weights":
        loss, _ = scikit_experiment(args, df, weights, plot_loc=args.plot_loc)
        save_results(args, loss)
    elif experiment == 'gosdt-bias-to-errors':
        gosdt_bias_experiment(args, df, weights)
    elif experiment == 'scikit-bias-to-errors':
        scikit_bias_experiment(args, df, weights)


