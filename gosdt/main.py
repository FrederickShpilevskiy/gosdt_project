import pandas as pd
import numpy as np
import time as t
import pathlib
import argparse
from sklearn.ensemble import GradientBoostingClassifier
import numpy.random as random
from gosdt.model.threshold_guess import compute_thresholds
from gosdt.model.gosdt import GOSDT

SAMPLE_TYPES = ['sampling', 'deterministic', 'mathias']
WEIGHTING_TYPES = ['exponential']


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

    print("evaluate the model, extracting tree and scores") 

    # get the results
    train_acc = model.score(X_train, y_train)
    train_loss = model.tree.loss()
    n_leaves = model.leaves()
    n_nodes = model.nodes()
    time = model.utime

    print(f"Training accuracy: {train_acc}")
    print(f"Training loss: {train_loss}")
    print(f"Model training time: {time}")
    return train_acc, train_loss, time


def gosdtDeterministic(data, weights, p):
    N = data.shape[0]
    dups = np.round(weights * N * p)
    duped_dataset = data.loc[data.index.repeat(dups)]
    print(duped_dataset.shape[0], N * p)
    dataset = duped_dataset.reset_index(drop=True)
    return perform_tree_fitting(dataset)


def gosdtSampling(data, weights, p):
    N = data.shape[0]
    sampled_data = data.sample(n=int(N * p), replace=True, weights=weights, ignore_index=True)
    return perform_tree_fitting(sampled_data)


def mathiasSampling(data, weights, p):
    N = data.shape[0]
    deter_count = np.floor(weights * N * p) # determinisitc part of duplication
    # print("disc\n", deter_count[:5])
    # print("p\n", (weights*N*p - deter_count)[:5])
    stoch_count = (np.random.rand(weights.shape[0]) < (weights * N * p - deter_count)).astype(int) # stochastic part
    # print("stoch\n", stoch_count[:5])
    sampled_dups = deter_count + stoch_count # combine to get the samples that should be duplicated
    # print("dups\n", sampled_dups[:5])
    duped_dataset = data.loc[data.index.repeat(sampled_dups)]
    dataset = duped_dataset.reset_index(drop=True)
    return perform_tree_fitting(dataset)


def sample_weights(dist, N, *kwargs):
    if dist == 'exponential':
        return random.exponential(scale=float(kwargs[0]), size=N)
    else:
        raise RuntimeError(f'Distribution of type {dist} cannot be handled')


def single_point_mass(N, p, multiplier=1):
    # idea: let x^ be some randomly selected point from the dataset with some weight w^
    #       for every point x_i =/= x^ we have normalized weight wn_i * N * p < 0.5
    #       suppose all w_i have same the weight of 1, then the normalized weight becomes
    #       wn_i = 1/[(N-1)*w_i + w^] --> N*p / [(N-1) * w^] < 0.5 so 
    #       w^ > 2N(p-0.5) - 1
    # multiplier controls how much mass to allocate to the single point, should be >= 1
    assert(multiplier > 1)
    w_hat_idx = random.randint(0,N)
    weights = [1]*N
    weights[w_hat_idx] = (2*N*(p-0.5)+1)*multiplier
    
    T = sum(weights)
    for i in range(N):
        print(weights[i]/T)
        assert(weights[i]/T < 0.5 if i != w_hat_idx else weights[i]/T > 0.5)

    return weights

def k_outlier_points(N, p, k, base_weight=1):
    # idea: randomly select some k points to be the outliers each with weight w^=b and 
    #       normalized weight mu^

    indexs = list(range(N))
    random.shuffle(indexs)
    O = indexs[:k]
    weights = [base_weight]*N

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

    # Sample weights from distribution
    weights = sample_weights(args.weight_dist, N, *args.weight_args)
    # print(weights[:10])
    weights = weights / weights.sum() # Normalize weights
    # print(weights[:10])
    # raise RuntimeError('PAUSE')
    print(f'Weight distribution {args.weight_dist}({", ".join(map(str, args.weight_args))}), \tp={args.p}')
    accuracy, loss, time = 0, 0, 0
    if args.sampling_method == 'mathias':
        accuracy, loss, time = mathiasSampling(data, weights, args.p)
    elif args.sampling_method == 'sampling':
        accuracy, loss, time = gosdtSampling(data, weights, args.p)
    elif args.sampling_method == 'deterministic':
        accuracy, loss, time = gosdtDeterministic(data, weights, args.p)
    else:
        raise RuntimeError(f'Sampling of type {args.sampling_method} cannot be handled')
    
    # Write to file
    if args.out is not None:
        import os.path
        add_header = not os.path.exists(args.out) 
        with open(args.out, 'a+') as file:
            if add_header:
                file.write('sampling_method, distribution, p, accuracy, loss, time\n')
            file.write(f'{args.sampling_method}, {args.weight_dist}({", ".join(map(str, args.weight_args))}), {args.p}, {accuracy}, {loss}, {time}\n')
            file.close()
    
            





