import numpy as np
import pandas as pd
import argparse
import random as rand

from WeightGenerator import sample_weights
from Experiment import run_experiment

WEIGHTING_TYPES = ['exponential', 'binary', 'none', 'adversarial-single-point', 'adversarial-class-bias']
DATA_GENERATION_TYPES = ['deterministic', 'sampling', 'mathias']
EXPERIMENT_TYPE = ['gosdt', 'scikit', 'gosdt-fit-without-weights', 'scikit-fit-without-weights']


def collect_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to data as csv')
    parser.add_argument('--seed', type=int, default=42, help='Seed for experiment')
    parser.add_argument('--weight_dist', type=str, choices=WEIGHTING_TYPES, help='Weighting distribution')
    parser.add_argument('--weight_args', nargs='*', type=float)
    parser.add_argument('--data_gen', type=str, choices=DATA_GENERATION_TYPES, help='Type of method to apply to weights')
    parser.add_argument('--experiment', type=str, choices=EXPERIMENT_TYPE, help='Type of tree to fit')
    parser.add_argument('--tree_depth', type=int, default=5, help="Max depth of trees in experiment")
    parser.add_argument('-p', type=float, help='Dataset multiplier')
    parser.add_argument('--out', type=str, help='Where results are written to')
    parser.add_argument('--logs', action='store_true', help="Toggle for logged messages")
    return parser.parse_args()

if __name__ == '__main__':
    args = collect_arguments()

    df = pd.read_csv(args.path)
    np.random.seed(args.seed)
    rand.seed(args.seed)

    selected_label = rand.choice(list(df.iloc[:,-1].unique()))
    is_selected_label = df[df.iloc[:,-1] == selected_label]

    weights = sample_weights(args.weight_dist, df.shape[0], args.p, is_selected_label.index.values, *args.weight_args)
    weights = weights / weights.sum()

    loss = run_experiment(args, df, weights)

    if args.out is not None:
        import os.path
        add_header = not os.path.exists(args.out)
        with open(args.out, 'a+') as file:
            if add_header:
                file.write('sampling_method,distribution,p,experiment,tree_depth,loss\n')
            file.write(f'{args.data_gen}, {args.weight_dist}({"-".join(map(str, args.weight_args))}), {args.p}, {args.experiment}, {args.tree_depth}, {loss}\n')
            file.close()
        

