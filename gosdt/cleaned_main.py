import numpy as np
import pandas as pd
import argparse
import random as rand

from WeightGenerator import sample_weights
from Experiment import run_experiment
from DataGenerator import generate_data

WEIGHTING_TYPES = ['exponential', 'binary', 'none', 'adversarial-single-point', 'adversarial-class-bias',\
                   'bias_one_class', 'bias_single_point']
DATA_DUPLICATION_TYPES = ['deterministic', 'sampling', 'mathias']
DATA_GENERATION_TYPES = ["xor", "lin_sep"]
EXPERIMENT_TYPE = ['gosdt', 'scikit', 'gosdt-fit-without-weights', 'scikit-fit-without-weights',\
                   'gosdt-bias-to-errors', 'scikit-bias-to-errors']


def collect_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='path to data as csv')
    parser.add_argument('--data_gen_type', type=str, choices=DATA_GENERATION_TYPES, help="Type of data generation")
    parser.add_argument('--data_gen_args', nargs='*', type=float, default=[])
    parser.add_argument('--seed', type=int, default=42, help='Seed for experiment')
    parser.add_argument('--weight_dist', type=str, choices=WEIGHTING_TYPES, help='Weighting distribution')
    parser.add_argument('--weight_args', nargs='*', type=float, default=[])

    # previously data_gen
    parser.add_argument('--data_dup', type=str, choices=DATA_DUPLICATION_TYPES, help='Type of method to apply to weights')
    parser.add_argument('--experiment', type=str, choices=EXPERIMENT_TYPE, help='Type of tree to fit')
    parser.add_argument('--exp_params', type=float, nargs="*", help="Hyper-param for experiment", default=[])
    parser.add_argument('--tree_depth', type=int, default=5, help="Max depth of trees in experiment")
    parser.add_argument('-p', type=float, help='Dataset multiplier')
    parser.add_argument('--out', type=str, help='Where results are written to')
    parser.add_argument('--logs', action='store_true', help="Toggle for logged messages")
    return parser.parse_args()

if __name__ == '__main__':
    args = collect_arguments()

    if args.file is not None and args.data_gen_type is not None:
        print("Cannot specify path to data and generate data in the same experiment")
        exit()
    
    if args.file is None and args.data_gen_type is None:
        print("Must specify one source of data (file or data gen type)")
        exit()

    if args.file is not None:
        df = pd.read_csv(args.file)
    else:
        df = generate_data(args.data_gen_type, *args.data_gen_args)

    np.random.seed(args.seed)
    rand.seed(args.seed)

    selected_label = rand.choice(list(df.iloc[:,-1].unique()))
    is_selected_label = df[df.iloc[:,-1] == selected_label]

    weights = sample_weights(args.weight_dist, df.shape[0], args.p, is_selected_label.index.values, *args.weight_args)
    weights = weights / weights.sum()

    run_experiment(args, df, weights)
        

