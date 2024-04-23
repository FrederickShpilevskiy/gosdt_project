import pandas as pd
import numpy as np
import argparse

from ast import literal_eval


def dash_list_to_py_list(x):
   replaced = x.replace("-", ", ")
   return literal_eval(replaced)

def py_list_to_dashes(x):
    return str(x).replace(", ", "-")

def extract_list_elemement(i):
    def get_item(x):
        index = i
        return x[index]
    
    return get_item

def extract_data_summary(row):
    data_type = row["data_source"]
    return f"{data_type}({", ".join(map(str, row["data_args"]))})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str, help='Location of results file')
    parser.add_argument("--out", type=str, help="Location of output", default="eda.txt")
    parser.add_argument("--to_list", action='store_true', help="convert string lists to python lists")
    parser.add_argument("--get_only_d", action='store_true', help="just extract the dimension")
    args = parser.parse_args()
    # Read data
    df = pd.read_csv(args.in_path, index_col=None)
    df["experiment"] = df['experiment'].str.strip()
    df["sampling_method"] = df['sampling_method'].str.strip()
    df["data_source"] = df["data_source"].str.strip()
    df["p"] = pd.to_numeric(df["p"]) 
    df["loss"] = pd.to_numeric(df["loss"]) 

    # messes with group with so don't do this too much
    if args.to_list:
        df["data_args"] = df["data_args"].apply(dash_list_to_py_list)
        df["dist_args"] = df["dist_args"].apply(dash_list_to_py_list)
        df["exp_params"] = df["exp_params"].apply(dash_list_to_py_list)
        df["N"] = df["data_args"].apply(extract_list_elemement(0))
        df["d"] = df["data_args"].apply(extract_list_elemement(1))
        df["class_weight"] = df["dist_args"].apply(extract_list_elemement(0))
        df["mistake_weight"] = df["exp_params"].apply(extract_list_elemement(0))
        df["d"] = df["data_args"].apply(extract_list_elemement(1))

    if args.get_only_d:
        df["d"] = df["data_args"].apply(dash_list_to_py_list).apply(extract_list_elemement(1))

    
    # gosdt_df = df[(df["experiment"] == "gosdt") & (df["d"] == 2)]
    # weighted_tree_df = df[(df["experiment"] == "weighted_tree") & (df["d"] == 2)]

    # grp = set(gosdt_df.columns) - {'seed', 'loss'}
    # print("Bias to mistakes")
    # weighted_means_agg_seed = weighted_tree_df.groupby(by=list(grp)).agg({"loss": "mean"}).reset_index()
    # print(weighted_means_agg_seed.sort_values('loss').iloc[0])
    # print("Gosdt")
    # gosdt_means_agg_seed = gosdt_df.groupby(by=list(grp)).agg({"loss": "mean"}).reset_index()

    ps = df["p"].unique()
    df = df[df["experiment"] != "gosdt"]
    print(df["N"].unique())
    print("p  & mathias & sampling")
    for p_val in ps:
        N = 300
        sub_df = df[(df["sampling_method"] == "sampling") & (df["p"] == p_val) &\
                    (df["N"]==N) & (df["d"] == 2) & (df["mistake_weight"] == 3)]
        sampling_var = sub_df["loss"].var()
        sub_df = df[(df["sampling_method"] == "mathias") & (df["p"] == p_val) &\
                    (df["N"]==N) & (df["d"] == 2) & (df["mistake_weight"] == 3)]
        mathias_var = sub_df["loss"].var()

        print(f"{p_val:.1f} & {mathias_var:.3f} & {sampling_var:.3f}")



