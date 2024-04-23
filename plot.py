import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ast import literal_eval

N = 10459

# takes strings from " binary(0.25-2.0-1.0)" to "binary(0.25)" if last selected is 0, "binary(2.0)" if selector is 1... 
def extract_dist_summary(row):
    param_selector = 0
    dist_string = row["distribution"]
    dist_type = dist_string.split('(')[0]
    extracted_str = dist_string.split('(')[1].split('-')[param_selector]
    if extracted_str[-1] == ")":
        extracted_str = extracted_str[:-1]
    return dist_type + "(" + extracted_str + ")"

def extract_multiple_dist_params(row):
    dist_string = row["distribution"]
    dist_type = dist_string.split('(')[0]
    params = dist_string.split('(')[1].split('-')
    return f"{dist_type}({params[0]}, {params[1]})"

def extract_dist_params_first_num_only(row):
    param_selector = 0
    dist_string = row["distribution"]
    dist_type = dist_string.split('(')[0]
    extracted_str = dist_string.split('(')[1].split('-')[param_selector]
    if extracted_str[-1] == ")":
        extracted_str = extracted_str[:-1]

    return extracted_str

def extract_dist_params_second_num_only(row):
    param_selector = 1
    dist_string = row["distribution"]
    dist_type = dist_string.split('(')[0]
    extracted_str = dist_string.split('(')[1].split('-')[param_selector]
    if extracted_str[-1] == ")":
        extracted_str = extracted_str[:-1]

    return extracted_str

# takes strings from " binary(0.25-2.0-1.0)" to "binary(0.25)" if last selected is 0, "binary(2.0)" if selector is 1... 
# def extract_data_summary(row):
#     param_selector = 0
#     dist_string = row["data_gen"]
#     dist_type = dist_string.split('(')[0]
#     extracted_str = dist_string.split('(')[1].split('-')[param_selector]
#     if extracted_str[-1] == ")":
#         extracted_str = extracted_str[:-1]
#     return dist_type + "(" + extracted_str + ")"

# For new formatted params
def extract_data_summary(row):
    param_selector = 0
    dist_string = row["data_gen"]
    dist_type = dist_string.split('(')[0]
    extracted_str = dist_string.split('(')[1].split('-')[param_selector]
    if extracted_str[-1] == ")":
        extracted_str = extracted_str[:-1]
    return dist_type + "(" + extracted_str + ")"

def extract_data_gen_params_first_num_only(row):
    param_selector = 0
    dist_string = row["data_gen"]
    dist_type = dist_string.split('(')[0]
    extracted_str = dist_string.split('(')[1].split('-')[param_selector]
    if extracted_str[-1] == ")":
        extracted_str = extracted_str[:-1]

    return extracted_str

def extract_data_gen_params_second_num_only(row):
    param_selector = 1
    dist_string = row["data_gen"]
    dist_type = dist_string.split('(')[0]
    extracted_str = dist_string.split('(')[1].split('-')[param_selector]
    if extracted_str[-1] == ")":
        extracted_str = extracted_str[:-1]

    return extracted_str

def extract_data_gen_params_third_num_only(row):
    param_selector = 2
    dist_string = row["data_gen"]
    dist_type = dist_string.split('(')[0]
    extracted_str = dist_string.split('(')[1].split('-')[param_selector]
    if extracted_str[-1] == ")":
        extracted_str = extracted_str[:-1]

    return extracted_str

def extract_data_gen_params_i(i):
    def extract_fn(row):
        param_selector = i
        dist_string = row["data_gen"]
        extracted_str = dist_string.split('(')[1].split('-')[param_selector]
        if extracted_str[-1] == ")":
            extracted_str = extracted_str[:-1]

        return extracted_str

    return extract_fn

def dash_list_to_py_list(x):
   replaced = x.replace("-", ", ")
   return literal_eval(replaced)

def extract_list_elemement(i):
    def get_item(x):
        index = i
        return x[index]
    
    return get_item


def compare_tree_depths(df):
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(8,5)

    x = ["0.5", "10.0", "100.0", "1000.0"]
    dists = df["distribution"].unique()
    df["tree_depth"] = df["tree_depth"].astype(str)
    df["experiment"] = df['experiment'].str.strip()
    # df["weighing scheme"] = df.apply(extract_multiple_dist_params, axis=1)
    df["param_target"] = df.apply(extract_dist_params_num_only, axis=1)
    df["weighing summary"] = df.apply(extract_dist_params, axis=1)
    # selected_df = df[(df["param_target"] == x) & (df["experiment"] == "gosdt")]
    print(df["param_target"].unique())
    selected_df = df[df["experiment"] == "gosdt"]
    selected_df = df[df["param_target"].isin(x)]
    print(selected_df)

    ax = sns.barplot(data=selected_df, hue="tree_depth", x="weighing summary", y="loss")
    ax.set_title(f"Fico: Loss for params = {x}")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"../tree_comparison.png")

def compare_sampling_methods(df):
    print(df.head())
    sampling_methods = df["sampling_method"].unique()
    df["param_target"] = df.apply(extract_dist_params_num_only, axis=1)
    df["weighing summary"] = df.apply(extract_dist_params, axis=1)
    df["tree_depth"] = df["tree_depth"].astype(str)
    print(sampling_methods)
    fig, axs = plt.subplots(1, len(sampling_methods))
    fig.set_size_inches(20, 4)
    for i in range(len(axs)):
        print(f"Doing: {sampling_methods[i]}")
        df_sampling_method = df[df["sampling_method"] == sampling_methods[i]]
        print(df_sampling_method)
        sns.barplot(data=df_sampling_method, x='weighing summary', y='loss', hue="tree_depth", ax=axs[i])
        # sns.move_legend(axs[i], "upper left", bbox_to_anchor=(1, 1))
        axs[i].set_title(sampling_methods[i])

    plt.tight_layout()
    plt.savefig(f'../sampling_methods.png')

def compare_experiments(df, out_path):

    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(8,5)

    # 
    # decent: 
    # better with unweighted: 

    # dists = df["distribution"].unique()
    df["experiment"] = df['experiment'].str.strip()
    df["data_source"] = df["data_source"].str.strip()
    df["p"] = pd.to_numeric(df["p"]) 

    df["data_args"] = df["data_args"].apply(dash_list_to_py_list)
    df["dist_args"] = df["dist_args"].apply(dash_list_to_py_list)
    df["exp_params"] = df["exp_params"].apply(dash_list_to_py_list)
    df["N"] = df["data_args"].apply(extract_list_elemement(0))
    df["d"] = df["data_args"].apply(extract_list_elemement(1))
    df["class_weight"] = df["dist_args"].apply(extract_list_elemement(0))
    df["mistake_weight"] = df["exp_params"].apply(extract_list_elemement(0))

    df["sampling_method"] = df['sampling_method'].str.strip()
    df["data_gen"] = df["data_gen"].str.strip()
    # df["weighing scheme"] = df.apply(extract_multiple_dist_params, axis=1)
    # df["weight_arg_1"] = df.apply(extract_dist_params_first_num_only, axis=1)
    # df["weight_arg_2"] = df.apply(extract_dist_params_second_num_only, axis=1)
    df["data_summary"] = df.apply(extract_data_summary, axis=1)
    # df["data_arg_1"] = df.apply(extract_data_gen_params_i(0), axis=1)
    # df["data_arg_2"] = df.apply(extract_data_gen_params_i(1), axis=1)
    df["data_arg_1"] = df.apply(extract_data_gen_params_i(0), axis=1)
    # df["data_arg_2"] = df.apply(extract_data_gen_params_i(1), axis=1)
    # df["data_arg_4"] = df.apply(extract_data_gen_params_i(3), axis=1)

    N = [300.0]
    d = [2.0]
    ps = [10.0]
    experiments = ["weighted_tree"]
    mistake_weights = [2.0]

    '''
    Bias to error (exp = weighted tree): 
        p = 2.0, N=300, d=2, mistake_weight=4
        p = 2.0, N=300, d=2, mistake_weight=5
        p = 2.0, N=300, d=3, mistake_weight=5
        p = 2.0, N=300, d=6, mistake_weight=8

        p = 2.0, N=600, d=6, mistake_weight=3

        p = 2.0, N=900, d=4, mistake_weight=3
        p = 2.0, N=900, d=5, mistake_weight=3
        p = 2.0, N=900, d=6, mistake_weight=3

        p = 2.0, N=6000, d=2, mistake_weight=3
    '''

    selected_df = df[(df["experiment"].isin(experiments))\
                     & (df["d"].isin(d))]
                    #  & (df["p"].isin(ps))\
                    #  & (df["N"].isin(N))\
                    #  & (df["mistake_weight"].isin(mistake_weights))]

    print(selected_df.head())
    print(df.columns)
    print("N")
    print(df["N"].unique())
    print("p")
    print(df["p"].unique())
    print("mistake weight")
    print(df["mistake_weight"].unique())
    print("class weight")
    print(df["class_weight"].unique())

    ax = sns.barplot(data=selected_df, hue="sampling_method", x="data_summary", y="loss")
    ax = sns.barplot(data=selected_df, hue="sampling_method", x="data_summary", y="loss")
    # ax.set_title(f"Loss for lin sep (mistake p={data_gen_args[0]}) point bias weight = {dist_args}")
    ax.set_title(f"Loss for Circular")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)

def plot_loss_diff(df):
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(8,5)

    diffs = []
    x = "0.001"
    df["weighing_scheme"] = df.apply(extract_multiple_dist_params, axis=1)
    df["param_target"] = df.apply(extract_dist_params_num_only, axis=1)
    df["experiment"] = df['experiment'].str.strip()
    selected_df = df[df["param_target"] == x]

    for scheme in selected_df["weighing_scheme"].unique():
        print(f"Doing scheme: {scheme}")

        gosdt_row = {}
        gosdt_row["Weighing Method"] = scheme
        gosdt_row["Experiment"] = "gosdt"
        gosdt = selected_df[(selected_df["weighing_scheme"] == scheme)&(selected_df["experiment"] == "gosdt")]["loss"].mean()
        gosdt_without = selected_df[(selected_df["weighing_scheme"] == scheme)&(selected_df["experiment"] == "gosdt-fit-without-weights")]["loss"].mean()
        gosdt_row["Difference in Loss"] = gosdt_without - gosdt

        scikit_row = {}
        scikit_row["Weighing Method"] = scheme
        scikit_row["Experiment"] = "scikit"
        scikit = selected_df[(selected_df["weighing_scheme"] == scheme)&(selected_df["experiment"] == "scikit")]["loss"].mean()
        scikit_without = selected_df[(selected_df["weighing_scheme"] == scheme)&(selected_df["experiment"] == "scikit-fit-without-weights")]["loss"].mean()
        scikit_row["Difference in Loss"] = scikit_without - scikit

        diffs.append(gosdt_row)
        diffs.append(scikit_row)

    
    ax = sns.barplot(data=pd.DataFrame(diffs), hue="Experiment", x="Weighing Method", y="Difference in Loss")
    ax.set_title(f"Difference in Loss For {x}")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"../loss_diff.png")



def resampling_error(df):
    df["N'"] = df['p']*N
    print(df)
    sns.catplot(data=df, col="N'",x="param", y="loss", kind="bar", hue="loss_type")

    plt.savefig("out.png") 
                    

def plot_ps(df, out_path):
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(8,5)

    df["experiment"] = df['experiment'].str.strip()
    df["data_source"] = df["data_source"].str.strip()
    df["p"] = pd.to_numeric(df["p"]) 

    df["data_args"] = df["data_args"].apply(dash_list_to_py_list)
    df["dist_args"] = df["dist_args"].apply(dash_list_to_py_list)
    df["exp_params"] = df["exp_params"].apply(dash_list_to_py_list)
    df["N"] = df["data_args"].apply(extract_list_elemement(0))
    df["d"] = df["data_args"].apply(extract_list_elemement(1))
    df["class_weight"] = df["dist_args"].apply(extract_list_elemement(0))
    df["mistake_weight"] = df["exp_params"].apply(extract_list_elemement(0))
    df["data_summary"] = df.apply(extract_data_summary, axis=1)


    Ns = [300.0]
    ds = [2.0]
    experiments = ["weighted_tree"]
    mistake_weights = [3.0]

    print(df.iloc[0])
    print(df["mistake_weight"].unique())

    selected_df = df[(df["experiment"].isin(experiments))\
                        & (df["d"].isin(ds))\
                        & (df["N"].isin(Ns))\
                        & (df["mistake_weight"].isin(mistake_weights))]

    print(selected_df.head())

    ax = sns.barplot(data=selected_df, hue="sampling_method", x="p", y="loss")
    # ax.set_title(f"Loss for lin sep (mistake p={data_gen_args[0]}) point bias weight = {dist_args}")
    ax.set_title(f"Loss for Circular")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str, help='Location of results file')
    parser.add_argument('--plot_type', type=str, help="Which kind of data to plot")
    parser.add_argument("--out", type=str, help="Location of output", default="plot.png")
    args = parser.parse_args()
    # Read data
    df = pd.read_csv(args.in_path, index_col=None)

    # Plot (# of sampling methods) subplots
    if args.plot_type == "compare":
        compare_experiments(df, args.out)
    elif args.plot_type == "resample":
        resampling_error(df)
    elif args.plot_type == "loss_diff":
        plot_loss_diff(df)
    elif args.plot_type == "tree":
        compare_tree_depths(df)
    elif args.plot_type == "sampling":
        compare_sampling_methods(df)
    elif args.plot_type == "plot_ps":
        plot_ps(df, args.out)



