import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

N = 10459

# takes strings from " binary(0.25-2.0-1.0)" to "binary(0.25)" if last selected is 0, "binary(2.0)" if selector is 1... 
def extract_dist_params(row):
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

def compare_experiments(df):
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(8,5)

    x = ["100"]
    dists = df["distribution"].unique()
    df["experiment"] = df['experiment'].str.strip()
    # df["weighing scheme"] = df.apply(extract_multiple_dist_params, axis=1)
    df["proportion_with_weight"] = df.apply(extract_dist_params_first_num_only, axis=1)
    df["applied weight"] = df.apply(extract_dist_params_second_num_only, axis=1)
    df["weighing summary"] = df.apply(extract_dist_params, axis=1)
    # selected_df = df[(df["param_target"] == x) & (df["experiment"] == "gosdt")]
    # selected_df = df[df["experiment"] == "gosdt"]
    selected_df = df[df["applied weight"].isin(x)]

    ax = sns.barplot(data=df, hue="proportion_with_weight", x="experiment", y="loss")
    ax.set_title(f"Loss for {'xor dataset'} weight param = {x}")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"../experiment_comparison.png")

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
                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str, help='Location of results file')
    parser.add_argument('--plot_type', type=str, help="Which kind of data to plot")
    args = parser.parse_args()
    # Read data
    df = pd.read_csv(args.in_path)

    # Plot (# of sampling methods) subplots
    if args.plot_type == "compare":
        compare_experiments(df)
    elif args.plot_type == "resample":
        resampling_error(df)
    elif args.plot_type == "loss_diff":
        plot_loss_diff(df)
    elif args.plot_type == "tree":
        compare_tree_depths(df)
    elif args.plot_type == "sampling":
        compare_sampling_methods(df)



