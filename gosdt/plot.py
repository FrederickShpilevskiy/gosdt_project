import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

N = 10459

def compare_sampling_methods(df):
    sampling_methods = df.sampling_method.unique()
    fig, axs = plt.subplots(1, len(sampling_methods))
    fig.set_size_inches(20, 4)
    for i in range(len(axs)):
        df_sampling_method = df[df.sampling_method == sampling_methods[i]]
        sns.barplot(data=df_sampling_method, x='distribution', y='accuracy', hue="p", ax=axs[i])
        sns.move_legend(axs[i], "upper left", bbox_to_anchor=(1, 1))
        axs[i].set_title(sampling_methods[i])
    plt.tight_layout()
    plt.savefig(f'../sampling_methods.png')

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
        compare_sampling_methods(df)
    elif args.plot_type == "resample":
        resampling_error(df)



