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
    weighted_loss = True 
    # future thing?: param for loss types
    if weighted_loss:
        select_types = ["base_gosdt_weighted_loss", "duplication_weighted_loss"]
    else:
        select_types = ["base_gosdt_accuracy", "duplication_accuracy"]
    df['loss_type'] = df['loss_type'].astype(str)
    df['loss_type'] = df['loss_type'].str.strip()
    # df.query("loss_type in @select_types", inplace=True)
    print(df.dtypes)
    df_selected = df[df['loss_type'].isin(select_types)]
    df_selected["N'"] = df_selected['p']*N
    print(df_selected)
    # plot = sns.catplot(data=df_selected, col="N'",x="param", y="loss", kind="bar", hue="loss_type")
    plot = sns.catplot(data=df_selected, col="N'",x="loss_type", y="loss", kind="bar")
    plot.fig.subplots_adjust(top=.8)
    if weighted_loss:
        plot.fig.suptitle("Weighted Loss with constant weights")
    else:
        plot.fig.suptitle("Accuracy (pretend the y says accuracy)")
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



