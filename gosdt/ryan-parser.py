import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Location of output file")
    args = parser.parse_args()
    f = open(args.file)
    lines = f.readlines()

    parsed_values: list[dict] = []
    mathias_sampling = True
    N = len(lines)
    i = 1
    while i < N:
        line = lines[i]
        if line.startswith("---"):
            q = float(line.split("=")[1].split(',')[0])
            p_or_r = float(line.split("=")[2].split(" ")[0])
            i += 6
            next = lines[i]
            acc = float(next.split(" ")[-1].split("\n")[0])
            i += 3
            if mathias_sampling:
                parsed_values.append({"Sampling": "mathias", "q": q, "p": p_or_r, "accuracy": acc})
            else:
                parsed_values.append({"Sampling": "gosdtwG", "q": q, "r": p_or_r, "accuracy": acc})
        elif line.startswith("~~~"):
            mathias_sampling = False
            i+=1
        else:
            i+= 1
        
    df = pd.DataFrame(parsed_values)
    fig, axs = plt.subplots(2)
    sns.barplot(data=df, x="q", y="accuracy", hue="p", ax=axs[0])
    sns.barplot(data=df, x="q", y="accuracy", hue="r", ax=axs[1])
    sns.move_legend(axs[0], "upper left", bbox_to_anchor=(1, 1))
    sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 1))
    axs[0].set_title("Mathias Sampling Method")
    axs[1].set_title("GOSDTwG Sampling Method")
    plt.tight_layout()
    plt.savefig("sampling_methods.png")