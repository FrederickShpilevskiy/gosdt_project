import numpy as np
import pandas as pd

def mathiasSampling(data, weights, data_allowance):
    deter_count = np.floor(weights * data_allowance) # determinisitc part of duplication
    # print("disc\n", deter_count[:5])
    # print("p\n", (weights*data_allowance - deter_count)[:5])
    stoch_count = (np.random.rand(weights.shape[0]) < ((weights * data_allowance) - deter_count)).astype(int) # stochastic part
    # print("stoch\n", stoch_count[:5])
    sampled_dups = deter_count + stoch_count # combine to get the samples that should be duplicated
    # print("dups\n", sampled_dups[:5])
    duped_dataset = data.loc[data.index.repeat(sampled_dups)]
    duped_dataset = duped_dataset.reset_index(drop=True)
    return duped_dataset

def generate_data(args, df, weights):
    data = df.copy()
    data_allowance = data.shape[0] * args.p

    sampling_method = args.data_gen
    if sampling_method == 'deterministic':
        dups = np.round(weights * data_allowance)
        duped_dataset = data.loc[data.index.repeat(dups)]
        return duped_dataset.reset_index(drop=True)

    if sampling_method == 'sampling':
        return data.sample(n=int(data_allowance), replace=True, weights=weights, ignore_index=True)

    if sampling_method == 'mathias':
        return mathiasSampling(df, weights, int(data_allowance))
    
    else: 
        print("Sampling method does not exist")
