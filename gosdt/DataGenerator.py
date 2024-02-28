import numpy as np
import pandas as pd

from random import randint, shuffle, choice


# start with 2 classes
# Idea, kth feature's domain depends on the k-1th feature's domain. Alternate so no lin sep
# Eg. feat 1 <= 0 so feat 2 > 0 so feat 3 <= 0
# def dep_prev_feat(d:int, n:int):


# Used to have seperable and overlapping features (commented out code) now just
# seperable data but we intentionally make mistakes for some % of data

def lin_seperable_with_mistakes(d_sep: int, n: int, num_classes: int, p_mistake:float):
    sep_feat_ranges = []
    for _ in range(d_sep):
        start = randint(-10, 0) # starting vals
        width = randint(1,5) # range of values for each class
        end = randint(start+num_classes, start + width*num_classes)
        sep_feat_ranges.append((start, width, end))

    # overlap_ranges = []
    # for _ in range(d_overlap):
    #     overlap_ranges.append((randint(-5,-3), randint(3, 5)))

    assert(n >= num_classes)

    data = []
    n_mistake = int(n*p_mistake)
    n_correct = n - n_mistake

    for label in range(num_classes):
        for _ in range(n_correct//num_classes):
            row = {}
            for feat in range(d_sep):
                start, width, end = sep_feat_ranges[feat]
                val = randint(start + width*label, start + width*(label+1) - 1) # -1 since inclusive
                row[f"sep_{feat}"] = val

                # for feat in range(d_overlap):
                #     start, end = overlap_ranges[feat]
                #     row[f"overlap_{feat}"] = randint(start, end)

            row["target"] = label
            data.append(row)

        for _ in range(n_mistake//num_classes):
            row = {}
            for feat in range(d_sep):
                start, width, end = sep_feat_ranges[feat]
                val = randint(start + width*label, start + width*(label+1) - 1) # -1 since inclusive
                row[f"sep_{feat}"] = val

            # Mistake = any label but the right one
            non_target_labels = list(set(range(num_classes)) - set([label]))
            row["target"] = choice(non_target_labels)
            data.append(row)


    shuffle(data)
    return pd.DataFrame(data)

    


# Consider all values > 0 to be true and <= 0 to be false
# XOR label for all given variables
# Two classes, one for xor = true and one for xor = false

def xor(d: int, n: int):
    assert(d > 1)
    data = []
    num_per_class = n // 2

    # Equal portion of each feature being <= 0
    num_per_d = num_per_class // d

    # avoids rounding errors when n//2 and d have weird gcd 
    num_per_class = num_per_d * d
    for i in range(num_per_class):
        row = {}
        curr_d_to_false = i // num_per_d

        for feat in range(d):
            if feat == curr_d_to_false:
                val = randint(-20,0)
            else:
                val = randint(1, 20)

            row[f"feat_{feat}"] = val
        
        row[f"xor"] = 1
        data.append(row)

    # Half to be all 0, half to be all 1
    # At worst, 0 class gets one less but I am willing to live with that 
    for _ in range(num_per_class//2):
        row = {}
        for i in range(d):
            val = randint(-20, 0)
            row[f"feat_{i}"] = val
        
        row[f"xor"] = 0
        data.append(row)
    
    for _ in range(num_per_class // 2):
        row = {}
        for i in range(d):
            val = randint(1, 20)
            row[f"feat_{i}"] = val
        
        row["xor"] = 0
        data.append(row)

    shuffle(data)
    return pd.DataFrame(data)


def generate_data(gen_method, *kwargs):
    if gen_method == "xor":
        return xor(int(kwargs[0]), int(kwargs[1]))
    elif gen_method == "lin_sep":
        return lin_seperable_with_mistakes(int(kwargs[0]), int(kwargs[1]), int(kwargs[2]), kwargs[3])
    else:
        print("Data Gen Method does not exist")