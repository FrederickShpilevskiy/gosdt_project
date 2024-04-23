import numpy as np
import pandas as pd
import sklearn.datasets as datasets
import random as rand


# start with 2 classes
# Idea, kth feature's domain depends on the k-1th feature's domain. Alternate so no lin sep
# Eg. feat 1 <= 0 so feat 2 > 0 so feat 3 <= 0
# def dep_prev_feat(d:int, n:int):


# Used to have seperable and overlapping features (commented out code) now just
# seperable data but we intentionally make mistakes for some % of data

def lin_seperable_with_mistakes(d_sep: int, n: int, num_classes: int, p_mistake:float):
    sep_feat_ranges = []
    for _ in range(d_sep):
        start = rand.randint(-10, 0) # starting vals
        width = rand.randint(3,5) # range of values for each class
        # end = randint(start+(num_classes-1)*width, start + width*num_classes)
        end = start + num_classes*width
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
                val = rand.randint(start + width*label, start + width*(label+1) - 1) # -1 since inclusive
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
                val = rand.randint(start + width*label, start + width*(label+1) - 1) # -1 since inclusive
                row[f"sep_{feat}"] = val

            # Mistake = any label but the right one
            non_target_labels = list(set(range(num_classes)) - set([label]))
            row["target"] = rand.choice(non_target_labels)
            data.append(row)


    rand.shuffle(data)
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
                val = rand.randint(-20,0)
            else:
                val = rand.randint(1, 20)

            row[f"feat_{feat}"] = val
        
        row[f"xor"] = 1
        data.append(row)

    # Half to be all 0, half to be all 1
    # At worst, 0 class gets one less but I am willing to live with that 
    for _ in range(num_per_class//2):
        row = {}
        for i in range(d):
            val = rand.randint(-20, 0)
            row[f"feat_{i}"] = val
        
        row[f"xor"] = 0
        data.append(row)
    
    for _ in range(num_per_class // 2):
        row = {}
        for i in range(d):
            val = rand.randint(1, 20)
            row[f"feat_{i}"] = val
        
        row["xor"] = 0
        data.append(row)

    rand.shuffle(data)
    return pd.DataFrame(data)

# According to https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
def generate_circular_points(N, d, radius, label):
    points_per_feat = []
    thetas = []
    for i in range(d-1):
      thetas.append(np.random.rand(N) * 2 * np.pi)
    
    for i in range(d-1):
      points_x_i = np.ones(N) * radius
      for j in range(i):
        points_x_i = points_x_i * np.sin(thetas[j])
      
      points_x_i = points_x_i * np.cos(thetas[i])
      points_per_feat.append(points_x_i)

    # last one is all sin
    points_x_last = np.ones(N) * radius
    for i in range(d-1):
      points_x_last = points_x_last * np.sin(thetas[i])

    points_per_feat.append(points_x_last)
    

    X = np.vstack(points_per_feat).T
    y = (np.ones(N) * label).reshape(-1, 1)

    return X, y

# Idea: make 3 circles, middle one with less weight and the outter + inner with equal larger weight
# Label inner and outer as 1, middle as 0
# TODO: consider adding more points to middle
def circular_d(N, d, center=None):
    # Parameters for the three circles
    n_samples = N // 3

    if center is not None:
        try:
            shape = center.shape
        except AttributeError:
            print("Error: given center is not a numpy array")
            exit()
        if shape != (d,):
            print(f"Error: given center has wrong shape, should be ({d},)")

    else:
        center = np.zeros(d)        

    # Generate points for each circle
    sphere_inner, y_inner = generate_circular_points(n_samples, d, 1, 1)
    sphere_middle, y_middle = generate_circular_points(n_samples, d, 2, 0)
    sphere_outer, y_outer = generate_circular_points(n_samples, d, 3, 1)

    sphere_inner += center
    sphere_middle += center
    sphere_outer += center

    # Combine points from all circles
    X = np.vstack([sphere_inner, sphere_middle, sphere_outer])

    # Create labels for the circles
    y = np.vstack([y_inner, y_middle, y_outer])

    col_labels = [f"x{i}" for i in range(d)]
    col_labels.append("y")
    return pd.DataFrame(np.concatenate((X, y), axis=1), columns=col_labels)

def multiball(N, d):
    # Very contrived centers for now (generalize later)
    N_balls = 4
    N_per_ball = N//N_balls
    all_pos = np.ones(d)*10
    all_neg = np.ones(d)*-10
    front_half = np.ones(d)
    front_half[:d//2] = front_half[:d//2]*-10
    front_half[d//2:] = front_half[d//2:]*10
    back_half = np.ones(d)
    back_half[:d//2] = back_half[:d//2]*10
    back_half[d//2:] = back_half[d//2:]*-10

    all_pos_df = circular_d(N_per_ball, d, center=all_pos)
    all_neg_df = circular_d(N_per_ball, d, center=all_neg)
    front_half_df = circular_d(N_per_ball, d, center=front_half)
    back_half_df = circular_d(N_per_ball, d, center=back_half)

    return pd.concat([all_pos_df, all_neg_df, front_half_df, back_half_df], ignore_index=True)

def generate_data(gen_method, seed, *kwargs):
    np.random.seed(seed)
    rand.seed(seed)
    if gen_method == "xor":
        return xor(int(kwargs[0]), int(kwargs[1]))
    elif gen_method == "lin_sep":
        return lin_seperable_with_mistakes(int(kwargs[0]), int(kwargs[1]), int(kwargs[2]), kwargs[3])
    elif gen_method == "circular":
        # I am sure there is a better way to do center as a hyper-param arg, but this works so going with the quick and dirty for now
        if kwargs[2:] == ():
            center = None
        else:
            center = np.array(kwargs[2:])
        return circular_d(int(kwargs[0]), int(kwargs[1]), center=center)
    elif gen_method == "multiball":
        return multiball(int(kwargs[0]), int(kwargs[1]))
    else:
        print("Data Gen Method does not exist")