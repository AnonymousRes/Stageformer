from __future__ import absolute_import
from __future__ import print_function

from DataGenerating import common_utils
from sklearn.preprocessing import StandardScaler
import numpy as np
import os



def load_data(reader, discretizer, small_part=False, return_names=False):
    N = reader.get_number_of_examples()
    if small_part:
        N = 1000
    ret = common_utils.read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]
    X_norm = []

    print("Discretizer")
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]

    for p in data:
        for row in p:
            X_norm.append(row)
    X_norm = np.array(X_norm, dtype=float)
    normalizer = StandardScaler()
    normalizer.fit(X_norm)
    data = [normalizer.transform(X) for X in data]
    print('X_norm_shape:', X_norm.shape)
    print('normalizer_mean:', normalizer.mean_)

    whole_data = (np.array(data), labels, names)
    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}


def save_results(names, pred, y_true, path):
    common_utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("stay,prediction,y_true\n")
        for (name, x, y) in zip(names, pred, y_true):
            f.write("{},{:.6f},{}\n".format(name, x, y))
