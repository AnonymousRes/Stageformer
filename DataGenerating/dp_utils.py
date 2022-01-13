from __future__ import absolute_import
from __future__ import print_function
from sklearn.preprocessing import StandardScaler

from DataGenerating import common_utils
import threading
import os
import numpy as np
import random
from tqdm import tqdm


def load_data(dataloader, discretizer):
    timestep = discretizer._timestep

    def get_bin(t):
        eps = 1e-6
        return int(t / timestep - eps)

    N = len(dataloader._data["X"])
    Xs = []
    ts = []
    masks = []
    curmasks = []
    ys = []
    names = []

    for i in tqdm(range(N), desc='Discretizer'):
        X = dataloader._data["X"][i]
        cur_ts = dataloader._data["ts"][i]
        cur_ys = dataloader._data["ys"][i]
        name = dataloader._data["name"][i]

        cur_ys = [int(x) for x in cur_ys]

        T = max(cur_ts)
        nsteps = get_bin(T) + 1
        mask = [1] * nsteps
        curmask = [0] * nsteps
        curmask[-1] = 1
        y = [0] * nsteps

        for pos, z in zip(cur_ts, cur_ys):
            y[get_bin(pos)] = z

        X = discretizer.transform(X, end=T)[0]


        Xs.append(X)
        masks.append(np.array(mask))
        curmasks.append(np.array(curmask))
        ys.append(np.array(y))
        names.append(name)
        ts.append(cur_ts)

        assert np.sum(mask) > 0
        assert len(X) == len(mask) and len(X) == len(y)


    yys = [np.expand_dims(yyi, axis=-1) for yyi in ys] #(B, T, 1)
    finaldata = [[Xs, masks, curmasks, names], yys]

    return finaldata
