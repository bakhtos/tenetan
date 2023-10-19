import tensorly as tl
import numpy as np
from tensorly import random

__all__ = ['SnapshotGraph']


class SnapshotGraph:

    def __init__(self, path, directed=True):
        self.path = path
        self.directed = directed

        # TODO Load network from path
        # If directed is true, n[i,j,t] =w; if false, also n[j,i,t] = w
        pass

    arr = np.loadtxt("D:/tenetan/tenetan/datasets/small.csv", delimiter=",")
    tensor = tl.tensor(arr)
    print(tensor)

