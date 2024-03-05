import tensorly as tl
import numpy as np

__all__ = ['SnapshotGraph']


class SnapshotGraph:

    def __init__(self, path, /, *, directed=True):

        # TODO Load network from path
        # If directed is true, n[i,j,t] =w; if false, also n[j,i,t] = w
        data = np.loadtxt(path, dtype=np.float32, delimiter=',', comments='#')
        max_coords = np.max(data[:, :3], axis=0) + 1
        value_array = np.full((max_coords[0], max_coords[1], max_coords[2]), 0.0)
        for row in data:
            x, y, z, value = map(int, row)
            value_array[x, y, z] = value

