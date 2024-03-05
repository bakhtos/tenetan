import tensorly as tl
import numpy as np

__all__ = ['SnapshotGraph']


class SnapshotGraph:

    def __init__(self, path, /, *, directed=True):

        # TODO Load network from path
        # If directed is true, n[i,j,t] =w; if false, also n[j,i,t] = w
        data = np.loadtxt(path, dtype=np.float32, delimiter=',', comments='#')
        max_coords = np.max(data[:, :3], axis=0) + 1
        tensor = np.full((max_coords[0], max_coords[1], max_coords[2]), 0.0)
        for row in data:
            i, j, t, w = int(row[0]), int(row[1]), int(row[2]), float(row[3])
            tensor[i, j, t] = w

        self.tensor = tensor

