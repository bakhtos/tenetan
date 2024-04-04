import tensorly as tl
import numpy as np

__all__ = ['SnapshotGraph']


class SnapshotGraph:

    # TODO make __init__ and loader separate
    def __init__(self, path, /, *, directed=True, dtype=np.float32):

        data = np.loadtxt(path, dtype=dtype, delimiter=',', comments='#')  # TODO add loadtxt kwargs to func kwargs
        max_coords = np.max(data[:, :3], axis=0).astype(int) + 1
        max_vertex = max(max_coords[0], max_coords[1])
        max_time = max_coords[2]
        tensor = np.full((max_vertex, max_vertex, max_time), 0.0)
        for row in data:
            i, j, t, w = int(row[0]), int(row[1]), int(row[2]), float(row[3])
            tensor[i, j, t] = w
            if directed is False:
                tensor[j, i, t] = w

        self._tensor = tl.tensor(tensor, dtype=dtype)

