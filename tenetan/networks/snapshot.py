import tensorly as tl
import numpy as np

__all__ = ['SnapshotGraph']

import tensorly.decomposition


class SnapshotGraph:

    def __init__(self, path, directed=True):

        # TODO Load network from path
        # If directed is true, n[i,j,t] =w; if false, also n[j,i,t] = w
        data = np.genfromtxt(path, delimiter=',', dtype=None)

        # Convert data to the desired format
        converted_data = []
        for row in data:
            try:
                converted_row = [int(row[0]), int(row[1]), int(row[2]), float(row[3])]
                converted_data.append(converted_row)
            except Exception:
                pass

        # Convert the list to a numpy array
        converted_data = np.array(converted_data)

        # Create a 3D tensor from the data
        self.tensor = converted_data.reshape((converted_data.shape[0], 1, 1, converted_data.shape[1]))
        tensorly.decomposition.non_negative_parafac(tensor=self.tensor, rank=3, n_iter_max=100)
        





