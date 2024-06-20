import tensorly as tl
import pandas as pd
import numpy as np

__all__ = ['SnapshotGraph']


class SnapshotGraph:

    def __init__(self):

        self._tensor = None
        self.vertices = []
        self.timestamps = []

    def load_csv(self, input_file, /, *, source_col='i', target_col='j', time_col='t', weight_col='w',
                 directed=True, dtype=np.float32, sort_vertices=False, sort_timestamps=False):

        data = pd.read_csv(input_file)

        vertex_list = list(pd.concat([data[source_col], data[target_col]]).unique())
        vertex_list.sort() if sort_vertices else None

        timestamp_list = list(data[time_col].unique())
        timestamp_list.sort() if sort_timestamps else None

        vertex_index_mapping = {value: index for index, value in enumerate(vertex_list)}
        timestamp_index_mapping = {value: index for index, value in enumerate(timestamp_list)}

        data['i'] = data[source_col].map(vertex_index_mapping)
        data['j'] = data[target_col].map(vertex_index_mapping)
        data['t'] = data[time_col].map(timestamp_index_mapping)
        data['w'] = data[weight_col]

        max_vertex = len(vertex_list)
        max_time = len(timestamp_list)

        tensor = np.full((max_vertex, max_vertex, max_time), 0.0)

        for row in data.itertuples(index=False):
            i, j, t, w = int(row.i), int(row.j), int(row.t), float(row.w)
            tensor[i, j, t] = w
            if directed is False:
                tensor[j, i, t] = w

        self._tensor = tl.tensor(tensor, dtype=dtype)

        self.vertices = vertex_list
        self.timestamps = timestamp_list