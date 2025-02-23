import csv
import os.path
import pathlib

import tensorly as tl
import pandas as pd
import numpy as np

__all__ = ['SnapshotGraph']


class SnapshotGraph:

    def __init__(self):

        self._tensor = None
        self._vertices: list = []
        self._timestamps: list = []
        self._vertex_index_mapping: dict[str, int] = {}
        self._timestamp_index_mapping: dict[str, int] = {}

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, new_value):
        assert type(new_value) is list
        assert len(self._vertices) == len(new_value)
        self._vertices = new_value
        self._vertex_index_mapping = {value: index for index, value in enumerate(new_value)}

    @property
    def tensor(self):
        return self._tensor

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def N(self):
        return len(self._vertices)

    @property
    def T(self):
        return len(self._timestamps)

    @timestamps.setter
    def timestamps(self, new_value):
        assert type(new_value) is list
        assert len(self._timestamps) == len(new_value)
        self._timestamps = new_value
        self._timestamp_index_mapping = {value: index for index, value in enumerate(new_value)}

    def load_csv(self, input_file, /, *, source_col='i', target_col='j', time_col='t', weight_col='w',
                 directed=True, dtype=np.float32, sort_vertices=False, sort_timestamps=False):

        data = pd.read_csv(input_file)

        self._load_pandas(data, directed, dtype, sort_timestamps, sort_vertices, source_col, target_col, time_col,
                          weight_col)

    def _load_pandas(self, dataframe, directed, dtype, sort_timestamps, sort_vertices, source_col, target_col, time_col,
                     weight_col):

        vertex_list = list(pd.concat([dataframe[source_col], dataframe[target_col]]).unique())
        vertex_list.sort() if sort_vertices else None
        timestamp_list = list(dataframe[time_col].unique())
        timestamp_list.sort() if sort_timestamps else None
        vertex_index_mapping = {value: index for index, value in enumerate(vertex_list)}
        timestamp_index_mapping = {value: index for index, value in enumerate(timestamp_list)}
        dataframe['i'] = dataframe[source_col].map(vertex_index_mapping)
        dataframe['j'] = dataframe[target_col].map(vertex_index_mapping)
        dataframe['t'] = dataframe[time_col].map(timestamp_index_mapping)
        dataframe['w'] = dataframe[weight_col]
        max_vertex = len(vertex_list)
        max_time = len(timestamp_list)
        tensor = np.full((max_vertex, max_vertex, max_time), 0.0)
        for row in dataframe.itertuples(index=False):
            i, j, t, w = int(row.i), int(row.j), int(row.t), float(row.w)
            tensor[i, j, t] = w
            if directed is False:
                tensor[j, i, t] = w
        self._tensor = tl.tensor(tensor, dtype=dtype)
        self._vertices = vertex_list
        self._timestamps = timestamp_list
        self._vertex_index_mapping = vertex_index_mapping
        self._timestamp_index_mapping = timestamp_index_mapping

    def load_csv_directory(self, directory, /, *, source_col='i', target_col='j', weight_col='w',
                           directed=True, dtype=np.float32, sort_vertices=False):
        all_data = []
        for file in sorted(os.listdir(directory)):
            data = pd.read_csv(os.path.join(directory, file))
            data['t'] = str(pathlib.Path(file).with_suffix(''))
            all_data.append(data)
        all_data = pd.concat(all_data, ignore_index=True)
        self._load_pandas(all_data, source_col=source_col, target_col=target_col, weight_col=weight_col, directed=directed,
                          dtype=dtype, sort_vertices=sort_vertices, time_col='t', sort_timestamps=False)

    def write_csv(self, path, /, *, source_col='i', target_col='j', time_col='t', weight_col='w'):

        csv_header = [source_col, target_col, time_col, weight_col]
        csv_rows = []

        for source_vertex, i in self._vertex_index_mapping.items():
            for target_vertex, j in self._vertex_index_mapping.items():
                for timestamp, t in self._timestamp_index_mapping.items():
                    if (w := self._tensor[i, j, t]) != 0.0:
                        csv_rows.append([source_vertex, target_vertex, timestamp, w])

        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_header)
            writer.writerows(csv_rows)
