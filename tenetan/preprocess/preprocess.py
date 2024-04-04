import os
import pathlib

import pandas as pd

__all__ = ['preprocess_directory', 'preprocess_names']


def preprocess_directory(input_dir, /, *, output_file=None, source_col='i', target_col='j', weight_col='w'):
    files = sorted(os.listdir(input_dir))
    all_data = []
    for file in files:
        data = pd.read_csv(os.path.join(input_dir, file))
        data['t'] = pathlib.Path(file).with_suffix('')
        all_data.append(data)
    all_data = pd.concat(all_data, ignore_index=True)
    all_data = all_data.rename(columns={source_col: 'i',
                                        target_col: 'j',
                                        weight_col: 'w'})

    if output_file is not None:
        if output_file == 'default':
            output_file = f'{input_dir}_concat.csv'
        all_data.to_csv(output_file, index=False, columns=['i', 'j', 't', 'w'], header=True)
    return all_data


def preprocess_names(input_file, /, *, source_col='i', target_col='j', time_col='t', weight_col='w',
                     output_file=None, vertex_file=None, timestamp_file=None,
                     sort_vertices=False, sort_timestamps=False):
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

    input_file = str(pathlib.Path(input_file).with_suffix(''))
    if output_file is not None:
        output_file = f'{input_file}_network.csv' if output_file == 'default' else output_file
        data.to_csv(output_file, header=False, index=False, columns=['i', 'j', 't', 'w'])

    if vertex_file is not None:
        vertex_file = f'{input_file}_vertices.txt' if vertex_file == 'default' else vertex_file
        with open(vertex_file, 'w') as f:
            for vertex in vertex_list:
                f.write(f'{vertex}\n')

    if timestamp_file is not None:
        timestamp_file = f'{input_file}_timestamps.txt' if timestamp_file == 'default' else timestamp_file
        with open(timestamp_file, 'w') as f:
            for timestamp in timestamp_list:
                f.write(f'{timestamp}\n')

    return data, vertex_list, timestamp_list


if __name__ == '__main__':

    print('ERROR: Call this package as main with "python -m tenetan.preprocess"')
