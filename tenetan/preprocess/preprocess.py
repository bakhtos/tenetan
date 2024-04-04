import os
import pandas as pd

__all__ = ['preprocess_directory', 'preprocess_names']


def preprocess_directory(input_dir, /, *, output_file=None, source_col='i', target_col='j', weight_col='w'):
    files = sorted(os.listdir(input_dir))
    all_data = []
    for time, file in enumerate(files):
        data = pd.read_csv(os.path.join(input_dir, file))
        data['t'] = time
        all_data.append(data)
    all_data = pd.concat(all_data, ignore_index=True)
    all_data = all_data.rename(columns={source_col: 'i',
                                        target_col: 'j',
                                        weight_col: 'w'})

    if output_file is not None:
        if output_file == 'default':
            output_file = f'{input_dir}_concat.csv'
        all_data.to_csv(output_file, index=False, columns=['i', 'j', 't', 'w'], header=False)
    return all_data


def preprocess_names(input_file, /, *, source_col='i', target_col='j', time_col='t', weight_col='w',
                     output_file=None, vertex_file=None, timestamp_file=None):
    data = pd.read_csv(input_file)

    vertex_list = sorted(pd.concat([data[source_col], data[target_col]]).unique().to_list())
    timestamp_list = sorted(data[time_col].unique().tolist())

    vertex_index_mapping = {value: index for index, value in enumerate(vertex_list)}
    timestamp_index_mapping = {value: index for index, value in enumerate(timestamp_list)}

    data['i'] = data[source_col].map(vertex_index_mapping)
    data['j'] = data[target_col].map(vertex_index_mapping)
    data['t'] = data[time_col].map(timestamp_index_mapping)
    data['w'] = data[weight_col]

    if output_file is not None:
        output_file = 'network.csv' if output_file == 'default' else output_file
        data.to_csv(output_file, header=False, index=False, columns=['i', 'j', 't', 'w'])

    if vertex_file is not None:
        vertex_file = 'vertices.txt' if vertex_file == 'default' else vertex_file
        with open(vertex_file, 'w') as f:
            for vertex in vertex_list:
                f.write(f'{vertex}\n')

    if timestamp_file is not None:
        timestamp_file = 'timestamps.txt' if timestamp_file == 'default' else timestamp_file
        with open(timestamp_file, 'w') as f:
            for timestamp in timestamp_list:
                f.write(f'{timestamp}\n')

    return data, vertex_list, timestamp_list


if __name__ == '__main__':

    print('ERROR: Call this package as main as "python -m tenetan.preprocess"')
