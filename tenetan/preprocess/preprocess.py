import argparse
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


def preprocess_names(input_file, /, source_col='i', target_col='j', time_col='t', weight_col='w',
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
        if output_file == 'default':
            output_file = 'network.csv'
        data.to_csv(output_file, header=False, index=False, columns=['i', 'j', 't', 'w'])

    if vertex_file is not None:
        if vertex_file == 'default':
            vertex_file = 'vertices.txt'
        with open(vertex_file, 'w') as f:
            for vertex in vertex_list:
                f.write(f'{vertex}\n')

    if timestamp_file is not None:
        if timestamp_file == 'default':
            timestamp_file = 'timestamps.txt'
        with open(timestamp_file, 'w') as f:
            for timestamp in timestamp_list:
                f.write(f'{timestamp}\n')

    return data, vertex_list, timestamp_list


if __name__ == '__main__':

    def p_folder(args):
        preprocess_directory(args.input_dir, output_file=args.output_file, source_col=args.source_col,
                             target_col=args.target_col, weight_col=args.weight_col)

    parser = argparse.ArgumentParser('Tenetan file preprocessing')
    subparsers = parser.add_subparsers()

    folder = subparsers.add_parser('dir', help='Process a directory of static networks into a temporal network')
    folder.add_argument('--input_dir', '-i', required=True, type=str, help='Name of directory containing static network files, '
                                                                       'will be processed in sorted order')
    folder.add_argument('--output_file', '-o', required=False, default='default', help='Path to save the output, if default, '
                                                                                       'will save to {input_dir}_concat.csv in CWD')
    folder.add_argument('--source_col', '-s', required=False, default='i', help='Name of the source column in the provided '
                                                                                'network files, will be renamed to "i" in the '
                                                                                'output')
    folder.add_argument('--target_col', '-t', required=False, default='j', help='Name of the target column in the provided '
                                                                                'network files, will be renamed to "j" in the '
                                                                                'output')
    folder.add_argument('--weight_col', '-w', required=False, default='w', help='Name of the weight column in the provided '
                                                                                'network files, will be renamed to "w" in the '
                                                                                'output')
    folder.set_defaults(func=p_folder)
    args = parser.parse_args()
    args.func(args)
