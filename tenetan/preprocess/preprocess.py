import os
import pathlib

import pandas as pd

__all__ = ['preprocess_directory']


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


if __name__ == '__main__':

    print('ERROR: Call this package as main with "python -m tenetan.preprocess"')
