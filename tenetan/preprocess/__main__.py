import argparse


def p_folder(args):
    from .preprocess import preprocess_directory
    preprocess_directory(args.input_dir, output_file=args.output_file, source_col=args.source_col,
                         target_col=args.target_col, weight_col=args.weight_col)


parser = argparse.ArgumentParser('Tenetan file preprocessing')
subparsers = parser.add_subparsers(title='Subcommands', description='Preprocessing stages')

folder = subparsers.add_parser('dir', help='Process a directory of static networks into a single temporal network file')
folder.add_argument('--input_dir', '-i', required=True, type=str,
                    help='Name of directory containing static network files, '
                         'will be processed in filename sorted order')
folder.add_argument('--output_file', '-o', required=False, default='default',
                    help='Path to save the output, if default, '
                         'will save to {input_dir}_concat.csv alongside input_dir')
folder.add_argument('--source_col', '-s', required=False, default='i', help='Name of the source column in the provided '
                                                                            'network files')
folder.add_argument('--target_col', '-t', required=False, default='j', help='Name of the target column in the provided '
                                                                            'network files')
folder.add_argument('--weight_col', '-w', required=False, default='w', help='Name of the weight column in the provided '
                                                                            'network files')
folder.set_defaults(func=p_folder)
args = parser.parse_args()
args.func(args)
