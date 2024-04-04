import argparse


def p_folder(args):
    from .preprocess import preprocess_directory
    preprocess_directory(args.input_dir, output_file=args.output_file, source_col=args.source_col,
                         target_col=args.target_col, weight_col=args.weight_col)


def p_names(args):
    from .preprocess import preprocess_names
    preprocess_names(args.input_dir, output_file=args.output_file, timestamp_file=args.timestamp_file,
                     vertex_file=args.vertex_file, source_col=args.source_col, target_col=args.target_col,
                     time_col=args.time_col, weight_col=args.weight_col)


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

names = subparsers.add_parser('names', help='Process a temporal network file vertices and timestamps into a 0-based index')
names.add_argument('--input_file', '-i', required=True, type=str, help='Input file, original temporal network')
names.add_argument('--output_file', '-o', required=False, default='default', help='Network output file; if "default", '
                                                                                  'will be {input_file}_network.csv')
names.add_argument('--vertex_file', required=False, default='default', help='Vertex list output file - vertices '
                                                                            'will be saved in same 0-based order; '
                                                                            'if "default", will be {input_file}_vertices.txt')
names.add_argument('--timestamp_file', required=False, default='default', help='Timestamp list output file - timestamps '
                                                                               'will be saved in same 0-based order; '
                                                                               'if "default", will be {input_file}_timestamps.txt')
names.add_argument('--source_col', '-s', required=False, default='i', help='Name of the source column in the provided '
                                                                           'network files')
names.add_argument('--target_col', '-ta', required=False, default='j', help='Name of the target column in the provided '
                                                                           'network files')
names.add_argument('--weight_col', '-w', required=False, default='w', help='Name of the weight column in the provided '
                                                                           'network files')
names.add_argument('--time_col', '-ti', required=False, default='t', help='Name of the time column in the provided '
                                                                         'network files')
names.set_defaults(func=p_names)

args = parser.parse_args()
args.func(args)
