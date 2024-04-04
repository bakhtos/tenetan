import argparse


def p_folder(args):
    from .preprocess import preprocess_directory
    preprocess_directory(args.input_dir, output_file=args.output_file, source_col=args.source_col,
                         target_col=args.target_col, weight_col=args.weight_col)


def p_names(args):
    from .preprocess import preprocess_names
    preprocess_names(args.input_file, output_file=args.output_file, timestamp_file=args.timestamp_file,
                     vertex_file=args.vertex_file, source_col=args.source_col, target_col=args.target_col,
                     time_col=args.time_col, weight_col=args.weight_col,
                     sort_vertices=args.sort_vertices, sort_timestamps=args.sort_timestamps)


parser = argparse.ArgumentParser('Tenetan file preprocessing')
subparsers = parser.add_subparsers(title='Subcommands', description='Preprocessing stages')

folder = subparsers.add_parser('dir', help='Process a directory of static networks into a single temporal network file')
folder.add_argument('--input_dir', '-i', required=True, type=str,
                    help='Name of directory containing static network files, will be processed in sorted order')
folder.add_argument('--output_file', '-o', required=True, type=str,
                    help='Path to save the output; if "default", will be {input_dir}_concat.csv alongside input_dir')
folder.add_argument('--source_col', '-s', required=False, type=str, default='i',
                    help='Name of the source column in the provided network files (default i)')
folder.add_argument('--target_col', '-t', required=False, type=str, default='j',
                    help='Name of the target column in the provided network files (default j)')
folder.add_argument('--weight_col', '-w', required=False, type=str, default='w',
                    help='Name of the weight column in the provided network files (default w)')
folder.set_defaults(func=p_folder)

names = subparsers.add_parser('names', help='Process temporal network file vertices and timestamps into a 0-based index')
names.add_argument('--input_file', '-i', required=True, type=str, help='Input file, original temporal network')
names.add_argument('--output_file', '-o', required=True, type=str,
                   help='Network output file; if "default", will be {input_file}_network.csv')
names.add_argument('--vertex_file', required=False, default=None, type=str,
                   help='Vertex list output file - vertices will be saved in same 0-based order; '
                        'if "default", will be {input_file}_vertices.txt')
names.add_argument('--timestamp_file', required=False, default=None, type=str,
                   help='Timestamp list output file - timestamps will be saved in same 0-based order; '
                        'if "default", will be {input_file}_timestamps.txt')
names.add_argument('--source_col', '-s', required=False, default='i', type=str,
                   help = 'Name of the source column in the provided network files (default i)')
names.add_argument('--target_col', '-ta', required=False, default='j', type=str,
                   help='Name of the target column in the provided network files (default j)')
names.add_argument('--weight_col', '-w', required=False, default='w', type=str,
                   help='Name of the weight column in the provided network files (default w)')
names.add_argument('--time_col', '-ti', required=False, default='t', type=str,
                   help='Name of the time column in the provided network files (default t)')
names.add_argument('--sort_vertices', action='store_true', required=False,
                   help='If set, vertices will be sorted to determine 0-based index, otherwise will go in encounter order')
names.add_argument('--sort_timestamps', action='store_true', required=False,
                   help='If set, timestamps will be sorted to determine 0-based index, otherwise will go in encounter order')
names.set_defaults(func=p_names)

args = parser.parse_args()
args.func(args)
