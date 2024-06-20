import argparse


def p_folder(args):
    from .preprocess import preprocess_directory
    preprocess_directory(args.input_dir, output_file=args.output_file, source_col=args.source_col,
                         target_col=args.target_col, weight_col=args.weight_col)


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

args = parser.parse_args()
args.func(args)
