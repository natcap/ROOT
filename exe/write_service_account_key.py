# encoding=UTF-8
"""Write a service account key from an environment variable."""
import os
import sys
import argparse


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('var', metavar='VAR', type=str,
                        help='The environment variable to write to a file.')
    parser.add_argument('file', metavar='FILE', type=str,
                        help='The file to write.')
    parser.add_argument('--no-crash', action='store_true', default=False,
                        help=("If provided, don't crash the program if the"
                              "environment variable is not found."))
    parsed_args = parser.parse_args(args)

    varname = parsed_args.var.upper()
    try:
        with open(parsed_args.file, 'w') as new_file:
            new_file.write(os.environ[varname])
    except KeyError:
        if parsed_args.no_crash:
            parser.exit(status=0, message=(
                "Could not find environment variable %s, but --no-crash "
                "specified.\n") % varname)
        else:
            parser.exit(status=1, message=(
                "Could not find environment variable %s\n") % varname)


if __name__ == '__main__':
    main()

