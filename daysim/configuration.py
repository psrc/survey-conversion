import argparse
import os


def add_run_args(parser, multiprocess=True):
    """
    Run command args
    """
    parser.add_argument(
        "-c", "--configs_dir", type=str, metavar="PATH", help="path to configs dir"
    )

parser = argparse.ArgumentParser()
add_run_args(parser)
args = parser.parse_args()
