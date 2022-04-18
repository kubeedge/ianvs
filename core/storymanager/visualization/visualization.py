import sys
from prettytable import from_csv


def print_table(rank_file):
    """ print rank of the test"""
    fp = open(rank_file, "r")
    table = from_csv(fp)
    print(table)
    fp.close()


def get_visualization_func(mode):
    """ get visualization func """
    return getattr(sys.modules[__name__], mode)
