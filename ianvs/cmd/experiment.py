import argparse
import sys
import yaml
from yaml.loader import SafeLoader

from ianvs.experiment import TestJob


def main():
    args = parse_args()
    config = yaml.load(args.config_file, Loader=SafeLoader)
    test_job = TestJob(config[str.lower(TestJob.__name__)])
    print(test_job.__dict__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file",
                        nargs='?', default=sys.stdin,
                        type=argparse.FileType(),
                        help="test config file to read from (if not provided"
                             " standard input is used instead)")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
