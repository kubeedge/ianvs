import argparse

from core.common.log import LOGGER
from core.common import utils
from core.cmd.obj.benchmarkingjob import TestJob


def main():
    args = parse_args()
    try:
        if args.config_file:
            config = utils.yaml2dict(args.config_file)
    except Exception as err:
        LOGGER.exception(f"load config file(url={args.config_file} failed, error: {err}.")

    try:
        test_job = TestJob(config[str.lower(TestJob.__name__)])
    except ValueError as err:
        LOGGER.exception(f"init test job failed, error: {err}")
        return

    try:
        test_job.run()
    except Exception as err:
        LOGGER.exception(f"test job(name={test_job.name}) runs failed, error: {err}.")
        return

    LOGGER.info(f"test job(name={test_job.name}) runs successfully!")


def parse_args():
    parser = argparse.ArgumentParser(description='local AI test tool')
    parser.add_argument("-f", "--config_file",
                        nargs="?", default="~/config_file.yaml",
                        # nargs="?",
                        # default= "/home/yj/ianvs/examples/pcb-aoi/benchmarkingjob/benchmarkingjob.yaml",
                        type=str,
                        help="the config file for local AI test must be yaml format")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
