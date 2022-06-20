# Copyright 2022 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from core.common.log import LOGGER
from core.common import utils
from core.cmd.obj.benchmarkingjob import BenchmarkingJob


def main():
    args = parse_args()
    try:
        if args.benchmarking_config_file:
            config = utils.yaml2dict(args.benchmarking_config_file)
    except Exception as err:
        LOGGER.exception(f"load config file(url={args.config_file} failed, error: {err}.")

    try:
        job = BenchmarkingJob(config[str.lower(BenchmarkingJob.__name__)])
    except ValueError as err:
        LOGGER.exception(f"init test job failed, error: {err}")
        return

    try:
        job.run()
    except Exception as err:
        LOGGER.exception(f"test job(name={job.name}) runs failed, error: {err}.")
        return

    LOGGER.info(f"test job(name={job.name}) runs successfully!")


def parse_args():
    parser = argparse.ArgumentParser(description='AI Benchmarking Tool')
    parser.add_argument("-f", "--benchmarking_config_file",
                        nargs="?", default="~/benchmarking_config_file.yaml",
                        type=str,
                        help="the benchmarking config file must be yaml/yml file")
    parser.add_argument("-v",  help="the version of tool")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
