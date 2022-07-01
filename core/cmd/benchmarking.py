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

"""main"""

import argparse

from core.common.log import LOGGER
from core.common import utils
from core.cmd.obj.benchmarkingjob import BenchmarkingJob
from core.__version__ import __version__


def main():
    """ main command-line interface to ianvs"""
    try:
        parser = _generate_parser()
        args = parser.parse_args()
        config_file = args.benchmarking_config_file
        if not utils.is_local_file(config_file):
            raise SystemExit(f"not found benchmarking config({config_file}) file in local")

        config = utils.yaml2dict(args.benchmarking_config_file)
        job = BenchmarkingJob(config[str.lower(BenchmarkingJob.__name__)])
        job.run()

        LOGGER.info("benchmarkingjob runs successfully.")
    except Exception as err:
        raise Exception(f"benchmarkingjob runs failed, error: {err}.") from err


def _generate_parser():
    parser = argparse.ArgumentParser(description='AI Benchmarking Tool')
    parser.prog = "ianvs"

    parser.add_argument("-f",
                        "--benchmarking_config_file",
                        nargs="?",
                        type=str,
                        help="the benchmarking config file must be yaml/yml file")

    parser.add_argument('-v',
                        '--version',
                        action='version',
                        version=__version__,
                        help='show program version info and exit.')

    return parser


if __name__ == '__main__':
    main()
