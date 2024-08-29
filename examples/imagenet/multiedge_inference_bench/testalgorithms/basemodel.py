# Modified Copyright 2022 The KubeEdge Authors.
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
import glob
import os
from collections import OrderedDict
from pathlib import Path

from sedna.common.class_factory import ClassType, ClassFactory

import yaml


__all__ = ["BaseModel"]

# set backend
os.environ["BACKEND_TYPE"] = "TORCH"


def make_parser():
    parser = argparse.ArgumentParser("ViT Eval")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--devices_info", default="./devices.yaml", type=str, help="devices conf")
    parser.add_argument("--model_parallel", default=True, action="store_true")
    
    # remove conflict with ianvs
    parser.add_argument("-f")
    return parser


@ClassFactory.register(ClassType.GENERAL, alias="Classification")
class BaseModel:

    def __init__(self, **kwargs) -> None:
        self.args = make_parser().parse_args()
        self.devices_info_url = self.args.devices_info
        self.model_parallel = self.args.model_parallel
        self.partition_point_list = self._parse_devices_info(self.devices_info_url).get('partition_points')
        return


    def load(self, model_url=None) -> None:
        return

    def predict(self, data, input_shape=None, **kwargs):
        return


    def _get_eval_loader(self, data_dir, coco, ids, class_ids, annotations, batch_size, is_distributed, testdev=False):
        return

    def _parse_devices_info(self, url):
        """Convert yaml file to the dict."""
        if url.endswith('.yaml') or url.endswith('.yml'):
            with open(url, "rb") as file:
                devices_info_dict = yaml.load(file, Loader=yaml.SafeLoader)
                return devices_info_dict
        else:
            raise RuntimeError('config file must be the yaml format')