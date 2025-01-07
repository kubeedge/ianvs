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
from collections import defaultdict
import time

from sedna.common.class_factory import ClassType, ClassFactory
from dataset import load_dataset

import yaml
import onnxruntime as ort
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pynvml


__all__ = ["BaseModel"]

# set backend
os.environ["BACKEND_TYPE"] = "ONNX"


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
    parser.add_argument("--split", default="val", type=str, help="split of dataset")
    parser.add_argument("--indices", default=None, type=str, help="indices of dataset")
    parser.add_argument("--shuffle", default=False, action="store_true", help="shuffle data")
    parser.add_argument("--model_name", default="google/vit-base-patch16-224", type=str, help="model name")
    parser.add_argument("--dataset_name", default="ImageNet", type=str, help="dataset name")
    parser.add_argument("--data_size", default=1000, type=int, help="data size to inference")
    # remove conflict with ianvs
    parser.add_argument("-f")
    return parser


@ClassFactory.register(ClassType.GENERAL, alias="Classification")
class BaseModel:

    def __init__(self, **kwargs) -> None:
        self.args = make_parser().parse_args()
        self.devices_info_url = str(Path(Path(__file__).parent.resolve(), self.args.devices_info))
        self.model_parallel = self.args.model_parallel
        self.partition_point_list = self._parse_devices_info(self.devices_info_url).get('partition_points')
        self.models = []
        return


    def load(self, models_dir=None, map_info=None) -> None:
        cnt = 0
        for model_name, device in map_info.items():
            model = models_dir + '/' + model_name
            if not os.path.exists(model):
                raise ValueError("=> No modle found at '{}'".format(model))
            if device == 'cpu':
                session = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
            elif 'gpu' in device:
                device_id = int(device.split('-')[-1])
                session = ort.InferenceSession(model, providers=[('CUDAExecutionProvider', {'device_id': device_id})])
            else:
                raise ValueError("Error device info: '{}'".format(device))
            self.models.append({
                'session': session,
                'name': model_name,
                'device': device,
                'input_names': self.partition_point_list[cnt]['input_names'],
                'output_names': self.partition_point_list[cnt]['output_names'],
            })
            cnt += 1
            print("=> Loaded onnx model: '{}'".format(model))
        return

    def predict(self, data, input_shape=None, **kwargs):
        pynvml.nvmlInit()
        root = str(Path(data[0]).parents[2])
        dataset_cfg = {
            'name': self.args.dataset_name,
            'root': root,
            'split': self.args.split,
            'indices': self.args.indices,
            'shuffle': self.args.shuffle
        }
        data_loader, ids = self._get_eval_loader(dataset_cfg)
        data_loader = tqdm(data_loader, desc='Evaluating', unit='batch')
        pred = []
        inference_time_per_device = defaultdict(int)
        power_usage_per_device = defaultdict(list)
        mem_usage_per_device = defaultdict(list)
        cnt = 0
        for data, id in zip(data_loader, ids):
            outputs = data[0].numpy()
            for model in self.models:
                start_time = time.time()
                outputs = model['session'].run(None, {model['input_names'][0]: outputs})[0]
                end_time = time.time()
                device = model.get('device')
                inference_time_per_device[device] += end_time - start_time
                if 'gpu' in device and cnt % 100 == 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(int(device.split('-')[-1]))
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2)
                    power_usage_per_device[device] += [power_usage]
                    mem_usage_per_device[device] += [memory_info]
            max_ids = np.argmax(outputs)
            pred.append((max_ids, id))
            cnt += 1
        data_loader.close()
        result = dict({})
        result["pred"] = pred
        result["inference_time_per_device"] = inference_time_per_device
        result["power_usage_per_device"] = power_usage_per_device
        result["mem_usage_per_device"] = mem_usage_per_device
        return result


    def _get_eval_loader(self, dataset_cfg):
        model_name = self.args.model_name
        data_size = self.args.data_size
        dataset, _, ids = load_dataset(dataset_cfg, model_name, data_size)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        return data_loader, ids

    def _parse_devices_info(self, url):
        """Convert yaml file to the dict."""
        if url.endswith('.yaml') or url.endswith('.yml'):
            with open(url, "rb") as file:
                devices_info_dict = yaml.load(file, Loader=yaml.SafeLoader)
                return devices_info_dict
        else:
            raise RuntimeError('config file must be the yaml format')