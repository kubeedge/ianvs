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

import importlib
import os
import sys
import time
import yaml
from importlib import import_module
from inspect import getfullargspec


def is_local_file(url):
    """ check if the url is a file and already exists locally """
    if not os.path.isfile(url):
        return False
    if not os.path.exists(url):
        return False
    return True


def get_file_format(url):
    """ get format of file """
    return os.path.splitext(url)[-1][1:]


def parse_kwargs(func, **kwargs):
    """ get valid parameters in kwargs """
    if not callable(func):
        return kwargs
    need_kw = getfullargspec(func)
    if need_kw.varkw == 'kwargs':
        return kwargs
    return {k: v for k, v in kwargs.items() if k in need_kw.args}


def get_local_time():
    """ get local time """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def py2dict(url):
    """ convert py file to the dict """
    if url.endswith('.py'):
        module_name = os.path.basename(url)[:-3]
        config_dir = os.path.dirname(url)
        sys.path.insert(0, config_dir)
        mod = import_module(module_name)
        sys.path.pop(0)
        raw_dict = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith('__')
        }
        sys.modules.pop(module_name)
    else:
        raise Exception('config file must be the py format')
    return raw_dict


def yaml2dict(url):
    """ convert yaml file to the dict """
    if url.endswith('.yaml') or url.endswith('.yml'):
        with open(url, "rb") as f:
            raw_dict = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        raise Exception('config file must be the yaml format')
    return raw_dict


def load_module(url):
    """ load python module"""
    module_path, module_name = os.path.split(url)
    if os.path.isfile(url):
        module_name = module_name.split(".")[0]

    sys.path.insert(0, module_path)
    try:
        importlib.import_module(module_name)
        sys.path.pop(0)
    except Exception as err:
        raise Exception(
            f"load module(url={url}) failed, error: {err}")
