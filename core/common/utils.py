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

"""This script contains some common tools."""

import importlib
import os
import sys
import time

import inspect
from importlib import import_module
import yaml


def is_local_file(url):
    """Check if the url is a file and already exists locally."""
    return os.path.isfile(url)


def is_local_dir(url):
    """Check if the url is a directory and already exists locally."""
    return os.path.isdir(url)


def get_file_format(url):
    """Get file format."""
    if os.path.basename(url) == "data_info.json":
        return "json"

    if os.path.basename(url) == "metadata.json":
        return "jsonforllm"

    # Check if the url
    return os.path.splitext(url)[-1][1:]


def parse_kwargs(func, **kwargs):
    """Get valid parameters of the func in kwargs."""
    if not callable(func):
        return kwargs
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return kwargs

    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return kwargs

    valid_args = {
        name for name, param in sig.parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    if not valid_args:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in valid_args}


def get_local_time():
    """Get local time."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def py2dict(url):
    """Convert py file to the dict."""
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

        return raw_dict

    raise RuntimeError('config file must be the py format')


def yaml2dict(url):
    """Convert yaml file to the dict."""
    if url.endswith('.yaml') or url.endswith('.yml'):
        with open(url, "rb") as file:
            raw_dict = yaml.load(file, Loader=yaml.SafeLoader)

        return raw_dict

    raise RuntimeError('config file must be the yaml format')


def load_module(url):
    """Load python module."""
    module_path, module_name = os.path.split(url)
    if os.path.isfile(url):
        module_name = module_name.split(".")[0]

    sys.path.insert(0, module_path)
    try:
        importlib.import_module(module_name)
        sys.path.pop(0)
    except Exception as err:
        raise RuntimeError(f"load module(url={url}) failed, error: {err}") from err
