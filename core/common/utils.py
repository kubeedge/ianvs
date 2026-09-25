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
import importlib.util
import os
import sys
import time

from importlib import import_module
from inspect import getfullargspec
import yaml


def is_local_file(url):
    """Check if the url is a file and already exists locally."""
    return os.path.isfile(url)


def is_local_dir(url):
    """Check if the url is a dir and already exists locally."""
    return os.path.isdir(url)


def get_file_format(url):
    """Get file format of the url."""
    # Check if the url
    if os.path.basename(url) == "metadata.json":
        return "jsonforllm"

    # Check if the url
    return os.path.splitext(url)[-1][1:]

def parse_kwargs(func, **kwargs):
    """Get valid parameters of the func in kwargs."""
    if not callable(func):
        return kwargs
    need_kw = getfullargspec(func)
    if need_kw.varkw == 'kwargs':
        return kwargs
    return {k: v for k, v in kwargs.items() if k in need_kw.args}


def get_local_time():
    """Get local time."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def py2dict(url):
    """Convert py file to the dict."""
    if url.endswith('.py'):
        if not os.path.isfile(url):
            raise RuntimeError(f"config file({url}) does not exist")
        module_name = os.path.basename(url)[:-3]
        try:
            spec = importlib.util.spec_from_file_location(module_name, url)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"load spec failed for {url}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            raw_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
            }
            return raw_dict
        except Exception as err:
            raise RuntimeError(f"convert py file({url}) to dict failed, error: {err}") from err

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

    try:
        if os.path.isfile(url):
            abs_url = os.path.abspath(url)
            cached_mod = sys.modules.get(module_name)
            if cached_mod is not None and getattr(cached_mod, "__file__", None) == abs_url:
                return cached_mod

            spec = importlib.util.spec_from_file_location(module_name, abs_url)
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)
                return mod

        sys.path.insert(0, module_path)
        try:
            mod = import_module(module_name)
            return mod
        finally:
            if module_path in sys.path:
                sys.path.remove(module_path)
    except Exception as err:
        raise RuntimeError(f"load module(url={url}) failed, error: {err}") from err
