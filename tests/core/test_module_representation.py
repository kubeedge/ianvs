# Copyright 2026 The KubeEdge Authors.
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

"""Pins S2: what Module.get_module_instance returns
(core/testcasecontroller/algorithm/module/module.py).

Sedna 0.6.0.1's LifelongLearning resolves its modules from
{"method": ..., "param": ...} dicts. Basemodel must stay an instance, because
single-task returns it directly and incremental and joint inference pass it as
`estimator=`.

Preservation tests pass on the unmodified code and must still pass after S2.
Change tests fail on the unmodified code and pass after S2.
"""

import itertools

import pytest
from sedna.common.class_factory import ClassFactory, ClassType

from core.testcasecontroller.algorithm.module.module import Module

_COUNTER = itertools.count()

_MODULE_SOURCE = """
from sedna.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.{class_type}, alias="{alias}")
class Impl:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
"""


@pytest.fixture(name="make_module")
def fixture_make_module(tmp_path):
    """Write a module file that registers a class, and build a Module for it."""

    def make(module_type, class_type, hyperparameters=None):
        alias = f"S2Test{next(_COUNTER)}"
        path = tmp_path / f"{alias.lower()}.py"
        path.write_text(_MODULE_SOURCE.format(class_type=class_type, alias=alias),
                        encoding="utf-8")
        module = Module({"type": module_type, "name": alias, "url": str(path)})
        module.hyperparameters = dict(hyperparameters or {})
        return module, alias

    return make


# Preservation tests


def test_basemodel_is_still_an_instance(make_module):
    module, _ = make_module("basemodel", "GENERAL", {"lr": 0.1})

    instance = module.get_module_instance("basemodel")

    assert not isinstance(instance, dict)
    assert instance.kwargs == {"lr": 0.1}


def test_hard_example_mining_is_still_a_dict(make_module):
    module, alias = make_module("hard_example_mining", "HEM", {"threshold": 0.5})

    func = module.get_module_instance("hard_example_mining")

    assert func == {"method": alias, "param": {"threshold": 0.5}}


def test_builtin_module_without_url_is_still_a_dict():
    module = Module({"type": "task_definition", "name": "TaskDefinitionByDataAttr"})
    module.hyperparameters = {"attribute": ["season"]}

    func = module.get_module_instance("task_definition")

    assert func == {"method": "TaskDefinitionByDataAttr",
                    "param": {"attribute": ["season"]}}


# Change tests


@pytest.mark.parametrize("module_type, class_type", [
    ("task_definition", "STP"),
    ("task_relationship_discovery", "STP"),
    ("task_allocation", "STP"),
    ("task_remodeling", "STP"),
    ("inference_integrate", "STP"),
    ("task_update_decision", "KM"),
    ("unseen_task_allocation", "UTP"),
    ("unseen_sample_recognition", "UTD"),
    ("unseen_sample_re_recognition", "UTD"),
])
def test_lifelong_modules_with_url_are_dicts(make_module, module_type, class_type):
    module, alias = make_module(module_type, class_type, {"origins": ["real", "sim"]})

    func = module.get_module_instance(module_type)

    assert func == {"method": alias, "param": {"origins": ["real", "sim"]}}
    # the class is still registered, so Sedna can build it from the dict
    assert ClassFactory.is_exists(getattr(ClassType, class_type), alias)
