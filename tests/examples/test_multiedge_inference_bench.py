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

"""Tests for hyperparameter propagation in imagenet/multiedge_inference_bench (#613)."""

import copy
import os
from pathlib import Path
import sys
import types

import pytest
import yaml
from sedna.common.class_factory import ClassFactory

from core.cmd.obj.benchmarkingjob import BenchmarkingJob
from core.common import utils
from core.storymanager.rank.rank import Rank
from core.testcasecontroller.algorithm import Algorithm
from core.testcasecontroller.algorithm.paradigm.multiedge_inference.multiedge_inference import (
    MultiedgeInference,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "examples" / "imagenet" / "multiedge_inference_bench"
AUTO_DIR = BENCH_DIR / "testalgorithms" / "automatic"
MANUAL_DIR = BENCH_DIR / "testalgorithms" / "manual"
AUTO_ALGO_YAML = AUTO_DIR / "classification_algorithm.yaml"
MANUAL_ALGO_YAML = MANUAL_DIR / "classification_algorithm.yaml"


def _make_stub_module(name, **attrs):
    mod = types.ModuleType(name)
    for key, val in attrs.items():
        setattr(mod, key, val)
    return mod


def _is_dynamic_example_module(mod_name: str) -> bool:
    return (
        mod_name in ("basemodel", "dataset", "model_cfg", "devices")
        or mod_name == "models"
        or mod_name.startswith("models.")
    )


@pytest.fixture(autouse=True)
def _isolate_ml_and_registry(monkeypatch):
    """Stub heavy ML deps and isolate dynamic imports and ClassFactory state."""
    saved_registry = copy.deepcopy(ClassFactory.__registry__)
    saved_sys_path = list(sys.path)
    saved_backend_type = os.environ.get("BACKEND_TYPE")
    saved_dynamic_modules = {
        mod_name: sys.modules.pop(mod_name)
        for mod_name in list(sys.modules)
        if _is_dynamic_example_module(mod_name)
    }

    class _DummyModule:
        pass

    torch_nn = _make_stub_module("torch.nn", Module=_DummyModule, ModuleList=list, LayerNorm=_DummyModule, Linear=_DummyModule, Identity=_DummyModule)
    torch_utils_data = _make_stub_module("torch.utils.data", DataLoader=lambda *a, **kw: ("loader", a, kw), Dataset=object, Subset=object)
    torch_utils = _make_stub_module("torch.utils", data=torch_utils_data)
    torch_onnx = _make_stub_module("torch.onnx", export=lambda *a, **kw: None)
    torch_mod = _make_stub_module(
        "torch",
        nn=torch_nn,
        Tensor=object,
        tensor=object,
        no_grad=lambda: (lambda fn: fn),
        randn=lambda *shape: ("randn", shape),
        device=lambda d: d,
        utils=torch_utils,
        onnx=torch_onnx,
    )

    tv_datasets = _make_stub_module("torchvision.datasets", ImageNet=object, ImageFolder=object)
    tv_mod = _make_stub_module("torchvision", datasets=tv_datasets)

    modeling_bert = _make_stub_module(
        "transformers.models.bert.modeling_bert",
        BertEmbeddings=object,
        BertIntermediate=object,
        BertOutput=object,
        BertPooler=object,
        BertSelfAttention=object,
        BertSelfOutput=object,
    )
    modeling_deit = _make_stub_module("transformers.models.deit.modeling_deit", DeiTEmbeddings=object)
    modeling_vit = _make_stub_module(
        "transformers.models.vit.modeling_vit",
        ViTEmbeddings=object,
        ViTAttention=object,
        ViTMLP=object,
    )
    tf_models = _make_stub_module("transformers.models")
    tf_models_bert = _make_stub_module("transformers.models.bert", modeling_bert=modeling_bert)
    tf_models_deit = _make_stub_module("transformers.models.deit", modeling_deit=modeling_deit)
    tf_models_vit = _make_stub_module("transformers.models.vit", modeling_vit=modeling_vit)
    transformers_mod = _make_stub_module(
        "transformers",
        AutoConfig=object,
        AutoImageProcessor=object,
        BertConfig=object,
        BertForSequenceClassification=object,
        BertModel=object,
        DeiTConfig=object,
        ViTConfig=object,
        models=tf_models,
    )

    ort_mod = _make_stub_module("onnxruntime", get_available_providers=lambda: ["CPUExecutionProvider"], InferenceSession=object)

    stubs = {
        "torch": torch_mod,
        "torch.nn": torch_nn,
        "torch.utils": torch_utils,
        "torch.utils.data": torch_utils_data,
        "torch.onnx": torch_onnx,
        "torchvision": tv_mod,
        "torchvision.datasets": tv_datasets,
        "transformers": transformers_mod,
        "transformers.models": tf_models,
        "transformers.models.bert": tf_models_bert,
        "transformers.models.bert.modeling_bert": modeling_bert,
        "transformers.models.deit": tf_models_deit,
        "transformers.models.deit.modeling_deit": modeling_deit,
        "transformers.models.vit": tf_models_vit,
        "transformers.models.vit.modeling_vit": modeling_vit,
        "onnxruntime": ort_mod,
    }
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)

    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(sys, "argv", ["ianvs", "-f", "classification_job.yaml"])

    try:
        yield
    finally:
        for mod_name in list(sys.modules):
            if _is_dynamic_example_module(mod_name):
                sys.modules.pop(mod_name, None)
        sys.modules.update(saved_dynamic_modules)
        sys.path[:] = saved_sys_path
        ClassFactory.__registry__.clear()
        ClassFactory.__registry__.update(saved_registry)
        if saved_backend_type is None:
            os.environ.pop("BACKEND_TYPE", None)
        else:
            os.environ["BACKEND_TYPE"] = saved_backend_type


def _write_profiler_yml(path: Path, layers: int = 2) -> Path:
    data = {
        "model_name": "google/vit-base-patch16-224",
        "dtype": "torch.float32",
        "batch_size": 1,
        "layers": layers,
        "profile_data": [
            {
                "layer": idx + 1,
                "memory": 10.0,
                "shape_in": [[3, 224, 224]] if idx == 0 else [[197, 768]],
                "shape_out": [[197, 768]] if idx == 0 else [[1000]],
                "time": 0.01,
            }
            for idx in range(layers)
        ],
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _write_auto_devices_yml(path: Path, count: int = 2) -> Path:
    data = {
        "devices": [
            {
                "name": f"gpu-{idx}",
                "type": "gpu",
                "memory": "1024",
                "freq": "2.6",
                "bandwith": "100",
            }
            for idx in range(count)
        ]
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _load_basemodel_module(variant: str):
    target_dir = AUTO_DIR if variant == "automatic" else MANUAL_DIR
    utils.load_module(str(target_dir / "basemodel.py"))
    return sys.modules["basemodel"]


def test_algorithm_yamls_expose_model_name_and_devices_info():
    for yaml_path in (AUTO_ALGO_YAML, MANUAL_ALGO_YAML):
        cfg = utils.yaml2dict(str(yaml_path))
        algo = Algorithm("classification", copy.deepcopy(cfg))
        assert len(algo.modules_list) == 1
        basemodel_mod = algo.modules_list[0]["basemodel"]
        assert basemodel_mod.hyperparameters == {
            "batch_size": 1,
            "model_name": "google/vit-base-patch16-224",
            "devices_info": "./devices.yaml",
        }


def test_ianvs_algorithm_propagates_hyperparameters_to_manual_basemodel(tmp_path):
    cfg = utils.yaml2dict(str(MANUAL_ALGO_YAML))
    cfg["algorithm"]["modules"][0]["hyperparameters"] = [
        {"batch_size": {"values": [1]}},
        {"model_name": {"values": ["facebook/deit-base-distilled-patch16-224"]}},
        {"devices_info": {"values": ["./devices.yaml", "./devices_one.yaml"]}},
    ]
    algo = Algorithm("classification", cfg)
    assert len(algo.modules_list) == 2

    instances = [mods["basemodel"].get_module_instance("basemodel") for mods in algo.modules_list]
    assert [inst.model_name for inst in instances] == [
        "facebook/deit-base-distilled-patch16-224",
        "facebook/deit-base-distilled-patch16-224",
    ]
    assert [inst.args.model_name for inst in instances] == [
        "facebook/deit-base-distilled-patch16-224",
        "facebook/deit-base-distilled-patch16-224",
    ]
    assert len(instances[0].partition_point_list) == 3
    assert len(instances[1].partition_point_list) == 1
    assert instances[1].partition_point_list[0]["device_name"] == "gpu-1"
    assert Path(instances[1].devices_info_url).name == "devices_one.yaml"


def test_ianvs_algorithm_propagates_hyperparameters_to_automatic_basemodel(tmp_path):
    profiler_yml = _write_profiler_yml(tmp_path / "profiler_results.yml")
    devices_yml = _write_auto_devices_yml(tmp_path / "devices_two.yaml", count=2)

    cfg = utils.yaml2dict(str(AUTO_ALGO_YAML))
    cfg["algorithm"]["modules"][0]["hyperparameters"] = [
        {"batch_size": {"values": [1]}},
        {"model_name": {"values": ["google/vit-large-patch16-224"]}},
        {"devices_info": {"values": [str(devices_yml)]}},
        {"profiler_info": {"values": [str(profiler_yml)]}},
    ]
    algo = Algorithm("classification", cfg)
    assert len(algo.modules_list) == 1

    inst = algo.modules_list[0]["basemodel"].get_module_instance("basemodel")
    assert inst.model_name == "google/vit-large-patch16-224"
    assert inst.args.model_name == "google/vit-large-patch16-224"
    assert inst.devices_info == str(devices_yml)
    assert len(inst.device_info["devices"]) == 2


def test_automatic_partition_and_eval_loader_use_configured_model_name(tmp_path, monkeypatch):
    profiler_yml = _write_profiler_yml(tmp_path / "profiler_results.yml", layers=2)
    devices_yml = _write_auto_devices_yml(tmp_path / "devices_one.yaml", count=1)

    bm_mod = _load_basemodel_module("automatic")
    shard_calls = []
    export_calls = []
    dataset_calls = []

    class _DummyShard:
        def eval(self):
            return None

    def _fake_shard_factory(model_name, model_file, layer_start, layer_end, stage):
        shard_calls.append((model_name, model_file, layer_start, layer_end, stage))
        return _DummyShard()

    def _fake_onnx_export(model, dummy_input, out_path, **kwargs):
        export_calls.append(out_path)

    def _fake_load_dataset(dataset_cfg, model_name, data_size):
        dataset_calls.append((dataset_cfg, model_name, data_size))
        return ([], [], [])

    monkeypatch.setattr(bm_mod.model_cfg, "module_shard_factory", _fake_shard_factory)
    monkeypatch.setattr(bm_mod.torch.onnx, "export", _fake_onnx_export)
    monkeypatch.setattr(bm_mod, "load_dataset", _fake_load_dataset)

    inst = bm_mod.BaseModel(
        model_name="facebook/deit-base-distilled-patch16-224",
        devices_info=str(devices_yml),
        profiler_info=str(profiler_yml),
        data_size=25,
    )
    init_weights = tmp_path / "DeiT_B_distilled.npz"
    init_weights.write_bytes(b"")
    models_dir, map_info = inst.partition(str(init_weights))

    assert models_dir == str(tmp_path.resolve())
    assert map_info == {"sub_model_1.onnx": "gpu-0"}
    assert shard_calls == [("facebook/deit-base-distilled-patch16-224", str(init_weights), 1, 2, 1)]
    assert len(export_calls) == 1

    inst._get_eval_loader({"name": "ImageNet", "root": ".", "split": "val", "indices": None, "shuffle": False})
    assert dataset_calls[0][1] == "facebook/deit-base-distilled-patch16-224"
    assert dataset_calls[0][2] == 25


def test_manual_eval_loader_uses_configured_model_name(monkeypatch):
    bm_mod = _load_basemodel_module("manual")
    dataset_calls = []

    def _fake_load_dataset(dataset_cfg, model_name, data_size):
        dataset_calls.append((dataset_cfg, model_name, data_size))
        return ([], [], [])

    monkeypatch.setattr(bm_mod, "load_dataset", _fake_load_dataset)

    inst = bm_mod.BaseModel(
        model_name="bert-base-uncased",
        devices_info="./devices_one.yaml",
        data_size=12,
    )
    inst._get_eval_loader({"name": "ImageNet", "root": ".", "split": "val", "indices": None, "shuffle": False})
    assert dataset_calls[0][1] == "bert-base-uncased"
    assert dataset_calls[0][2] == 12
    assert len(inst.partition_point_list) == 1


@pytest.mark.parametrize(
    "argv",
    [
        ["ianvs", "-f", "./examples/imagenet/multiedge_inference_bench/classification_job_manual.yaml"],
        ["ianvs", "--benchmarking_config_file", "job.yaml"],
        ["pytest", "-v", "-k", "test_multiedge", "--tb=short"],
        ["unittest", "-f"],
    ],
)
def test_unrelated_cli_flags_do_not_crash_basemodel(argv, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", argv)
    bm_mod = _load_basemodel_module("manual")
    inst = bm_mod.BaseModel(model_name="google/vit-large-patch16-224", devices_info="./devices_one.yaml")
    assert inst.model_name == "google/vit-large-patch16-224"
    assert len(inst.partition_point_list) == 1


def test_cli_args_used_as_fallback_when_kwargs_omitted(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["basemodel.py", "--model_name", "facebook/deit-tiny-distilled-patch16-224", "--devices_info", "./devices_one.yaml"],
    )
    bm_mod = _load_basemodel_module("manual")
    inst = bm_mod.BaseModel()
    assert inst.model_name == "facebook/deit-tiny-distilled-patch16-224"
    assert inst.args.devices_info == "./devices_one.yaml"
    assert len(inst.partition_point_list) == 1


def test_none_and_blank_hyperparameters_preserve_defaults():
    bm_mod = _load_basemodel_module("manual")
    inst = bm_mod.BaseModel(model_name="   ", devices_info=None, split="", data_size=None)
    assert inst.model_name == "google/vit-base-patch16-224"
    assert inst.devices_info == "./devices.yaml"
    assert inst.args.split == "val"
    assert inst.args.data_size == 1000
    assert len(inst.partition_point_list) == 3


def test_repo_root_relative_devices_info_path():
    bm_mod = _load_basemodel_module("manual")
    inst = bm_mod.BaseModel(
        devices_info="./examples/imagenet/multiedge_inference_bench/testalgorithms/manual/devices_one.yaml"
    )
    assert len(inst.partition_point_list) == 1
    assert Path(inst.devices_info_url).resolve() == (MANUAL_DIR / "devices_one.yaml").resolve()


def test_boolean_string_and_invalid_hyperparameter_validation(tmp_path):
    bm_mod = _load_basemodel_module("manual")

    inst = bm_mod.BaseModel(model_parallel=False, shuffle="true", data_size="42")
    assert inst.model_parallel is False
    assert inst.args.shuffle is True
    assert inst.args.data_size == 42

    inst_str_false = bm_mod.BaseModel(model_parallel="false")
    assert inst_str_false.model_parallel is False

    with pytest.raises(ValueError, match="model_name"):
        bm_mod.BaseModel(model_name=123)

    with pytest.raises(ValueError, match="model_parallel"):
        bm_mod.BaseModel(model_parallel="maybe")

    with pytest.raises(ValueError, match="data_size"):
        bm_mod.BaseModel(data_size=True)

    with pytest.raises(ValueError, match="data_size"):
        bm_mod.BaseModel(data_size="not_an_int")

    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="YAML mapping"):
        bm_mod.BaseModel(devices_info=str(empty_yaml))


def test_rank_extracts_model_name_and_devices_info_hyperparameters():
    cfg = utils.yaml2dict(str(MANUAL_ALGO_YAML))
    algo = Algorithm("classification", cfg)
    algo.modules = algo.modules_list[0]
    hps = Rank._get_algorithm_hyperparameters(algo)
    assert hps == {
        "basemodel-batch_size": "1",
        "basemodel-model_name": "google/vit-base-patch16-224",
        "basemodel-devices_info": "./devices.yaml",
    }


def test_automatic_cli_fallback_when_kwargs_omitted(tmp_path, monkeypatch):
    profiler_yml = _write_profiler_yml(tmp_path / "profiler_results.yml")
    devices_yml = _write_auto_devices_yml(tmp_path / "devices.yaml", count=1)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "basemodel.py",
            "--model_name",
            "bert-base-uncased",
            "--devices_info",
            str(devices_yml),
            "--profiler_info",
            str(profiler_yml),
        ],
    )
    bm_mod = _load_basemodel_module("automatic")
    inst = bm_mod.BaseModel()
    assert inst.model_name == "bert-base-uncased"
    assert inst.args.model_name == "bert-base-uncased"
    assert len(inst.device_info["devices"]) == 1


def test_multiedge_inference_paradigm_manual_single_device_e2e(tmp_path, monkeypatch):
    cfg = utils.yaml2dict(str(MANUAL_ALGO_YAML))
    cfg["algorithm"]["modules"][0]["hyperparameters"] = [
        {"batch_size": {"values": [1]}},
        {"model_name": {"values": ["facebook/deit-base-distilled-patch16-224"]}},
        {"devices_info": {"values": ["./devices_one.yaml"]}},
    ]
    init_model = tmp_path / "model.onnx"
    init_model.write_bytes(b"")
    cfg["algorithm"]["initial_model_url"] = str(init_model)

    algo = Algorithm("classification", cfg)
    modules = algo.modules_list[0]

    extracted = []

    def _fake_extract(src, dst, input_names, output_names):
        extracted.append((src, dst, input_names, output_names))
        Path(dst).write_bytes(b"submodel")

    monkeypatch.setattr("onnx.utils.extract_model", _fake_extract)

    class _FakeDatasetObj:
        x = [str(tmp_path / "val" / "0" / "img.JPEG")]

    class _FakeDataset:
        train_url = "train.txt"
        test_url = "test.txt"

        @staticmethod
        def load_data(url, mode):
            return _FakeDatasetObj()

    class _FakeSession:
        def __init__(self, model_path, providers=None):
            self.model_path = model_path
            self.providers = providers

        def run(self, _outputs, inputs):
            return [[[0.1, 0.9]]]

    paradigm = MultiedgeInference(
        workspace=str(tmp_path / "ws"),
        modules=modules,
        dataset=_FakeDataset(),
        initial_model_url=str(init_model),
    )

    bm_mod = sys.modules["basemodel"]
    monkeypatch.setattr(bm_mod.ort, "InferenceSession", _FakeSession)
    seen_models = []

    def _fake_load_dataset(dataset_cfg, model_name, data_size):
        seen_models.append(model_name)
        return ([], [], [0])

    class _DummyTensor:
        def numpy(self):
            return "dummy"

    monkeypatch.setattr(bm_mod, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(bm_mod, "DataLoader", lambda ds, **kw: [(_DummyTensor(),)])

    res, _ = paradigm.run()
    assert len(extracted) == 1
    assert extracted[0][2:] == (["pixel_values"], ["logits"])
    assert seen_models == ["facebook/deit-base-distilled-patch16-224"]
    assert res["pred"] == [(1, 0)]


def test_multiedge_inference_paradigm_automatic_e2e(tmp_path, monkeypatch):
    profiler_yml = _write_profiler_yml(tmp_path / "profiler_results.yml", layers=4)
    devices_yml = _write_auto_devices_yml(tmp_path / "devices.yaml", count=2)
    init_weights = tmp_path / "DeiT_B_distilled.npz"
    init_weights.write_bytes(b"")

    cfg = utils.yaml2dict(str(AUTO_ALGO_YAML))
    cfg["algorithm"]["initial_model_url"] = str(init_weights)
    cfg["algorithm"]["modules"][0]["hyperparameters"] = [
        {"batch_size": {"values": [2]}},
        {"model_name": {"values": ["facebook/deit-base-distilled-patch16-224"]}},
        {"devices_info": {"values": [str(devices_yml)]}},
        {"profiler_info": {"values": [str(profiler_yml)]}},
    ]
    algo = Algorithm("classification", cfg)
    modules = algo.modules_list[0]

    class _FakeDatasetObj:
        x = [str(tmp_path / "val" / "0" / "img.JPEG")]

    class _FakeDataset:
        train_url = "train.txt"
        test_url = "test.txt"

        @staticmethod
        def load_data(url, mode):
            return _FakeDatasetObj()

    class _DummyShard:
        def eval(self):
            return None

    class _FakeSession:
        def __init__(self, model_path, providers=None):
            self.model_path = model_path
            self.providers = providers

        def run(self, _outputs, inputs):
            return [[[0.2, 0.8]]]

    class _DummyTensor:
        def numpy(self):
            return "dummy"

    paradigm = MultiedgeInference(
        workspace=str(tmp_path / "ws"),
        modules=modules,
        dataset=_FakeDataset(),
        initial_model_url=str(init_weights),
    )

    bm_mod = sys.modules["basemodel"]
    shard_calls = []
    seen_models = []

    def _fake_shard_factory(model_name, model_file, layer_start, layer_end, stage):
        shard_calls.append((model_name, model_file, layer_start, layer_end, stage))
        return _DummyShard()

    def _fake_onnx_export(model, dummy_input, out_path, **kwargs):
        Path(out_path).write_bytes(b"submodel")

    def _fake_load_dataset(dataset_cfg, model_name, data_size):
        seen_models.append(model_name)
        return ([], [], [0])

    monkeypatch.setattr(bm_mod.model_cfg, "module_shard_factory", _fake_shard_factory)
    monkeypatch.setattr(bm_mod.torch.onnx, "export", _fake_onnx_export)
    monkeypatch.setattr(bm_mod.ort, "InferenceSession", _FakeSession)
    monkeypatch.setattr(bm_mod, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(bm_mod, "DataLoader", lambda ds, **kw: [(_DummyTensor(),)])

    res, _ = paradigm.run()
    assert len(shard_calls) == 2
    assert all(call[0] == "facebook/deit-base-distilled-patch16-224" for call in shard_calls)
    assert seen_models == ["facebook/deit-base-distilled-patch16-224"]
    assert set(res["inference_time_per_device"]) == {"gpu-0", "gpu-1"}
    assert res["pred"] == [(1, 0)]


def test_kwargs_override_cli_flags_in_both_variants(tmp_path, monkeypatch):
    profiler_yml = _write_profiler_yml(tmp_path / "profiler_results.yml")
    devices_yml = _write_auto_devices_yml(tmp_path / "devices.yaml", count=1)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "basemodel.py",
            "--model_name",
            "bert-base-uncased",
            "--devices_info",
            "./devices.yaml",
            "--profiler_info",
            str(profiler_yml),
        ],
    )

    manual_mod = _load_basemodel_module("manual")
    manual_inst = manual_mod.BaseModel(
        model_name="google/vit-large-patch16-224",
        devices_info="./devices_one.yaml",
    )
    assert manual_inst.model_name == "google/vit-large-patch16-224"
    assert manual_inst.devices_info == "./devices_one.yaml"
    assert len(manual_inst.partition_point_list) == 1

    for mod_name in list(sys.modules):
        if _is_dynamic_example_module(mod_name):
            sys.modules.pop(mod_name, None)

    auto_mod = _load_basemodel_module("automatic")
    auto_inst = auto_mod.BaseModel(
        model_name="facebook/deit-small-distilled-patch16-224",
        devices_info=devices_yml,
        profiler_info=profiler_yml,
    )
    assert auto_inst.model_name == "facebook/deit-small-distilled-patch16-224"
    assert auto_inst.devices_info == str(devices_yml)
    assert len(auto_inst.device_info["devices"]) == 1


def test_batch_size_data_size_and_config_error_paths(tmp_path):
    bm_mod = _load_basemodel_module("manual")

    inst_default = bm_mod.BaseModel(batch_size=None)
    assert inst_default.batch_size == 1

    inst_str = bm_mod.BaseModel(batch_size="4")
    assert inst_str.batch_size == 4

    for bad_batch in (True, 1.5, 0, -2, "not_int"):
        with pytest.raises(ValueError, match="batch_size"):
            bm_mod.BaseModel(batch_size=bad_batch)

    for bad_data_size in (True, 1.5, 0, -5, "not_int"):
        with pytest.raises(ValueError, match="data_size"):
            bm_mod.BaseModel(data_size=bad_data_size)

    with pytest.raises(FileNotFoundError):
        bm_mod.BaseModel(devices_info=str(tmp_path / "does_not_exist.yaml"))

    txt_file = tmp_path / "devices.txt"
    txt_file.write_text("devices: []\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="yaml format"):
        bm_mod.BaseModel(devices_info=str(txt_file))


def test_benchmarking_job_manual_e2e_writes_hyperparameters_to_rank(tmp_path, monkeypatch):
    img_dir = tmp_path / "dataset" / "val" / "0"
    img_dir.mkdir(parents=True)
    img_path = img_dir / "img.JPEG"
    img_path.write_bytes(b"")

    train_txt = tmp_path / "train.txt"
    test_txt = tmp_path / "test.txt"
    train_txt.write_text(f"{img_path} 0\n", encoding="utf-8")
    test_txt.write_text(f"{img_path} 0\n", encoding="utf-8")

    metric_py = tmp_path / "multiedge_test_accuracy.py"
    metric_py.write_text(
        "from sedna.common.class_factory import ClassType, ClassFactory\n"
        "@ClassFactory.register(ClassType.GENERAL, alias='accuracy')\n"
        "def accuracy(y_true, y_pred, **kwargs):\n"
        "    return 100.0\n",
        encoding="utf-8",
    )

    testenv_yaml = tmp_path / "testenv.yaml"
    testenv_yaml.write_text(
        yaml.safe_dump(
            {
                "testenv": {
                    "dataset": {
                        "train_index": str(train_txt),
                        "test_index": str(test_txt),
                    },
                    "metrics": [{"name": "accuracy", "url": str(metric_py)}],
                }
            }
        ),
        encoding="utf-8",
    )

    init_model = tmp_path / "vit.onnx"
    init_model.write_bytes(b"")

    algo_cfg = utils.yaml2dict(str(MANUAL_ALGO_YAML))
    algo_cfg["algorithm"]["initial_model_url"] = str(init_model)
    algo_cfg["algorithm"]["modules"][0]["hyperparameters"] = [
        {"batch_size": {"values": [1]}},
        {"model_name": {"values": ["facebook/deit-base-distilled-patch16-224"]}},
        {"devices_info": {"values": ["./devices_one.yaml"]}},
    ]
    algo_yaml = tmp_path / "classification_algorithm.yaml"
    algo_yaml.write_text(yaml.safe_dump(algo_cfg), encoding="utf-8")

    job_cfg = utils.yaml2dict(str(BENCH_DIR / "classification_job_manual.yaml"))
    job_cfg["benchmarkingjob"]["workspace"] = str(tmp_path / "workspace")
    job_cfg["benchmarkingjob"]["testenv"] = str(testenv_yaml)
    job_cfg["benchmarkingjob"]["test_object"]["algorithms"][0]["url"] = str(algo_yaml)

    monkeypatch.setattr(sys, "argv", ["ianvs", "-f", str(tmp_path / "job.yaml")])
    monkeypatch.setattr("onnx.utils.extract_model", lambda s, d, i, o: Path(d).write_bytes(b"sub"))

    bm_mod = _load_basemodel_module("manual")

    class _FakeSession:
        def __init__(self, model_path, providers=None):
            pass

        def run(self, _outputs, inputs):
            return [[[0.1, 0.9]]]

    class _DummyTensor:
        def numpy(self):
            return "dummy"

    seen_models = []

    def _fake_load_dataset(dataset_cfg, model_name, data_size):
        seen_models.append(model_name)
        return ([], [], [0])

    monkeypatch.setattr(bm_mod.ort, "InferenceSession", _FakeSession)
    monkeypatch.setattr(bm_mod, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(bm_mod, "DataLoader", lambda ds, **kw: [(_DummyTensor(),)])

    try:
        job = BenchmarkingJob(job_cfg["benchmarkingjob"])
        job.run()
    finally:
        sys.modules.pop("multiedge_test_accuracy", None)

    assert seen_models == ["facebook/deit-base-distilled-patch16-224"]
    rank_csv = Path(job.rank.all_rank_file).read_text(encoding="utf-8")
    assert "facebook/deit-base-distilled-patch16-224" in rank_csv
    assert "./devices_one.yaml" in rank_csv
