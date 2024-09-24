import model_cfg
import torch
from pathlib import Path

initial_model = "/Users/wuyang/Desktop/秋招/项目/ianvs/ianvs/initial_model/ViT-B_16-224.npz"
model = model_cfg.module_shard_factory("google/vit-base-patch16-224", initial_model, 37, 48, 1)
dummy_input = torch.randn(1, *[197, 768])
torch.onnx.export(model,
                dummy_input,
                str(Path(Path(initial_model).parent.resolve())) + "/sub_model_" + str(3) + ".onnx",
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=['input_' + str(37)],
                output_names=['output_' + str(48)])