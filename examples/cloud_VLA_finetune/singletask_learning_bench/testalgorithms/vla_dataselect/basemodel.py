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

from __future__ import absolute_import, division, print_function

import os
import logging
from typing import Any, Dict

from sedna.common.config import Context
from sedna.common.class_factory import ClassType, ClassFactory
from finetuning import run_finetune, build_cfg
from generation import run_inference, build_generation_cfg
from vlaScripts.finetune import FinetuneConfig
import zipfile

logging.disable(logging.WARNING)

__all__ = ['OpenVLA_Finetune']

os.environ['BACKEND_TYPE'] = 'TORCH'

@ClassFactory.register(ClassType.GENERAL, alias="OpenVLA_Finetune")
class OpenVLA_Finetune:
    """
    只负责调用你现有的 finetune.py。
    参数从 algorithm.yaml 传入，自动构造 FinetuneConfig 并调用 finetune(cfg)。
    """
    def __init__(self, **kwargs):
        # #NOTE: train cfg
        self.cfg = build_cfg()
        print(f"[INFO] FinetuneConfig parameters: {self.cfg}")
        
        # #NOTE: test cfg
        self.gen_cfg = build_generation_cfg()
        print(f"[INFO] FinetuneConfig parameters: {self.gen_cfg}")
        
        # self.train_cfgs = train_cfgs
        # self.valid_cfgs = valid_cfgs

        # self.train_cfgs.lr = kwargs.get("learning_rate", 0.0001)
        # self.train_cfgs.momentum = kwargs.get("momentum", 0.9)
        # self.train_cfgs.epochs = kwargs.get("epochs", 2)
        # self.train_cfgs.checkname = ''
        # self.train_cfgs.batch_size = 4
        # self.train_cfgs.depth = True
        # print("init")
        # os.environ["MODEL_NAME"] = "model_best_mapi_only.pth"

        # self.train_cfgs.gpu_ids = ['cuda:0']
        # self.valid_cfgs.gpu_ids = ['cuda:0']
        # self.valid_cfgs.depth = True

        # self.validator = Validator(self.valid_cfgs)
        # self.trainer = Trainer(self.train_cfgs)
        # self.trainer.saver = Saver(self.train_cfgs)
        # self.checkpoint_path = self.load(
        #     Context.get_parameters("base_model_url"))
    # # TODO： train 返回的是训练后的 checkpoint 的 path 
    def train(self, non_data):
        self.train_data = non_data
        # cfg = build_cfg(
        #     max_steps=800,
        #     wandb_project="openvla",
        #     wandb_entity="me",
        # )
        # run_finetune(
        #     nproc_per_node=1,
        #     cfg=cfg
        # )
        # # 这个地方需要返回checkpoint的 path
        print("train_OK")
    # #TODO:save接收 train返回的checkpoint path：.../.../.../xxxx
    def save(self, model_path=None):
        model_path = "/inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/model/finetune_model/models--openvla--openvla-7b+libero_10_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug"
        print("loading save")
        if not model_path:
            raise Exception("model path is None.")
        # #TODO:使用self.checkpoint
        # model_dir, model_name = os.path.split(self.checkpoint_path)
        model_dir, model_name = os.path.split(model_path)
        zip_path = os.path.join(model_path, "model.zip")
        if os.path.isfile(zip_path):
            return zip_path
        else:
            models = [model for model in os.listdir(model_dir) if model_name == model]
            
            if os.path.splitext(model_path)[-1] != ".zip":
                model_path = os.path.join(model_path, "model.zip")

            if not os.path.isdir(os.path.dirname(model_path)):
                os.makedirs(os.path.dirname(model_path))
                
            file_paths = []
            for model_file in models:
                model_file_path = os.path.join(model_dir, model_file)
                for file in os.listdir(model_file_path):
                    file_path = os.path.join(model_file_path, file)
                    file_paths.append(file_path)
            print("[DEBUG] len_file_paths:", len(file_paths))
            with zipfile.ZipFile(model_path, "w") as f:
                for file_path in file_paths:
                    print("[DEBUG] file_path:",file_path)
                    f.write(file_path, model_file,compress_type=zipfile.ZIP_DEFLATED)
            return model_path
    
    def load(self, model_url=None):
        print(model_url)
        if model_url is None:
            model_url = '/inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/model/finetune_model/models--openvla--openvla-7b+libero_10_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug.pth'
        print("load_OK")
         
    # #TODO:这个地方也可以直接使用checkpoint_path
    # #TODO:返回的是 infer 的 result
    def predict(self, data):
        # run_inference(self.gen_cfg)
        print("predict_OK")
        
        
def test_save(obj):
    model_path = "/inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/model/finetune_model/models--openvla--openvla-7b+libero_10_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug"
    obj.save(model_path)
if __name__=="__main__":
    obj = OpenVLA_Finetune
    test_save(obj)