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
import sys
import logging
from typing import Any, Dict
import numpy as np

from sedna.common.config import Context
from sedna.common.class_factory import ClassType, ClassFactory
from finetuning import run_finetune, build_cfg
from generation import run_inference, build_generation_cfg
from vla_component.vlaScripts.finetune import FinetuneConfig

COMPONENT_DIR = "/inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/ianvs/examples/cloud_VLA_finetune/singletask_learning_bench/testalgorithms/vla_dataselect/vla_component"
if COMPONENT_DIR not in sys.path:
    sys.path.append(COMPONENT_DIR)
    
import zipfile

logging.disable(logging.WARNING)

__all__ = ['OpenVLA_Finetune']

os.environ['BACKEND_TYPE'] = 'TORCH'

@ClassFactory.register(ClassType.GENERAL, alias="OpenVLA_Finetune")
class OpenVLA_Finetune:
    def __init__(self, **kwargs):
        
        # #NOTE：set train_cfg
        # need to dataset_name
        self.config_args = {
            'vla_path': "/inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/model/models--openvla--openvla-7b", #微调模型
            'data_root_dir': "/inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/data/openvla/modified_libero_rlds", #数据根目录
            'dataset_name': "libero_10_no_noops", #dataset name
            'run_root_dir': "/inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/model/finetune_model/test", #保存根目录
            'adapter_tmp_dir': "/inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/model/finetune_model/test", #adapter保存目录
            'batch_size': 1, #batch_size
            'max_steps': 50, #max train step
            'save_steps': 50, #save model step
            'learning_rate': 5e-4, #learning rate
            'grad_accumulation_steps': 1, 
            'image_aug': True, # true
            'shuffle_buffer_size': 100000,
            'save_latest_checkpoint_only': True, 
            'use_lora': True,
            'lora_rank': 32,
            'lora_dropout': 0.0,
            'use_quantization': False,
            'wandb_project': None,
            'wandb_entity': None,
            'run_id_note': None,
        }
        #  kwargs values
        self.config_args["dataset_name"] = kwargs.get("dataset_name", None)

        # #NOTE: train 
        self.checkpoint_root_path = self.config_args["run_root_dir"]
        self.finetuned_model_name = self.config_args["vla_path"].split('/')[-1]
        self.dataset_name = self.config_args['dataset_name']
        self.bs = self.config_args['batch_size']
        self.learning_rate = self.config_args['learning_rate']
        self.lora_rank = self.config_args['lora_rank']
        self.dropout = self.config_args["lora_dropout"]
        
        # set ckpt
        self.checkpoint_name = f"{self.finetuned_model_name}+{self.dataset_name}+b{self.bs}+lr-{self.learning_rate}+lora-r{self.lora_rank}+dropout-{self.dropout}--image_aug"
        self.checkpoint_path = os.path.join(self.checkpoint_root_path, self.checkpoint_name)
        self.checkpoint_path = "/inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/model/models--openvla--openvla-7b-finetuned-libero-object"
        
        # #NOTE: test_cfg
        # need to num_trials_per_task/task_suite_name / local_log_dir
        self.test_config_args = {
            'model_family': "openvla",  # model family
            'pretrained_checkpoint': self.checkpoint_path,  # eval model path
            'task_suite_name': "libero_10",  # task suite
            'center_crop': True,  
            'seed': 7,  
            'load_in_8bit': False,  
            'load_in_4bit': False,  
            'num_steps_wait': 10,  
            'num_trials_per_task': 2,  # task times
            'local_log_dir': "/inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/oppos/test",  # log
            'run_id_note': None, 
            'use_wandb': False, 
            'wandb_project': None,  
            'wandb_entity': None,  
        }
        self.test_config_args["task_suite_name"] = kwargs.get("task_suite_name", None)
        self.test_config_args["num_trials_per_task"] = kwargs.get("num_trials_per_task", None) 
        
    # # TODO： return trained checkpoint 的 path 
    def train(self, train_data):
        self.train_data = train_data
        #set train cfg
        self.cfg = build_cfg(**self.config_args)
        logging.info(f"FinetuneConfig parameters: {self.cfg}")
        self.checkpoint_path = run_finetune(
            nproc_per_node=1,
            cfg = self.cfg
        )
        self.test_config_args["pretrained_checkpoint"] = self.checkpoint_path
        logging.info("train: ", self.checkpoint_path)
        return self.checkpoint_path
        
        
    # #TODO: return trained checkpoint path
    def save(self, model_path=None):
        save_model = False
        if save_model:
            model_path = "/inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/model/finetune_model/models--openvla--openvla-7b+libero_10_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug"
            logging.info("loading save")
            if not model_path:
                raise Exception("model path is None.")
            # #TODO:using self.checkpoint
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
                with zipfile.ZipFile(model_path, "w") as f:
                    for file_path in file_paths:
                        f.write(file_path, model_file,compress_type=zipfile.ZIP_DEFLATED)
                return model_path
    
    def load(self, model_url=None):
        logging.info("loading load")
        load_model = False
        if load_model:
            if model_url is None:
                model_url = '/inspire/hdd/global_user/chaimingxu-240108540141/jwq-test/model/finetune_model/models--openvla--openvla-7b+libero_10_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug.pth'
            return model_url
         
    # #TODO:return infer result
    def predict(self, eval_data):
        logging.info("loading predict")
        self.eval_data = eval_data
        self.gen_cfg = build_generation_cfg(**self.test_config_args)
        logging.info(f"FinetuneConfig parameters: {self.gen_cfg}")
        result_path = run_inference(self.gen_cfg)
        return result_path