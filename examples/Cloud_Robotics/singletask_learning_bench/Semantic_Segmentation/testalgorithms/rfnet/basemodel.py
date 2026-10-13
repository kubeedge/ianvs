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
import gc
import zipfile
import logging

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.common.config import Context
from sedna.common.file_ops import FileOps

from RFNet.dataloaders import make_data_loader
from RFNet.dataloaders import custom_transforms as tr
from RFNet.utils.lr_scheduler import LR_Scheduler
from RFNet.train import Trainer
from RFNet.eval import Validator, load_my_state_dict
from RFNet.utils.saver import Saver
from RFNet.utils.args import TrainArgs, ValArgs

logging.disable(logging.WARNING)

__all__ = ["BaseModel"]

os.environ['BACKEND_TYPE'] = 'PYTORCH'


@ClassFactory.register(ClassType.GENERAL, alias="RFNet")
class BaseModel:

    def __init__(self, **kwargs):
        self.train_args = TrainArgs(**kwargs)
        self.train_args.depth = True
        self.train_args.batch_size = kwargs.get("batch_size", 4)
        self.trainer = None

        self.val_args = ValArgs(**kwargs)
        self.val_args.depth = True
        label_save_dir = Context.get_parameters("INFERENCE_RESULT_DIR", "./inference_results")
        self.val_args.color_label_save_path = os.path.join(label_save_dir, "color")
        self.val_args.merge_label_save_path = os.path.join(label_save_dir, "merge")
        self.val_args.label_save_path = os.path.join(label_save_dir, "label")
        self.validator = Validator(self.val_args)

        model_url = kwargs.get("model_url") or Context.get_parameters("base_model_url", None)
        self.checkpoint_path = self.load(model_url)

    def train(self, train_data, valid_data=None, **kwargs):
        if train_data is None or train_data.x is None or train_data.y is None:
            raise Exception("Train data is None.")

        self.trainer = Trainer(self.train_args, train_data=train_data, valid_data=valid_data)
        print("Total epochs:", self.trainer.args.epochs)

        for epoch in range(self.trainer.args.start_epoch, self.trainer.args.epochs):
            if epoch == 0 and self.trainer.val_loader:
                self.trainer.validation(epoch)
            self.trainer.training(epoch)

            if self.trainer.args.no_val and (
                epoch % self.trainer.args.eval_interval == (self.trainer.args.eval_interval - 1)
                or epoch == self.trainer.args.epochs - 1
            ):
                is_best = False
                self.checkpoint_path = self.trainer.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.trainer.model.state_dict(),
                    'optimizer': self.trainer.optimizer.state_dict(),
                    'best_pred': self.trainer.best_pred,
                }, is_best)

        self.trainer.writer.close()
        return self.checkpoint_path

    def save(self, model_path):
        if not model_path:
            raise Exception("model path is None.")

        model_dir, model_name = os.path.split(self.checkpoint_path)
        models = [model for model in os.listdir(model_dir) if model_name in model]

        if os.path.splitext(model_path)[-1] != ".zip":
            model_path = os.path.join(model_path, "model.zip")

        if not os.path.isdir(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))

        with zipfile.ZipFile(model_path, "w") as f:
            for model_file in models:
                model_file_path = os.path.join(model_dir, model_file)
                f.write(model_file_path, model_file, compress_type=zipfile.ZIP_DEFLATED)

        return model_path

    def predict(self, data, **kwargs):
        if not isinstance(data[0][0], dict):
            data = self._preprocess(data)

        if isinstance(data, np.ndarray):
            data = data.tolist()

        self.validator.test_loader = DataLoader(
            data, batch_size=self.val_args.test_batch_size, shuffle=False, pin_memory=True
        )
        return self.validator.validate()

    def load(self, model_url=None):
        if model_url is None:
            model_url = './models/model_best_mapi_only.pth'
        if FileOps.exists(model_url):
            self.validator.new_state_dict = torch.load(model_url, map_location=torch.device("cpu"))
            self.validator.model = load_my_state_dict(
                self.validator.model, self.validator.new_state_dict['state_dict']
            )
        return model_url

    def _preprocess(self, image_urls):
        transformed_images = []
        for paths in image_urls:
            if len(paths) == 2:
                img_path, depth_path = paths
                _img = Image.open(img_path).convert('RGB')
                _depth = Image.open(depth_path)
            else:
                img_path = paths[0]
                _img = Image.open(img_path).convert('RGB')
                _depth = _img

            sample = {'image': _img, 'depth': _depth, 'label': _img}
            composed_transforms = transforms.Compose([
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])
            transformed_images.append((composed_transforms(sample), img_path))

        return transformed_images
