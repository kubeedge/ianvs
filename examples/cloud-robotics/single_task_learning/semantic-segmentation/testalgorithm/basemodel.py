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
from RFNet.eval import Validator
from RFNet.utils.saver import Saver
from RFNet.eval import load_my_state_dict
import RFNet.train_config as train_cfgs
import RFNet.eval_config as valid_cfgs

logging.disable(logging.WARNING)

__all__ = ["BaseModel"]

os.environ['BACKEND_TYPE'] = 'TORCH'


@ClassFactory.register(ClassType.GENERAL, alias="RFNet")
class BaseModel:

    def __init__(self, **kwargs):
        self.train_cfgs = train_cfgs
        self.valid_cfgs = valid_cfgs

        self.train_cfgs.lr = kwargs.get("learning_rate", 0.0001)
        self.train_cfgs.momentum = kwargs.get("momentum", 0.9)
        self.train_cfgs.epochs = kwargs.get("epochs", 2)
        self.train_cfgs.checkname = ''
        self.train_cfgs.batch_size = 4
        self.train_cfgs.depth = True

        os.environ["MODEL_NAME"] = "model_best_mapi_only.pth"

        self.train_cfgs.gpu_ids = ['cuda:0']
        self.valid_cfgs.gpu_ids = ['cuda:0']
        self.valid_cfgs.depth = True

        self.validator = Validator(self.valid_cfgs)
        self.trainer = Trainer(self.train_cfgs)
        self.trainer.saver = Saver(self.train_cfgs)
        self.checkpoint_path = self.load(
            Context.get_parameters("base_model_url"))

    def train(self, train_data, valid_data=None, **kwargs):
        if train_data is None or train_data.x is None or train_data.y is None:
            raise Exception("Train data is None.")

        self.trainer.train_loader, self.trainer.val_loader, _, _ = \
            make_data_loader(
                self.trainer.args, train_data=train_data, valid_data=valid_data, **kwargs)

        # Define lr scheduler
        self.trainer.scheduler = LR_Scheduler(self.trainer.args.lr_scheduler, self.trainer.args.lr,
                                              self.trainer.args.epochs, len(self.trainer.train_loader))
        print("Total epoches:", self.trainer.args.epochs)

        for epoch in range(self.trainer.args.start_epoch, self.trainer.args.epochs):
            if epoch == 0 and self.trainer.val_loader:
                self.trainer.validation(epoch)
            self.trainer.training(epoch)

            if self.trainer.args.no_val and \
                    (epoch % self.trainer.args.eval_interval == (self.trainer.args.eval_interval - 1)
                     or epoch == self.trainer.args.epochs - 1):
                # save checkpoint when it meets eval_interval or the training finished
                is_best = False
                self.checkpoint_path = self.trainer.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.trainer.model.state_dict(),
                    'optimizer': self.trainer.optimizer.state_dict(),
                    'best_pred': self.trainer.best_pred,
                }, is_best)

            # if not self.trainer.args.no_val and \
            #         epoch % self.train_args.eval_interval == (self.train_args.eval_interval - 1) \
            #         and self.trainer.val_loader:
            #     self.trainer.validation(epoch)
            # self.checkpoint_path = os.path.join(self.temp_dir, '{}_'.format(
            #                 cfgs.DATASET_NAME) + str(_global_step) + "_" + str(time.time()) + '_model.ckpt')
            #             saver.save(sess, self.checkpoint_path)
            #             print('Weights have been saved to {}.'.format(self.checkpoint_path))

        self.trainer.writer.close()

        return self.checkpoint_path

    def save(self, model_path):
        if not model_path:
            raise Exception("model path is None.")

        model_dir, model_name = os.path.split(self.checkpoint_path)
        models = [model for model in os.listdir(
            model_dir) if model_name in model]

        if os.path.splitext(model_path)[-1] != ".zip":
            model_path = os.path.join(model_path, "model.zip")

        if not os.path.isdir(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))

        with zipfile.ZipFile(model_path, "w") as f:
            for model_file in models:
                model_file_path = os.path.join(model_dir, model_file)
                f.write(model_file_path, model_file,
                        compress_type=zipfile.ZIP_DEFLATED)

        return model_path

    def predict(self, data, **kwargs):
        if not isinstance(data[0][0], dict):
            data = self._preprocess(data)

        if type(data) is np.ndarray:
            data = data.tolist()

        self.validator.test_loader = DataLoader(
            data, batch_size=self.validator.args.test_batch_size, shuffle=False, pin_memory=True)
        return self.validator.validate()

    def load(self, model_url):
        model_url = '/home/wxc/dev/ianvs/models/model_best_mapi_only.pth'
        if FileOps.exists(model_url):
            self.validator.new_state_dict = torch.load(
                model_url, map_location=torch.device("cpu"))
        else:
            raise Exception("model url does not exist.")
        self.validator.model = load_my_state_dict(
            self.validator.model, self.validator.new_state_dict['state_dict'])
        return model_url

    def _preprocess(self, image_urls):
        transformed_images = []
        for img_path, depth_path in image_urls:
            _img = Image.open(img_path).convert('RGB')
            _depth = Image.open(depth_path)

            sample = {'image': _img, 'depth': _depth, 'label': _img}
            composed_transforms = transforms.Compose([
                # tr.CropBlackArea(),
                # tr.FixedResize(size=self.args.crop_size),
                tr.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
                tr.ToTensor()])

            transformed_images.append((composed_transforms(sample), img_path))

        return transformed_images