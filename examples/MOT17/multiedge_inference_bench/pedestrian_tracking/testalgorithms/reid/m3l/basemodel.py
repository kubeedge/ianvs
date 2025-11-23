<<<<<<< HEAD
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

from __future__ import print_function, absolute_import
import os
import time
import random
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from sedna.common.class_factory import ClassType, ClassFactory

from reid import models
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.meters import AverageMeter
from reid.utils import to_torch

__all__ = ["BaseModel"]

# set backend
os.environ["BACKEND_TYPE"] = "TORCH"


@ClassFactory.register(ClassType.GENERAL, alias="M3L")
class BaseModel:
    def __init__(self, **kwargs):

        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        cudnn.deterministic = True
        cudnn.benchmark = True
        self.model = None
        self.batch_size = kwargs.get("batch_size", 1)

    def load(self, model_url=None):
        if model_url:
            arch = re.compile("_([a-zA-Z]+).pth").search(model_url).group(1)
            # Create model
            self.model = models.create(
                arch, num_features=0, dropout=0, norm=True, BNNeck=True
            )
            # use CUDA
            self.model.cuda()
            self.model = nn.DataParallel(self.model)
            if Path(model_url).is_file():
                checkpoint = torch.load(model_url, map_location=torch.device('cpu'))
                print("=> Loaded checkpoint '{}'".format(model_url))
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                raise ValueError("=> No checkpoint found at '{}'".format(model_url))
        else:
            raise Exception(f"model url is None")

    def predict(self, data, input_shape=None, **kwargs):
        train_dataset = kwargs.get("train_dataset")
        gallery = [
            (x, int(y.split("/")[-1]), -1, 1)
            for x, y in zip(train_dataset.x, train_dataset.y)
        ]
        root = Path(train_dataset.x[0]).parents[2]
        query = [(x, -1, -1, 1) for x in data]
        test_loader = self._get_test_loader(Path(root, "query"), list(set(query) | set(gallery)), 256, 128, self.batch_size, 4)
        features = self._extract_features(self.model, test_loader)
        distmat = self._pairwise_distance(
            features, query, gallery
        )
        distmat = self.to_numpy(distmat)
        gallery_ids = np.asarray([pid for _, pid, _, _ in gallery])
        return distmat, gallery_ids

    def _get_test_loader(
        self, data_dir, data, height, width, batch_size, workers, testset=None
    ):
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        test_transformer = T.Compose(
            [T.Resize((height, width), interpolation=3), T.ToTensor(), normalizer]
        )

        if testset is None:
            testset = data

        test_loader = DataLoader(
            Preprocessor(testset, root=data_dir, transform=test_transformer),
            batch_size=batch_size,
            num_workers=workers,
            shuffle=False,
            pin_memory=True,
        )

        return test_loader

    def _extract_cnn_feature(self, model, inputs):
        inputs = to_torch(inputs).cuda()
        outputs = model(inputs.contiguous())
        outputs = outputs.data.cpu()
        return outputs

    def _extract_features(self, model, data_loader, print_freq=50):
        model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        features = OrderedDict()

        end = time.time()
        with torch.no_grad():
            for i, (imgs, fnames, pids, _, _, _) in enumerate(data_loader):
                data_time.update(time.time() - end)

                outputs = self._extract_cnn_feature(model, imgs)
                for fname, output, _ in zip(fnames, outputs, pids):
                    features[fname] = output

                batch_time.update(time.time() - end)
                end = time.time()

                if (i + 1) % print_freq == 0:
                    print("Extract Features: [{}/{}]\t"
                        "Time {:.3f} ({:.3f})\t"
                        "Data {:.3f} ({:.3f})\t".format(i + 1, len(data_loader), batch_time.val, batch_time.avg, data_time.val, data_time.avg,))

        return features

    def _pairwise_distance(self, features, query=None, gallery=None):
        x = torch.cat([features[f].unsqueeze(0) for f, _, _, _ in query], 0)
        y = torch.cat([features[f].unsqueeze(0) for f, _, _, _ in gallery], 0)
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        dist_m = (
            torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n)
            + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        )
        dist_m.addmm_(1, -2, x, y.t())
        return dist_m

    def to_numpy(self, tensor):
        if torch.is_tensor(tensor):
            return tensor.cpu().numpy()
        elif type(tensor).__module__ != 'numpy':
            raise ValueError("Cannot convert {} to numpy array"
                            .format(type(tensor)))
        return tensor
=======
version https://git-lfs.github.com/spec/v1
oid sha256:289e53e792e4418b18e37172cb1f5a21412b60b934992f4a605dc5b563f359cd
size 5817
>>>>>>> 9676c3e (ya toh aar ya toh par)
