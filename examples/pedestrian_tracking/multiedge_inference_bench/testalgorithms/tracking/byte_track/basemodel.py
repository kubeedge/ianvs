# Modified Copyright 2022 The KubeEdge Authors.
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

import argparse
import glob
import os
from collections import OrderedDict
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import motmetrics as mm
from loguru import logger
from sedna.common.class_factory import ClassType, ClassFactory
from yolox.data import ValTransform
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, setup_logger
from yolox.evaluators import MOTEvaluator

from dataset import MOTDataset

__all__ = ["BaseModel"]

# set backend
os.environ["BACKEND_TYPE"] = "TORCH"


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # det self.args
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    # tracking self.args
    parser.add_argument(
        "--track_thresh", type=float, default=0.6, help="tracking confidence threshold"
    )
    parser.add_argument(
        "--track_buffer", type=int, default=30, help="the frames for keep lost tracks"
    )
    parser.add_argument(
        "--match_thresh",
        type=float,
        default=0.9,
        help="matching threshold for tracking",
    )
    parser.add_argument("--min-box-area", type=float, default=100, help="filter out tiny boxes")
    parser.add_argument(
        "--mot20", dest="mot20", default=False, action="store_true", help="test mot20."
    )
    # remove conflict with ianvs
    parser.add_argument("-f")
    return parser


@ClassFactory.register(ClassType.GENERAL, alias="ByteTrack")
class BaseModel:
    def __init__(self, **kwargs) -> None:
        self.args = make_parser().parse_args()
        self.exp = get_exp(str(Path(Path(__file__).parent.resolve(), "yolox_x_ablation.py")), None)
        self.exp.merge(self.args.opts)
        self.args.experiment_name = self.exp.exp_name

        num_gpu = torch.cuda.device_count()
        assert num_gpu <= torch.cuda.device_count()
        self.is_distributed = num_gpu > 1
        # set environment variables for distributed training
        cudnn.benchmark = True
        self.rank = 0
        file_name = os.path.join(self.exp.output_dir, self.args.experiment_name)
        os.makedirs(file_name, exist_ok=True)

        setup_logger(file_name, distributed_rank=self.rank, filename="val_log.txt", mode="a")
        logger.info(f"args: {self.args}")
        self.exp.test_conf = self.args.conf
        self.exp.nmsthre = self.args.nms
        self.model = self.exp.get_model()
        logger.info(f"Model Summary: {get_model_info(self.model, self.exp.test_size)}")

        self.batch_size = kwargs.get("batch_size", 1)

    def load(self, model_url=None) -> None:
        logger.info("loading checkpoint")
        # load the model state dict
        if model_url:
            loc = f"cuda:{self.rank}"
            ckpt = torch.load(model_url, map_location=loc)
            self.model.load_state_dict(ckpt["model"])
            logger.info("loaded checkpoint done.")
            if self.is_distributed:
                self.model = DDP(self.model, device_ids=[self.rank])

            logger.info("\tFusing model...")
            self.model = fuse_model(self.model)
        else:
            raise Exception("model url is None")

    def predict(self, data, input_shape=None, **kwargs):
        if data is None:
            raise Exception("Predict data is None")

        data_dir = data["data_dir"]
        coco = data["coco"]
        ids = data["ids"]
        class_ids = data["class_ids"]
        annotations = data["annotations"]

        inference_output_dir = os.getenv("RESULT_SAVED_URL")
        os.makedirs(inference_output_dir, exist_ok=True)
        val_loader = self._get_eval_loader(
            data_dir, coco, ids, class_ids, annotations, self.batch_size, self.is_distributed, False
        )
        self.evaluator = MOTEvaluator(
            args=self.args,
            dataloader=val_loader,
            img_size=self.exp.test_size,
            confthre=self.exp.test_conf,
            nmsthre=self.exp.nmsthre,
            num_classes=self.exp.num_classes,
        )

        torch.cuda.set_device(self.rank)
        self.model.cuda(self.rank)

        self.model.eval()
        # start evaluate
        *_, summary = self.evaluator.evaluate(
            self.model,
            self.is_distributed,
            True,
            None,
            None,
            self.exp.test_size,
            inference_output_dir,
        )
        logger.info("\n" + summary)

        tsfiles = [
            f
            for f in glob.glob(os.path.join(inference_output_dir, "*.txt"))
            if not os.path.basename(f).startswith("eval")
        ]
        ts = OrderedDict(
            [
                (
                    os.path.splitext(Path(f).parts[-1])[0],
                    mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=-1),
                )
                for f in tsfiles
            ]
        )

        return ts

    def _get_eval_loader(
        self, data_dir, coco, ids, class_ids, annotations, batch_size, is_distributed, testdev=False
    ):
        valdataset = MOTDataset(
            data_dir=data_dir,
            coco=coco,
            ids=ids,
            class_ids=class_ids,
            annotations=annotations,
            img_size=self.exp.test_size,
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        self.data_num_workers = 0
        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader
