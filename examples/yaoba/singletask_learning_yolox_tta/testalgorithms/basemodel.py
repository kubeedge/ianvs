from __future__ import absolute_import, division, print_function
import json
from sedna.common.class_factory import ClassType, ClassFactory
import copy
import os
import os.path as osp
import time
import mmcv
import torch
from mmcv import Config
from mmcv.utils import get_git_hash
from multiprocessing import Pool
from mmdet import __version__
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals,
                         setup_multi_processes, update_data_root)
from mmdet.apis import init_detector, inference_detector

__all__ = ["BaseModel"]

from tqdm import tqdm

from examples.yaoba.singletask_learning_yolox_tta.resource.utils.general_TTA_v5 import TTA_single_img

# set backend
os.environ['BACKEND_TYPE'] = 'PYTORCH'


@ClassFactory.register(ClassType.GENERAL, alias="mmlab")
class BaseModel:

    def __init__(self, config, work_dir, **kwargs):
        self.config = config
        self.work_dir = work_dir
        cfg = Config.fromfile(self.config)
        cfg.work_dir = self.work_dir
        cfg = replace_cfg_vals(cfg)
        update_data_root(cfg)
        setup_multi_processes(cfg)
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        if cfg.get('work_dir', None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            cfg.work_dir = osp.join('./work_dirs',
                                    osp.splitext(osp.basename(self.config))[0])
        self.distributed = False
        cfg.gpu_ids = [0]
        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        # dump config
        cfg.dump(osp.join(cfg.work_dir, osp.basename(self.config)))
        # init the logger before other steps
        self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, f'{self.timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
        self.meta = dict()
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        self.meta['env_info'] = env_info
        self.meta['config'] = cfg.pretty_text
        # log some basic info
        logger.info(f'Distributed training: {self.distributed}')
        logger.info(f'Config:\n{cfg.pretty_text}')
        cfg.device = get_device()
        self.device = cfg.device
        seed = 1
        logger.info(f'Set random seed to {seed}, '
                    f'deterministic: False')
        cfg.seed = seed
        self.meta['seed'] = seed
        self.meta['exp_name'] = osp.basename(self.config)
        self.cfg = cfg

    def train(self, train_data, valid_data=None, **kwargs):
        img_prefix, ann_file = train_data[0], train_data[1]
        # Get the number of categories and their names.
        labels = json.load(open(ann_file, mode="r", encoding="utf-8"))
        categories = labels['categories']
        classes = [c['name'] for c in categories]
        num_classes = len(categories)

        if "dataset" in self.cfg.data.train.keys():
            self.cfg.data.train.dataset.classes = classes
            self.cfg.data.train.dataset.ann_file = ann_file
            self.cfg.data.train.dataset.img_prefix = img_prefix
        else:
            self.cfg.data.train.classes = classes
            self.cfg.data.train.ann_file = ann_file
            self.cfg.data.train.img_prefix = img_prefix
        self.cfg.model.bbox_head.num_classes = num_classes
        self.cfg.data.val.classes = classes
        self.cfg.data.test.classes = classes

        self.model = build_detector(
            self.cfg.model,
            train_cfg=self.cfg.get('train_cfg'),
            test_cfg=self.cfg.get('test_cfg'))
        self.model.init_weights()

        print("A total of", self.cfg.runner.max_epochs, "epoch")

        datasets = [build_dataset(self.cfg.data.train)]
        if len(self.cfg.workflow) == 2:
            val_dataset = copy.deepcopy(self.cfg.data.val)
            val_dataset.pipeline = self.cfg.data.train.pipeline
            datasets.append(build_dataset(val_dataset))
        if self.cfg.checkpoint_config is not None:
            # save mmdet version, config file content and class names
            # checkpoints as meta data
            self.cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__ + get_git_hash()[:7],
                CLASSES=datasets[0].CLASSES)
            # add an attribute for visualization convenience
        self.model.CLASSES = datasets[0].CLASSES
        train_detector(
            self.model,
            datasets,
            self.cfg,
            distributed=self.distributed,
            validate=False,
            timestamp=self.timestamp,
            meta=self.meta)
        self.last_checkpoint_path = os.path.join(self.cfg.work_dir, "latest.pth")
        return self.last_checkpoint_path

    def save(self, model_path):
        return self.last_checkpoint_path

    def load(self, model_url=None):
        self.load_model = init_detector(self.cfg, model_url)
        return self.load_model

    def predict(self, data, **kwargs):
        # ianvs格式
        predict_dict = {}
        for imgPath in data:
            print("inference predict->", imgPath)
            imgName = osp.basename(imgPath)
            predict_dict[str(imgName)] = []
            predict_result = inference_detector(self.load_model, imgPath)
            for i, vs in enumerate(predict_result):
                temp_dict = {}
                temp_dict["category_id"] = i
                temp_dict["bbox"] = vs
                predict_dict[str(imgName)].append(temp_dict)
        return predict_dict

    def tta_predict(self, data, strategy):
        torch.multiprocessing.set_start_method('spawn', force=True)
        predict_dict = {}
        pbar = tqdm(total=len(data))
        pbar.set_description('progress bar')
        update = lambda *args: pbar.update()
        start = time.time()
        with Pool(processes=4) as pool:  # Specify the number of processes
            merged_results = [
                pool.apply_async(TTA_single_img, (self.load_model, img_file, strategy),
                                 callback=update)
                for img_file in data]
            end = time.time()
            merged_results = [res.get() for res in merged_results]
            pool.close()
            pool.join()
        total_time = ((end - start) * 1000)
        print(
            f"The total inference time is:{total_time}mm, "
            f"and the average inference time for a single sample is{total_time / len(data)}")
        # Convert to the Ianvs format
        for merged_result in merged_results:
            name = merged_result['name']
            predict_dict[name] = []
            result = merged_result['result']
            for i, vs in enumerate(result):
                temp_dict = {}
                temp_dict["category_id"] = i
                temp_dict["bbox"] = vs
                predict_dict[name].append(temp_dict)
        return predict_dict
