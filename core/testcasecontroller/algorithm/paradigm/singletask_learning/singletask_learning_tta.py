# pylint: disable=C0114
# pylint: disable=E0401
# pylint: disable=C0115
# pylint: disable=W0246
# pylint: disable=R1725
# pylint: disable=R1732
# pylint: disable=C0103
# pylint: disable=R0801
import json
import os
import torch
from mmdet.apis import init_detector
from core.common.constant import ParadigmType
from examples.yaoba.singletask_learning_yolox_tta.resource.utils.TTA_strategy import TTA_Strategy
from .singletask_learning import SingleTaskLearning


class SingleTaskLearningTTA(SingleTaskLearning):

    def __init__(self, workspace, **kwargs):
        super(SingleTaskLearningTTA, self).__init__(workspace, **kwargs)

    def run(self):
        # Build an experimental task
        job = self.build_paradigm_job(str(ParadigmType.SINGLE_TASK_LEARNING.value))

        # If there are no initialized model weights, then train a new model from scratch
        if self.initial_model != "":
            trained_model = self.initial_model
        else:
            trained_model = self._train(job, None)

        # Search for the optimal test-time augmentation policies
        searched_strategy = self._search_tta_strategy(job, trained_model)

        # Merging the optimal policies with original default policy
        merged_strategy = self._prepare_infer_strategy(job, searched_strategy)

        # infer the test set with searched policies
        inference_result = self._inference_w_tta(job, trained_model, merged_strategy)
        self.system_metric_info['use_raw']=True
        return inference_result, self.system_metric_info

    def _inference_w_tta(self, job, trained_model, strategy):
        # Load test set data
        img_prefix = self.dataset.image_folder_url
        ann_file_path = self.dataset.test_url
        ann_file = json.load(open(ann_file_path, mode="r", encoding="utf-8"))
        test_set = []
        for i in ann_file['images']:
            test_set.append(os.path.join(img_prefix, i['file_name']))

        # Perform inference with data augmentation policy.
        job.load(trained_model)
        print(f"Total infer strategy is :{strategy}")
        infer_res = job.tta_predict(test_set, strategy)

        return infer_res

    def _prepare_infer_strategy(self, job, searched_strategy):
        default_img_size = None
        # The default inference policy
        for p in job.cfg.data.test.pipeline:
            if p['type'] == 'MultiScaleFlipAug':
                default_img_size = p['img_scale']
        if default_img_size:
            combined_strategy = [[("TTA_Resize", default_img_size), ]]
        else:
            raise ValueError("can not find img_scale model cfg")
        combined_strategy.append(searched_strategy[0])

        return combined_strategy

    def _search_tta_strategy(self, job, model_url):
        # Load validation dataset
        img_prefix = self.dataset.image_folder_url
        ann_file = self.dataset.val_url

        # Create a search agent to search for the best data augmentation strategy.
        model_cfg = job.cfg
        model = init_detector(model_cfg, model_url)
        torch.multiprocessing.set_start_method("spawn", force=True)
        search_agent = TTA_Strategy(
            model=model,
            val_image_path=img_prefix,
            val_anno_path=ann_file,
            log_dir=os.path.join(model_cfg.work_dir, "log"),
            worker=6,
            nms_thr=0.5
        )
        # Search for single policies for TTA
        single_strategies = search_agent.search_single_strategy(top_num=3)

        # Search for Cascade policies for TTA, which based on single policies
        cascade_strategies = search_agent.search_cascade_strategy(
            single_strategies,
            cascade_num=3,
            top_num=5
        )
        return cascade_strategies

    def _train(self, job, initial_model):
        img_prefix = self.dataset.image_folder_url
        ann_file = self.dataset.train_url
        checkpoint_path = job.train((img_prefix, ann_file))
        return checkpoint_path
