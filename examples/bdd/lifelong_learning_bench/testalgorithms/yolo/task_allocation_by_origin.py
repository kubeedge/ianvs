from curses import raw
from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType
from mmcls.apis import init_model
from mmcv.parallel import collate, scatter
from mmcls.datasets.pipelines import Compose
from mmcls.apis import set_random_seed
set_random_seed(0, deterministic=True)
import torch
import numpy as np
import heapq
import yaml

__all__ = ('TaskAllocationByOrigin',)
config_file = 'examples/bdd/lifelong_learning_bench/testalgorithms/yolo/model_selector/choose_net_b64.py'# 神经网络配置文件
checkpoint_file = 'examples/bdd/lifelong_learning_bench/testalgorithms/yolo/model_selector/adaptive_selector_7w5_6w.pth'  # 神经网络权重参数
device = 'cuda:0'  # 推理设备
weight_list = ['all.pt', 'bdd.pt', 'traffic_0.pt', 'bdd_street.pt', 'bdd_clear.pt', 'bdd_daytime.pt',
                'bdd_highway.pt', 'traffic_2.pt', 'bdd_overcast.pt', 'bdd_residential.pt', 'traffic_1.pt', 
                'bdd_snowy.pt', 'bdd_rainy.pt', 'bdd_night.pt', 'soda.pt', 'bdd_cloudy.pt', 'bdd_cloudy_night.pt',
                'bdd_highway_residential.pt', 'bdd_snowy_rainy.pt', 'soda_t1.pt']
model_selector = init_model(config_file, checkpoint_file, device=device)
def inference_model(model, img):
    """Inference image(s) with the classifier.

    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.

    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=img), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
        data = dict(img=img)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device])[0]

    with torch.no_grad():
        scores = model(return_loss=False, **data)
        scores = np.array(scores)
        pred_score = np.max(scores[0])
        pred_label = np.argmax(scores[0])
        top5_index = heapq.nlargest(5, range(len(scores[0])), scores[0].__getitem__)
        top15_index = heapq.nlargest(len(weight_list), range(len(scores[0])), scores[0].__getitem__)
        result = {'pred_label': pred_label, 'pred_score': float(pred_score), 'scores': scores[0], 'top5':top5_index, 'top15':top15_index}

    return result


@ClassFactory.register(ClassType.STP, alias="TaskAllocationByOrigin")
class TaskAllocationByOrigin:
    """
    Corresponding to `TaskDefinitionByOrigin`

    Parameters
    ----------
    task_extractor : Dict
        used to match target tasks
    origins: List[Metadata]
        metadata is usually a class feature
        label with a finite values.
    """

    def __init__(self, **kwargs):
        self.default_origin = kwargs.get("default", None)
    
    def __call__(self, task_extractor, samples: BaseDataSource):
        result = inference_model(model_selector, samples.x[0][0])  # 默认传入数据地址，传入数据的值也可以
        allocations = result['top5']

        return samples, allocations