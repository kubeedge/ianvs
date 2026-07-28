import time
from multiprocessing import Pool
import warnings

warnings.filterwarnings("ignore")
import copy
from mmdet.datasets import build_dataset
import json
import os
from mmdet.core.post_processing.bbox_nms import batched_nms
import torch
from mmdet.core import bbox2result
from tqdm import tqdm
from mmdet.apis import inference_detector
from .TTA_augs_xyxy_cv2 import *

torch.set_printoptions(sci_mode=False)


# 与V4版本相比，获取传入模型的cuda信息来决定用哪个cuda算，同时去掉model_TTA_infer内部的进度条，
# 进度条让TTA_strategy展示，这样方便看整体进度


def model_TTA_infer(model, img_path, anno_path, augs=None, worker=4, nms_thr=0.5):
    # 读取标注信息
    with open(anno_path, 'r', encoding='utf-8') as fp:
        labels = json.load(fp)
    pending_infer_images = labels['images']  # 待预测的图片列表
    # 多进程执行 TTA 预测
    with Pool(processes=worker) as pool:
        tasks = []
        for img_file in pending_infer_images:
            single_img_path = os.path.join(img_path, img_file['file_name'])
            task = pool.apply_async(TTA_single_img, (model, single_img_path, augs, nms_thr))
            tasks.append(task)
        merged_results = [task.get() for task in tasks]
        pool.close()
        pool.join()
    # 计算平均精度值
    dataset = model.cfg.data.test
    ap = eval_ap_coco(dataset, merged_results)

    return ap, None


def TTA_single_img(model, img_path, augs, nms_thr=0.5):
    """对单张图片进行 TTA 预测

    Args:
        model (nn.Module): 模型
        img_path (str): 图片路径
        augs (list[list[tuple]]): 图像增广列表，每个元素为一个变换方法及其参数列表
        nms_thr (float, optional): NMS 阈值. 默认是 to 0.5.
    Returns:
        dict: 预测结果字典，包含图片名和预测框等信息
    """
    # 读取图片
    orig_img = mmcv.imread(img_path)
    # 存储所有变换的预测结果
    total_bboxes = []  # 所有变换后的预测框
    total_scores = []  # 所有变换后的预测框得分
    total_idxs = []  # 所有变换后的预测框所属类别索引

    # 对每种图像增广方式进行 TTA 预测
    for aug in augs:
        # 对原图像进行深拷贝
        img = copy.deepcopy(orig_img)

        # 存储当前 augment 中所有的变换方法和对应参数
        factors = []  # 所有变换因子（例如缩放倍数、旋转角度等）
        methods = []  # 所有变换方法名称

        # 对当前 augment 中所有的变换方法进行遍历
        for transform in aug:
            method, v = transform[0], transform[1]
            img, factor = TTA_aug_single_img(img, method, v)
            factors.append(factor)
            methods.append(method)

        # 将 factors 和 methods 列表中的元素反转，以便逆向执行各个变换
        factors.reverse()
        methods.reverse()

        # 进行预测并还原预测框坐标
        predict_result = inference_detector(model, img)
        for idx, results in enumerate(predict_result):
            if len(results) > 0:
                for result in results:
                    # 还原当前预测框的坐标至原始图像中
                    recover_box = [result[:-1]]
                    for method, factor in zip(methods, factors):
                        recover_box = mapping_bboxes_back(recover_box, method, factor)

                    # 组合所有变换后的预测结果
                    total_bboxes.append(recover_box[0])
                    total_scores.append(result[-1])
                    total_idxs.append(idx)

    # 对所有变换后的预测框进行 NMS 并将结果保存到 merged_result 中
    total_bboxes = np.asarray(total_bboxes)
    total_scores = np.asarray(total_scores)
    total_idxs = np.asarray(total_idxs)
    total_bboxes = torch.from_numpy(total_bboxes).float().to(f"cuda:0")
    total_scores = torch.from_numpy(total_scores).float().to(f"cuda:0")
    total_idxs = torch.from_numpy(total_idxs).int().to(f"cuda:0")
    if len(total_bboxes) == 0:
        merged_dict = dict(name=os.path.basename(img_path), result=predict_result)
        return merged_dict
    else:
        kept_bboxes, keep = batched_nms(total_bboxes, total_scores, total_idxs,
                                        nms_cfg=dict(type='nms', iou_threshold=nms_thr))
        merged_result = bbox2result(kept_bboxes, total_idxs[keep], len(predict_result))
        merged_dict = dict(name=os.path.basename(img_path), result=merged_result)

        return merged_dict


def TTA_aug_single_img(img, method, factor):
    """对单张图片应用指定的数据增强方法.

    Args:
        img (ndarray): 待增强的图像数组.
        method (str): 数据增强方法名称.
        factor (float): 方法强度.

    Returns:
        tuple: 增强后的图像数组和方法强度.
    """
    methods = {
        'TTA_Rotate_no_pad': TTA_Rotate_no_pad,
        'TTA_Brightness': TTA_Brightness,
        'TTA_Flip': TTA_Flip,
        'TTA_Resize': TTA_Resize_mmcv,
        'TTA_Color': TTA_Color,
        'TTA_Contrast': TTA_Contrast,
        'TTA_Sharpness': TTA_Sharpness,
        'TTA_AutoContrast': TTA_AutoContrast,
        'TTA_Equalize': TTA_Equalize,
        'TTA_Invert': TTA_Invert,
        'TTA_Posterize': TTA_Posterize,
        'TTA_Solarize': TTA_Solarize,
        'TTA_HSV': TTA_HSV,
        'TTA_PepperNoise': TTA_PepperNoise,
        'TTA_GaussNoise': TTA_GaussNoise,
        'TTA_Grey': TTA_Grey
    }

    try:
        aug_func = methods[method]
    except KeyError:
        raise ValueError('未实现的数据增强方法')

    aug_img, _, factors = aug_func(img, factor)
    return aug_img, factors


def eval_ap_coco(dataset_config, predict_results):
    dataset_config.test_mode = True
    dataset = build_dataset(dataset_config)
    predict_results = [i['result'] for i in predict_results]
    metric = dataset.evaluate(predict_results)
    return metric['bbox_mAP_50']


def mapping_bboxes_back(bboxes, method, factor):
    """根据指定的方法和强度反向映射边界框.
    Args:
        bboxes (list): 边界框列表.
        method (str): 反向映射方法.
        factor (float): 方法对应强度.
    Returns:
        list: 映射回原始尺寸的边界框列表.
    """
    methods = {
        'TTA_Rotate_no_pad': TTA_Rotate_no_pad_re,
        'TTA_Flip': TTA_Flip_re,
        'TTA_Resize': TTA_Resize_re
    }

    if method in methods:
        return methods[method](bboxes, factor)
    else:
        return bboxes
