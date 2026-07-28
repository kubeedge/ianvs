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

from hashlib import new
from tqdm import tqdm

from sedna.common.class_factory import ClassType, ClassFactory


__all__ = ('map')

import sys
import torch
import random

from collections import Counter
from enum import Enum
from typing import List, Dict

import numpy as np

class MethodAveragePrecision(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.

        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """
    AllPointsInterpolation = 1
    ElevenPointsInterpolation = 2


class Box:
    def __init__(self, xtl: float, ytl: float, xbr: float, ybr: float):
        """
                    0,0 ------> x (width)
             |
             |  (Left,Top)
             |      *_________
             |      |         |
                    |         |
             y      |_________|
          (height)            *
                        (Right,Bottom)

        Args:
            xtl: Float value representing the X top-left coordinate of the bounding box.
            ytl: Float value representing the Y top-left coordinate of the bounding box.
            xbr: Float value representing the X bottom-right coordinate of the bounding box.
            ybr: Float value representing the Y bottom-right coordinate of the bounding box.
        """
        self.xtl = xtl
        self.ytl = ytl
        self.xbr = xbr
        self.ybr = ybr

    @property
    def width(self) -> float:
        return self.xbr - self.xtl

    @property
    def height(self) -> float:
        return self.ybr - self.ytl

    @property
    def area(self) -> float:
        return (self.xbr - self.xtl + 1) * (self.ybr - self.ytl + 1)

    @classmethod
    def intersection_over_union(cls, box1: 'Box', box2: 'Box') -> float:
        """
        Intersection Over Union (IOU) is measure based on Jaccard Index that evaluates the overlap between
        two bounding boxes.
        """
        # if boxes dont intersect
        if not Box.is_intersecting(box1, box2):
            return 0
        intersection = Box.get_intersection_area(box1, box2)
        union = Box.get_union_areas(box1, box2, intersection_area=intersection)
        # intersection over union
        iou = intersection / union
        assert iou >= 0, '{} = {} / {}, {}, {}'.format(iou, intersection, union, box1, box2)
        return iou

    @classmethod
    def is_intersecting(cls, box1: 'Box', box2: 'Box') -> bool:
        if box1.xtl > box2.xbr:
            return False  # boxA is right of boxB
        if box2.xtl > box1.xbr:
            return False  # boxA is left of boxB
        if box1.ybr < box2.ytl:
            return False  # boxA is above boxB
        if box1.ytl > box2.ybr:
            return False  # boxA is below boxB
        return True

    @classmethod
    def get_intersection_area(cls, box1: 'Box', box2: 'Box') -> float:
        xtl = max(box1.xtl, box2.xtl)
        ytl = max(box1.ytl, box2.ytl)
        xbr = min(box1.xbr, box2.xbr)
        ybr = min(box1.ybr, box2.ybr)
        # intersection area
        return (xbr - xtl + 1) * (ybr - ytl + 1)

    @staticmethod
    def get_union_areas(box1: 'Box', box2: 'Box', intersection_area: float = None) -> float:
        if intersection_area is None:
            intersection_area = Box.get_intersection_area(box1, box2)
        return float(box1.area + box2.area - intersection_area)


class BoundingBox(Box):
    def __init__(self, image_name: str, label: str, xtl: float, ytl: float, xbr: float, ybr: float,
                 score=None):
        """Constructor.
        Args:
            image_name: String representing the image name.
            label: String value representing class id.
            xtl: Float value representing the X top-left coordinate of the bounding box.
            ytl: Float value representing the Y top-left coordinate of the bounding box.
            xbr: Float value representing the X bottom-right coordinate of the bounding box.
            ybr: Float value representing the Y bottom-right coordinate of the bounding box.
            score: (optional) Float value representing the confidence of the detected
                class. If detectionType is Detection, classConfidence needs to be informed.
        """
        super().__init__(xtl, ytl, xbr, ybr)
        self.image_name = image_name
        self.score = score
        self.label = label


class MetricPerClass:
    def __init__(self):
        self.label = None
        self.precision = None
        self.recall = None
        self.ap = None
        self.interpolated_precision = None
        self.interpolated_recall = None
        self.num_groundtruth = None
        self.num_detection = None
        self.tp = None
        self.fp = None

    @staticmethod
    def get_mAP(results: Dict[str, 'MetricPerClass']):
        return np.average([m.ap for m in results.values() if m.num_groundtruth > 0])


def get_pascal_voc_metrics(gold_standard: List[BoundingBox],
                           predictions: List[BoundingBox],
                           iou_threshold: float = 0.45,
                           method: MethodAveragePrecision = MethodAveragePrecision.AllPointsInterpolation
                           ) -> Dict[str, MetricPerClass]:
    """Get the metrics used by the VOC Pascal 2012 challenge.

    Args:
        gold_standard: Object of the class BoundingBoxes representing ground truth bounding boxes;
        predictions: Object of the class BoundingBoxes representing detected bounding boxes;
        iou_threshold: IOU threshold indicating which detections will be considered TP or FP (default value = 0.5);
        method: It can be calculated as the implementation
            in the official PASCAL VOC toolkit (EveryPointInterpolation), or applying the 11-point
            interpolation as described in the paper "The PASCAL Visual Object Classes(VOC) Challenge"
            or EveryPointInterpolation"  (ElevenPointInterpolation);
    Returns:
        A list of dictionaries. Each dictionary contains information and metrics of each class.
            The keys of each dictionary are:
                dict['class']: class representing the current dictionary;
                dict['precision']: array with the precision values;
                dict['recall']: array with the recall values;
                dict['AP']: average precision;
                dict['interpolated precision']: interpolated precision values;
                dict['interpolated recall']: interpolated recall values;
                dict['total positives']: total number of ground truth positives;
                dict['total TP']: total number of True Positive detections;
                dict['total FP']: total number of False Negative detections;
    """
    ret = {}  # list containing metrics (precision, recall, average precision) of each class

    # Get all classes
    classes = sorted(set(b.label for b in gold_standard + predictions))

    # Precision x Recall is obtained individually by each class
    # Loop through by classes
    for c in classes:
        preds = [b for b in predictions if b.label == c]  # type: List[BoundingBox]
        golds = [b for b in gold_standard if b.label == c]  # type: List[BoundingBox]
        npos = len(golds)
        # print('npos', npos)

        # sort detections by decreasing confidence
        preds = sorted(preds, key=lambda b: b.score, reverse=True)
        tps = np.zeros(len(preds))
        fps = np.zeros(len(preds))

        # create dictionary with amount of gts for each image
        counter = Counter([cc.image_name for cc in golds])
        for key, val in counter.items():
            counter[key] = np.zeros(val)

        # Loop through detections
        for i in range(len(preds)):
            # Find ground truth image
            gt = [b for b in golds if b.image_name == preds[i].image_name]
            max_iou = sys.float_info.min
            mas_idx = -1
            for j in range(len(gt)):
                iou = Box.intersection_over_union(preds[i], gt[j])
                if iou > max_iou:
                    max_iou = iou
                    mas_idx = j
            # Assign detection as true positive/don't care/false positive
            # print('max iou', max_iou)
            if max_iou >= iou_threshold:
                if counter[preds[i].image_name][mas_idx] == 0:
                    tps[i] = 1  # count as true positive
                    counter[preds[i].image_name][mas_idx] = 1  # flag as already 'seen'
                else:
                    # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
                    fps[i] = 1  # count as false positive
            else:
                fps[i] = 1  # count as false positive
        # compute precision, recall and average precision
        # print('fps', fps)
        # print('tps', tps)
        cumulative_fps = np.cumsum(fps)
        cumulative_tps = np.cumsum(tps)
        # print('cfps', cumulative_fps)
        # print('ctps', cumulative_tps)
        recalls = np.divide(cumulative_tps, npos, out=np.full_like(cumulative_tps, np.nan), where=npos != 0)
        precisions = np.divide(cumulative_tps, (cumulative_fps + cumulative_tps))
        # Depending on the method, call the right implementation
        if method == MethodAveragePrecision.AllPointsInterpolation:
            ap, mpre, mrec, _ = calculate_all_points_average_precision(recalls, precisions)
        else:
            ap, mpre, mrec = calculate_11_points_average_precision(recalls, precisions)
        # print('ap', ap)
        # add class result in the dictionary to be returned
        r = MetricPerClass()
        r.label = c
        r.precision = precisions
        r.recall = recalls
        r.ap = ap
        r.interpolated_recall = mrec
        r.interpolated_precision = mpre
        r.tp = np.sum(tps)
        r.fp = np.sum(fps)
        r.num_groundtruth = len(golds)
        r.num_detection = len(preds)
        ret[c] = r
    return ret


def calculate_all_points_average_precision(recall, precision):
    """
    All points interpolated average precision

    Returns:
        average precision
        interpolated precision
        interpolated recall
        interpolated points
    """
    mrec = [0] + [e for e in recall] + [1]
    mpre = [0] + [e for e in precision] + [0]
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii


def calculate_11_points_average_precision(recall, precision):
    """
    11-point interpolated average precision

    Returns:
        average precision
        interpolated precision
        interpolated recall
    """
    mrec = [e for e in recall]
    mpre = [e for e in precision]
    recall_values = np.linspace(0, 1, 11)
    recall_values = list(recall_values[::-1])
    rho_interp = []
    recall_valid = []
    # For each recallValues (0, 0.1, 0.2, ... , 1)
    for r in recall_values:
        # Obtain all recall values higher or equal than r
        arg_greater_recalls = np.argwhere(mrec[:] >= r)
        pmax = 0
        # If there are recalls above r
        if arg_greater_recalls.size != 0:
            pmax = max(mpre[arg_greater_recalls.min():])
        recall_valid.append(r)
        rho_interp.append(pmax)
    # By definition AP = sum(max(precision whose recall is above r))/11
    ap = sum(rho_interp) / 11
    # Generating values for the plot
    rvals = [recall_valid[0]] + [e for e in recall_valid] + [0]
    pvals = [0] + [e for e in rho_interp] + [0]
    # rhoInterp = rhoInterp[::-1]
    cc = []
    for i in range(len(rvals)):
        p = (rvals[i], pvals[i - 1])
        if p not in cc:
            cc.append(p)
        p = (rvals[i], pvals[i])
        if p not in cc:
            cc.append(p)
    recall_values = [i[0] for i in cc]
    rho_interp = [i[1] for i in cc]
    return ap, rho_interp, recall_values


def get_map(ret, class_tea_list, class_pred_list):
    ap_list = [0 for _ in range(2)]
    for class_index in range(2):
        if (class_index in class_tea_list) and (class_index in class_pred_list):
            # case 1: both prediction and teacher detect the same class, which should be the main case
            ap_list[class_index] = ret[class_index].ap
        elif (class_index not in class_tea_list) and (class_index not in class_pred_list):
            # case 2: both prediction and teacher not detect the class, which is correct and the ap should be 1
            ap_list[class_index] = 1.
        elif ((class_index not in class_tea_list) and (class_index in class_pred_list)) or ((class_index in class_tea_list) and (class_index not in class_pred_list)):
            # case 3: only one detect the class, in all the cases it is not correct
            ap_list[class_index] = 0.
    m_ap = np.mean(ap_list)
    return m_ap


def performance_evaluation_map_teacher_feedback(img, teacher, model, conf_thre=0.6):
    
    teacher_boxes = []
    teacher_results = teacher(img, conf=conf_thre)
    teacher_results = np.array(teacher_results.pandas().xyxy[0])

    box_tea_list = teacher_results[:, 0:4]
    confidence_tea_list = teacher_results[:, 4]
    class_tea_list = teacher_results[:, 5]

    for bb_index in range(len(teacher_results)):
        bb = BoundingBox('filename', class_tea_list[bb_index], box_tea_list[bb_index][0], box_tea_list[bb_index][1], box_tea_list[bb_index][2], box_tea_list[bb_index][3], confidence_tea_list[bb_index])
        teacher_boxes.append(bb)

    results = model(img, conf=conf_thre)
    results = np.array(results.pandas().xyxy[0])

    inference_boxs = []

    # print(results)

    box_pred_list = results[:, 0:4]
    confidence_list = results[:, 4]
    class_pred_list = results[:, 5]

    for bb_index in range(len(results)):
        bb = BoundingBox('filename', class_pred_list[bb_index], box_pred_list[bb_index][0], box_pred_list[bb_index][1], box_pred_list[bb_index][2], box_pred_list[bb_index][3], confidence_list[bb_index])
        inference_boxs.append(bb)

    ret = get_pascal_voc_metrics(teacher_boxes, inference_boxs)

    m_ap = get_map(ret, class_tea_list, class_pred_list)

    return m_ap, results


def get_boxes_with_results(results, from_numpy=False):
    # if from_numpy is False:
    #     results = np.array(results.pandas().xyxy[0])
    boundingbox_list = []
    box_list = results[:, 0:4]
    confidence_list = results[:, 4]
    class_list = results[:, 5]

    for bb_index in range(len(results)):
        bb = BoundingBox('filename', class_list[bb_index], box_list[bb_index][0], box_list[bb_index][1], box_list[bb_index][2], box_list[bb_index][3], confidence_list[bb_index])
        boundingbox_list.append(bb)
    return boundingbox_list, class_list


def get_mAP_with_results(results1, results2, from_numpy=False):
    boxes1, class_list1 = get_boxes_with_results(results1, from_numpy=from_numpy)
    boxes2, class_list2 = get_boxes_with_results(results2, from_numpy=from_numpy)
    ret = get_pascal_voc_metrics(boxes1, boxes2)
    m_ap = get_map(ret, class_list1, class_list2)
    return m_ap


def performance_evaluation_map_teacher(img, teacher, model, conf_thre=0.6):
    
    teacher_boxes = []
    teacher_results = teacher(img, conf=conf_thre)

    teacher_results = np.array(teacher_results.pandas().xyxy[0])

    box_tea_list = teacher_results[:, 0:4]
    confidence_tea_list = teacher_results[:, 4]
    class_tea_list = teacher_results[:, 5]

    for bb_index in range(len(teacher_results)):
        bb = BoundingBox('filename', class_tea_list[bb_index], box_tea_list[bb_index][0], box_tea_list[bb_index][1], box_tea_list[bb_index][2], box_tea_list[bb_index][3], confidence_tea_list[bb_index])
        teacher_boxes.append(bb)

    results = model(img, conf=conf_thre)

    results = np.array(results.pandas().xyxy[0])

    inference_boxs = []

    # print(results)

    box_pred_list = results[:, 0:4]
    confidence_list = results[:, 4]
    class_pred_list = results[:, 5]

    for bb_index in range(len(results)):
        bb = BoundingBox('filename', class_pred_list[bb_index], box_pred_list[bb_index][0], box_pred_list[bb_index][1], box_pred_list[bb_index][2], box_pred_list[bb_index][3], confidence_list[bb_index])
        inference_boxs.append(bb)

    ret = get_pascal_voc_metrics(teacher_boxes, inference_boxs)

    m_ap = get_map(ret, class_tea_list, class_pred_list)
    return m_ap


def performance_fusion_evaluation_map_teacher(img, teacher, model_list, conf_thre=0.6):
    
    teacher_boxes = []
    teacher_results = teacher(img, conf=conf_thre)

    teacher_results = np.array(teacher_results.pandas().xyxy[0])

    box_tea_list = teacher_results[:, 0:4]
    confidence_tea_list = teacher_results[:, 4]
    class_tea_list = teacher_results[:, 5]

    for bb_index in range(len(teacher_results)):
        bb = BoundingBox('filename', class_tea_list[bb_index], box_tea_list[bb_index][0], box_tea_list[bb_index][1], box_tea_list[bb_index][2], box_tea_list[bb_index][3], confidence_tea_list[bb_index])
        teacher_boxes.append(bb)

    results = model_inference_fusion(img, model_list)

    results = np.array(results.pandas().xyxy[0])

    inference_boxs = []

    # print(results)

    box_pred_list = results[:, 0:4]
    confidence_list = results[:, 4]
    class_pred_list = results[:, 5]

    for bb_index in range(len(results)):
        bb = BoundingBox('filename', class_pred_list[bb_index], box_pred_list[bb_index][0], box_pred_list[bb_index][1], box_pred_list[bb_index][2], box_pred_list[bb_index][3], confidence_list[bb_index])
        inference_boxs.append(bb)

    ret = get_pascal_voc_metrics(teacher_boxes, inference_boxs)

    m_ap = get_map(ret, class_tea_list, class_pred_list)
    return m_ap, results


def get_box(img, teacher, model, img_name):

    teacher_boxes_each = []
    inference_boxes_each = []

    teacher_results = teacher(img)
    teacher_results = np.array(teacher_results.pandas().xyxy[0])

    box_tea_list = teacher_results[:, 0:4]
    confidence_tea_list = teacher_results[:, 4]
    class_tea_list = teacher_results[:, 5]

    for bb_index in range(len(teacher_results)):
        tea_bb = BoundingBox(img_name, class_tea_list[bb_index], box_tea_list[bb_index][0], box_tea_list[bb_index][1], box_tea_list[bb_index][2], box_tea_list[bb_index][3], confidence_tea_list[bb_index])  # for one single img, there is only one result
        teacher_boxes_each.append(tea_bb)

    results = model(img)
    results = np.array(results.pandas().xyxy[0])

    box_pred_list = results[:, 0:4]
    confidence_list = results[:, 4]
    class_pred_list = results[:, 5]

    for bb_index in range(len(results)):
        infer_bb = BoundingBox(img_name, class_pred_list[bb_index], box_pred_list[bb_index][0], box_pred_list[bb_index][1], box_pred_list[bb_index][2], box_pred_list[bb_index][3], confidence_list[bb_index])  # for one single img, there is only one result
        inference_boxes_each.append(infer_bb)

    return teacher_boxes_each, inference_boxes_each, class_tea_list, class_pred_list, results


def get_boxes(imgs, teacher, models):
    tea_class_all_img_list = []
    pred_class_all_img_list = []

    teacher_boxes = []
    inference_boxes = []

    img_name_list = random.sample(range(1, 20), 10)  # generation of 10 names for the image naming
    
    for img_index in range(len(imgs)):
        img = imgs[img_index]
        model = models[img_index]  # model index is the same of img_index
        
        tea_bb_list, infer_bb_list, class_tea_list, class_pred_list, _ = get_box(img, teacher, model, str(img_name_list[img_index]))

        teacher_boxes = list(np.concatenate((teacher_boxes, tea_bb_list), axis=0))
        inference_boxes = list(np.concatenate((inference_boxes, infer_bb_list), axis=0))
        
        tea_class_all_img_list = np.concatenate((tea_class_all_img_list, class_tea_list), axis=0)
        pred_class_all_img_list = np.concatenate((pred_class_all_img_list, class_pred_list), axis=0)

    # print(len(inference_boxes[0]))
    tea_class = sorted(list(set(tea_class_all_img_list)))
    pred_class = sorted(list(set(pred_class_all_img_list)))

    return inference_boxes, teacher_boxes, pred_class, tea_class


@ClassFactory.register(ClassType.GENERAL, alias="map")
def map(y_true, y_pred, **kwargs):
    map = []
    assert len(y_pred) == len(y_true)
    space = []
    for i in range(0, len(y_pred)):
        with open(y_true[i], "r")as f:
            text = f.readlines()
        new_l = []
        for item in text:
            new_item= []
            item_split = []
            item_split = item.split("\n")[:-1][0].split(" ")
            for j in range(1,5):
                new_item.append(float(item_split[j]))
            new_item.append(float(1))
            new_item.append(int(float(item_split[0])))
            new_l.append(np.array(new_item))
        if not new_l:
            space.append(i)
            continue
        # y_pred_i = np.array([item.cpu().detach().numpy() for item in y_pred[i]])
        # map.append(get_mAP_with_results(np.array(new_l), np.array(y_pred_i)))
        map.append(get_mAP_with_results(np.array(new_l), np.array(y_pred[i])))
    # with open(y_true, "r")as f:
    #     text = f.readlines()
    return np.mean(map)