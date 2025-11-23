<<<<<<< HEAD
import json
import os
import cv2
import numpy as np
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from utils.c1c.cal_json_file_MAP import topleftxywh_to_xyxy

"""
接口参考map计算接口，评估一个模型在某json文件下的F1-score
"""
THR = 0.5
SCORE_THR = 0.9


def _single_F1score(predict_result, labels, thr=THR):
    """
    计算单张图片的F1-score
    """
    mis_classify = 0
    mis_iou = 0
    all_mis = 0
    TP = 0
    predict_count = 0
    FN = np.zeros(len(labels))
    for index, cur_category_result in enumerate(predict_result):
        if cur_category_result.shape[0] == 0:
            continue
        else:
            for each_result in cur_category_result:
                predict_box = each_result[:-1]
                confidence = each_result[-1]
                if confidence < 0.3:
                    continue
                max_iou = 0
                label_category = -1
                flag = -1
                for j, label in enumerate(labels):
                    label_box = label[:-1]
                    iou = compute_iou(predict_box, label_box)
                    if iou > max_iou:
                        max_iou = iou
                        label_category = label[-1]
                        flag = j  # 标记是哪个标签框被成功预测了
                if max_iou > thr:
                    if index == label_category:
                        TP += 1
                        FN[flag] = 1
                    else:
                        mis_classify += 1
                elif max_iou < thr and index == label_category:
                    mis_iou += 1
                elif max_iou < thr and index != label_category:
                    all_mis += 1
                predict_count += 1
    FP = predict_count - TP
    FN = len(labels) - np.count_nonzero(FN)
    precision = TP / (TP + FP + 1e-5)
    recall = TP / (TP + FN + 1e-5)
    f1score = 2 * precision * recall / (precision + recall + 1e-5)
    return f1score, precision, recall


def F1score(config_file, checkpoint_file, img_path, anno_path, out_path):
    fp = open(anno_path, 'r', encoding='utf8')
    loaded_json = json.load(fp)
    annos = loaded_json['annotations']
    images = loaded_json['images']
    fp.close()
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    image_name_id_list = []
    for i in images:
        image_name_id_list.append((i['file_name'], i['id']))
    for item in tqdm(image_name_id_list):
        image_name = item[0]
        image_id = item[1]
        image = os.path.join(img_path, image_name)
        predict_result = inference_detector(model, image)
        label = []
        for anno in annos:
            anno_belong_img_id = anno['image_id']
            if anno_belong_img_id == image_id:
                bbox = topleftxywh_to_xyxy(anno['bbox'])
                category = anno['category_id']
                bbox.append(category)
                label.append(bbox)
        f1score, precision, recall = _single_F1score(predict_result, label, THR)
        if precision>=SCORE_THR and recall>=SCORE_THR:
            write_names(out_path, image_name, True)
        else:
            write_names(out_path, image_name, False)
        write_txt(out_path, image_name, f1score)


def write_txt(path, name, score):
    txt = open(path, mode="a+", encoding='utf-8')
    txt.write(name)
    txt.write("    ")
    txt.write(str(score))
    txt.write('\n')
    txt.close()


def write_names(path, name, qualified):
    dir_name = os.path.dirname(path)
    if qualified:
        txt = open(os.path.join(dir_name, "pass.txt"), mode="a+", encoding='utf-8')
    else:
        txt = open(os.path.join(dir_name, "NG.txt"), mode="a+", encoding='utf-8')
    txt.write(name)
    txt.write('\n')
    txt.close()


def compute_iou(rec_1, rec_2):
    s_rec1 = (rec_1[2] - rec_1[0]) * (rec_1[3] - rec_1[1])
    s_rec2 = (rec_2[2] - rec_2[0]) * (rec_2[3] - rec_2[1])
    sum_s = s_rec1 + s_rec2
    left = max(rec_1[0], rec_2[0])
    right = min(rec_1[2], rec_2[2])
    bottom = max(rec_1[1], rec_2[1])
    top = min(rec_1[3], rec_2[3])
    if left >= right or top <= bottom:
        return 0
    else:
        inter = (right - left) * (top - bottom)
        iou = (inter / (sum_s - inter)) * 1.0
        return iou


if __name__ == '__main__':
    F1score(
        config_file="/custom_code/instance_based/model/fpn/known/faster_rcnn_r50_fpn_1x_yaoba.py",
        checkpoint_file="/custom_code/instance_based/model/fpn/known/epoch_48.pth",
        img_path='/media/huawei_YaoBa/Images',
        anno_path='/custom_code/instance_based/json/known_test.json',
        out_path='/custom_code/instance_based/txt/known_part/f1score.txt')
=======
version https://git-lfs.github.com/spec/v1
oid sha256:f8d583bb644b36b93e2cb313886a81fa16230f74e5f492d6831515793141835a
size 4728
>>>>>>> 9676c3e (ya toh aar ya toh par)
