import json
import os
import cv2
from mmdet.apis import inference_detector, init_detector
from custom_code.instance_based.utils.transform_unkonwn import xyxy_to_xywh
from tqdm import tqdm
from custom_code.test_time_aug.general_TTA_v4 import topleftxywh_to_xyxy

CATEGORY = [{"id": 0, "name": "yanse"},
            {"id": 1, "name": "huahen"},
            {"id": 2, "name": "mosun"},
            ]


def compute_iou(rec_1, rec_2):
    """
    计算两个矩形框的交并比(IOU)
    :param rec_1: 矩形框1，格式为 (x_min, y_min, x_max, y_max)
    :param rec_2: 矩形框2，格式为 (x_min, y_min, x_max, y_max)
    :return: 交并比(IOU)
    """
    # 计算两个矩形框的面积
    s_rec1 = (rec_1[2] - rec_1[0]) * (rec_1[3] - rec_1[1])
    s_rec2 = (rec_2[2] - rec_2[0]) * (rec_2[3] - rec_2[1])
    sum_s = s_rec1 + s_rec2

    # 计算两个矩形框的重叠部分面积
    left = max(rec_1[0], rec_2[0])
    right = min(rec_1[2], rec_2[2])
    bottom = max(rec_1[1], rec_2[1])
    top = min(rec_1[3], rec_2[3])
    if left >= right or top <= bottom:
        return 0
    else:
        inter = (right - left) * (top - bottom)

        # 计算交并比
        iou = (inter / (sum_s - inter)) * 1.0
        return iou


def compute_error(predict_result_json):
    new_json = []
    iou_0_num = 0  # 标注框没有任何预测框的个数
    label_num = 0  # 标注框总个数
    with open(predict_result_json) as f:
        info = json.load(f)
        for key in info.keys():
            output = []
            output.append(key)
            preds = info[key]['predict']  # 预测框信息
            labels = info[key]['label']  # 标注框信息(x_min, y_min, x_max, y_max, c)
            for label in labels:
                new_label = label.copy()
                xyxy = label[:4]  # 取出标注框的坐标信息(x_min, y_min, x_max, y_max)
                c = label[4]  # 取出标注框的类别信息
                max_iou = [0, 0, 0]  # 最高的交并比，对应的预测框，对应预测框的类别概率

                # 对于每个标注框，遍历所有的预测框，找到最高的交并比以及对应的预测框和类别概率
                for i, pred in enumerate(preds):
                    for pred_box in pred:
                        if pred_box:
                            iou = compute_iou(pred_box[:4], xyxy)  # 计算交并比
                            if iou > max_iou[0]:
                                max_iou[0] = iou
                                max_iou[1] = i
                                max_iou[2] = pred_box[4]

                if max_iou == [0, 0, 0]:  # 如果找不到任何预测框和标注框重叠，认为交并比为0
                    iou_0_num += 1
                    error = 2  # 认为错误程度最大，用2表示
                    new_label.append(error)
                else:
                    error_iou = abs(1 - max_iou[0])  # 交并比的误差
                    if c == max_iou[1]:  # 如果预测的类别和标注的类别一致
                        error_c = abs(1 - max_iou[2])  # 类别概率的误差
                    else:
                        error_c = 1  # 类别不同，则认为是最大误差，用1表示
                    error = error_iou + error_c  # 误差综合
                    new_label.append(error)

                output.append(new_label)  # 把该标注框的信息加入到输出结果列表中
                label_num += 1
            new_json.append(output)  # 把该图片的输出结果加入到整个输出结果列表中

    # 输出统计信息
    print('标注框没有任何预测框的个数', iou_0_num)
    print('标注框总个数', label_num)

    return new_json


def get_new_train_json(predict_results, img_path, known_name_list, unknown_name_list, out_dir):
    fp_known = open(known_name_list, mode="r", encoding="utf-8")
    fp_unknown = open(unknown_name_list, mode="r", encoding="utf-8")
    known_result = fp_known.readlines()
    unknown_result = fp_unknown.readlines()
    for i in range(len(known_result)):
        known_result[i] = known_result[i][:-1]
    for i in range(len(unknown_result)):
        unknown_result[i] = unknown_result[i][:-1]
    images = []
    annotations = []
    id_num = 0
    image_id = 0
    for item in tqdm(predict_results):
        img_name = item[0]
        img_id = image_id
        image = {}
        image['file_name'] = img_name
        image["id"] = img_id
        height, width = cv2.imread(img_path + "/" + img_name).shape[:2]
        image["height"] = height
        image["width"] = width
        images.append(image)
        image_id+=1
        for i in range(1, len(item)):
            bbox_item = item[i]
            bbox = xyxy_to_xywh([bbox_item[0], bbox_item[1], bbox_item[2], bbox_item[3]])
            error = bbox_item[5]
            annotation = {}
            annotation["image_id"] = img_id
            annotation["bbox"] = bbox
            annotation["category_id"] = bbox_item[4]
            if img_name in known_result:
                annotation["weight"] = (2 - error) + 1
            elif img_name in unknown_result:
                annotation["weight"] = error + 1
            annotation["id"] = id_num
            annotation["iscrowd"] = 0
            annotation["segmentation"] = []
            annotation["area"] = bbox[2] * bbox[3]
            id_num += 1
            annotations.append(annotation)
    dataset_dict = {}
    dataset_dict["images"] = images
    dataset_dict["annotations"] = annotations
    dataset_dict["categories"] = CATEGORY
    json_str = json.dumps(dataset_dict)
    with open(out_dir, 'w') as json_file:
        json_file.write(json_str)
    print("json file write done...")


def infer_anno(config_file, checkpoint_file, img_path, anno_path, out_path):
    dir=os.path.dirname(out_path)
    if not os.path.exists(dir):
        os.mkdir(dir)
    fp = open(anno_path, 'r', encoding='utf8')
    label_json = json.load(fp)
    annos = label_json['annotations']
    image_infos = label_json['images']
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    predict_results = {}
    for image_info in tqdm(image_infos):
        image_name = image_info['file_name']
        image_id = image_info['id']
        predict_result = {}
        image = os.path.join(img_path, image_name)
        model_out = inference_detector(model, image)
        for i in range(len(model_out)):
            model_out[i] = model_out[i].tolist()
        predict_result["predict"] = model_out
        bboxes = []
        for anno in annos:
            anno_belong_image = anno['image_id']
            if anno_belong_image == image_id:
                bbox = topleftxywh_to_xyxy(anno['bbox'])
                bbox.append(anno['category_id'])
                bboxes.append(bbox)
        predict_result["label"] = bboxes
        predict_results[image_name] = predict_result
    json_str = json.dumps(predict_results)
    with open(out_path, 'w') as json_file:
        json_file.write(json_str)


def merge_predict_results(result1, result2, out_dir):
    new_predict_result = {}
    fp1 = open(result1, 'r', encoding='utf8')
    fp2 = open(result2, 'r', encoding='utf8')
    json1 = json.load(fp1)
    json2 = json.load(fp2)
    for item in json1.keys():
        result = json1[item]
        new_predict_result[item] = result
    for item in json2.keys():
        result = json2[item]
        new_predict_result[item] = result
    json_str = json.dumps(new_predict_result)
    with open(out_dir, 'w') as json_file:
        json_file.write(json_str)


def gen_txt_according_json(label_json, out_dir):
    fp = open(label_json, 'r', encoding='utf8')
    anno = json.load(fp)
    w = open(out_dir, 'a+', encoding='utf8')
    images = anno['images']
    for image in images:
        w.write(image['file_name'])
        w.write('\n')
