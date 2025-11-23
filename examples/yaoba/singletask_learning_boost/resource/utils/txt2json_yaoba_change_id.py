<<<<<<< HEAD
import json
from tqdm import tqdm
import os
import random

category_list = ['yanse', 'huahen', 'mosun']


def xyxy_to_xywh(boxes):
    width = abs(boxes[2] - boxes[0])
    height = abs(boxes[3] - boxes[1])
    if boxes[0] < boxes[2]:
        top_left_x = boxes[0]
    else:
        top_left_x = boxes[2]
    if boxes[1] < boxes[3]:
        top_left_y = boxes[1]
    else:
        top_left_y = boxes[3]
    return [top_left_x, top_left_y, width, height]


def Convert_ZhouCheng_to_COCO(anno_dir, name_txt, output_dir):
    categories = [{"id": 0, "name": "yanse"},
                  {"id": 1, "name": "huahen"},
                  {"id": 2, "name": "mosun"},
                  ]
    with open(name_txt, mode="r", encoding='utf-8') as fp:
        results = fp.readlines()
        name_list = []
        for item in results:
            item = item[:-1]
            name_list.append(item)
    fp.close()
    images = []
    annotations = []
    id_num = 0
    image_id = 10000
    for item in tqdm(name_list):
        image = {}
        image["file_name"] = item
        image["id"] = image_id
        with open(os.path.join(anno_dir, item.replace(".jpg", ".json")), 'r', encoding='utf8') as fp:
            json_data = json.load(fp)
            height, width = json_data["imageHeight"], json_data["imageWidth"]
            image["height"] = height
            image["width"] = width
            images.append(image)
            for v in json_data['shapes']:
                annotation = {}
                label = v['label']
                if len(v['points']) == 1:
                    continue
                else:
                    bbox = [int(v['points'][0][0]), int(v['points'][0][1]), int(v['points'][1][0]),
                            int(v['points'][1][1])]
                    bbox = xyxy_to_xywh(bbox)
                    annotation["image_id"] = image_id
                    annotation["bbox"] = bbox
                    annotation["category_id"] = category_list.index(label)
                    annotation["id"] = id_num
                    annotation["iscrowd"] = 0
                    annotation["weight"] = random.random()
                    annotation["segmentation"] = []
                    annotation["area"] = bbox[2] * bbox[3]
                    id_num += 1
                    annotations.append(annotation)
        image_id += 1
    dataset_dict = {}
    dataset_dict["images"] = images
    dataset_dict["annotations"] = annotations
    dataset_dict["categories"] = categories
    json_str = json.dumps(dataset_dict)
    with open(output_dir, 'w') as json_file:
        json_file.write(json_str)


if __name__ == '__main__':
    Convert_ZhouCheng_to_COCO(anno_dir=r"/media/huawei_YaoBa/Annotations",
                              name_txt=r"/home/wjj/wjj/Public/code/huawei/custom_code/instance_based/txt/merged_part/NG_test.txt",
                              output_dir=r'/custom_code/instance_based/json/NG_test.json')
=======
version https://git-lfs.github.com/spec/v1
oid sha256:1a21ee93be777395bce342135375e204bfe80b5d05887dee7037829b81a8d84e
size 3029
>>>>>>> 9676c3e (ya toh aar ya toh par)
