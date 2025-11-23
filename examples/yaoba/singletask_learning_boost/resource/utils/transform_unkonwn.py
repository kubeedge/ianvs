<<<<<<< HEAD
import json
import shutil
import os.path
import copy
from tqdm import tqdm

from .TTA_augs_xyxy_cv2 import *


def topleftxywh_to_xyxy(boxes):
    """
    args:
        boxes:list of topleft_x,topleft_y,width,height,
    return:
        boxes:list of x,y,x,y,cooresponding to top left and bottom right
    """
    x_top_left = boxes[0]
    y_top_left = boxes[1]
    x_bottom_right = boxes[0] + boxes[2]
    y_bottom_right = boxes[1] + boxes[3]
    return [x_top_left, y_top_left, x_bottom_right, y_bottom_right]


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


def aug_image_bboxes_single(img_path, bboxes, labels, aug, image_id, anno_id, out_path):
    new_bboxes = []
    bbox_category = []
    for i in range(len(bboxes)):
        x, y, w, h = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]
        xyxy_box = topleftxywh_to_xyxy([x, y, w, h])
        bbox_category.append(labels[i])
        new_bboxes.append(xyxy_box)
    img = mmcv.imread(img_path)
    method, v = aug[0], aug[1]
    if method == "flip":
        aug_img, aug_bboxes, _ = TTA_Flip(img, v=v, bboxes=new_bboxes)
    elif method == "rotate":
        aug_img, aug_bboxes, _ = TTA_Rotate_no_pad(img, v=v, bboxes=new_bboxes)
    elif method == "brightness":
        aug_img, aug_bboxes, _ = TTA_Brightness(img, v=v, bboxes=new_bboxes)
    elif method == "resize":
        aug_img, aug_bboxes, _ = TTA_Resize(img, v=v, bboxes=new_bboxes)
    else:
        raise ValueError("not implement")
    aug_bboxes = [xyxy_to_xywh(aug_bbox) for aug_bbox in aug_bboxes]
    image = {}
    img_name = os.path.basename(img_path)
    if img_name.endswith("jpg"):
        img_name = img_name.replace(".jpg", "")
    elif img_name.endswith("jpeg"):
        img_name = img_name.replace(".jpeg", "")
    else:
        raise ValueError("current code only support .jpg or .jpeg")
    aug_image_name = img_name + f"_{method}_{str(v)}.jpg"
    height, width = img.shape[:2]
    image["file_name"] = aug_image_name
    image["height"] = height
    image["width"] = width
    image["id"] = image_id
    annotations=[]
    for i in range(len(aug_bboxes)):
        annotation = {}
        annotation["image_id"] = image_id
        annotation["bbox"] = aug_bboxes[i]
        annotation["category_id"] = bbox_category[i]
        annotation["id"] = anno_id
        annotation["iscrowd"] = 0
        annotation["segmentation"] = []
        annotation["area"] = aug_bboxes[i][2] * aug_bboxes[i][3]
        annotations.append(annotation)
        anno_id += 1
    mmcv.imwrite(aug_img, os.path.join(out_path, aug_image_name))
    return image, annotations, anno_id


def aug_image_bboxes(anno, augs, image_path, out_path):
    if not os.path.exists(os.path.join(out_path,"aug_img_folder")):
        os.mkdir(os.path.join(out_path,"aug_img_folder"))
    aug_img_folder=os.path.join(out_path,"aug_img_folder")
    fp = open(anno, mode="r", encoding="utf-8")
    anno_json = json.load(fp)
    fp.close()
    images = anno_json['images']
    annotations = anno_json['annotations']
    categories = anno_json['categories']
    new_images = copy.deepcopy(images)
    new_annotations = copy.deepcopy(annotations)
    next_img_id = new_images[-1]['id'] + 1
    next_bbox_id = new_annotations[-1]['id'] + 1

    for item in os.listdir(image_path):
        shutil.copy(os.path.join(image_path,item),aug_img_folder)
    for image in tqdm(images):
        image_name, img_id = image['file_name'], image['id']
        bboxes = []
        labels = []
        for annotation in annotations:
            if annotation['image_id'] == img_id:
                bboxes.append(annotation['bbox'])
                labels.append(annotation['category_id'])
        for aug in augs:
            image, annotation, next_bbox_id = aug_image_bboxes_single(os.path.join(image_path, image_name),
                                                                           bboxes,
                                                                           labels,
                                                                           aug,
                                                                           next_img_id,
                                                                           next_bbox_id,
                                                                           aug_img_folder,
                                                                           )
            new_images.append(image)
            for i in annotation:
                new_annotations.append(i)
            next_img_id += 1
    dataset_dict = {}
    dataset_dict["images"] = new_images
    dataset_dict["annotations"] = new_annotations
    dataset_dict["categories"] = categories
    json_str = json.dumps(dataset_dict)
    with open(os.path.join(out_path, "aug_unknown.json"), 'w') as json_file:
        json_file.write(json_str)
    return aug_img_folder

def merge_two_anno(anno1, anno2, out_dir):
    fp1 = open(anno1, mode="r", encoding="utf-8")
    anno_json1 = json.load(fp1)
    fp1.close()
    fp2 = open(anno2, mode="r", encoding="utf-8")
    anno_json2 = json.load(fp2)
    fp2.close()
    categories = anno_json1['categories']
    anno1_images = anno_json1['images']
    anno1_bbox = anno_json1['annotations']
    anno2_images = anno_json2['images']
    anno2_bbox = anno_json2['annotations']
    next_image_id = anno1_images[-1]['id'] + 1
    next_bbox_id = anno1_bbox[-1]['id'] + 1
    for image in anno2_images:
        old_image_id = image['id']
        for bbox in anno2_bbox:
            if bbox['image_id'] == old_image_id:
                bbox['id'] = next_bbox_id
                bbox['image_id'] = next_image_id
                anno1_bbox.append(bbox)
                next_bbox_id += 1
        image['id'] = next_image_id
        anno1_images.append(image)
        next_image_id += 1
    dataset_dict = {}
    dataset_dict["images"] = anno1_images
    dataset_dict["annotations"] = anno1_bbox
    dataset_dict["categories"] = categories
    json_str = json.dumps(dataset_dict)
    with open(os.path.join(out_dir, "merged.json"), 'w') as json_file:
        json_file.write(json_str)
=======
version https://git-lfs.github.com/spec/v1
oid sha256:0a835c16dd64cd27d1e2e14dbaec2787a35868f40dd99d292fe84288a23f4e4a
size 6419
>>>>>>> 9676c3e (ya toh aar ya toh par)
