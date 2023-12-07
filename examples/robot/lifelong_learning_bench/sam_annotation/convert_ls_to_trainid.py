import json
import numpy as np
from PIL import Image
import cv2
import os
import argparse


def main(args):
    dataset_dir = args.dataset_dir
    ls_json_name = args.ls_json_name
    config_path = args.config_path

    ls_data_dir = os.path.join(dataset_dir, 'ls_data')
    if not os.path.exists(ls_data_dir):
        os.makedirs(ls_data_dir)
    output_trainid_dir = os.path.join(dataset_dir, 'train_ids')
    if not os.path.exists(output_trainid_dir):
        os.makedirs(output_trainid_dir)

    ls_json_path = os.path.join(ls_data_dir, ls_json_name)

    with open(ls_json_path, 'r') as json_file:
        ls_label_json = json.load(json_file)

    with open(config_path, 'r') as config_file:
        label_config = json.load(config_file)

    label_values = list(label_config["id2label"].values())

    for item in ls_label_json:
        image_name = os.path.basename(item["image"]).split("-")[-1]
        image_width = item['labels'][0]['original_width']
        image_height = item['labels'][0]['original_height']
        labels = item["labels"]
        label_map = (np.ones((image_height, image_width), dtype=np.uint16)) * 255
        output_path = os.path.join(output_trainid_dir, f"{os.path.splitext(image_name)[0]}_TrainIds.png")

        for label in labels:
            ls_points = label["points"]
            polygon_label = label["polygonlabels"][0]
            label_id = ""
            if polygon_label in label_values:
                label_id = label_values.index(polygon_label)
            else:
                print("current label is not in the label configuration file.")

            real_points = [[ls_x * image_width / 100.0, ls_y * image_height / 100.0] for [ls_x, ls_y] in ls_points]
            real_points = np.array(real_points, dtype=np.int32)
            cv2.fillPoly(label_map, [real_points], label_id)

        trainid_image = Image.fromarray(label_map)
        trainid_image.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Label Studio JSON exports to segmentation label images for training.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--ls_json_name", type=str, required=True, help="Name of the Label Studio JSON export file.")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the JSON configuration file containing ID-to-label mappings.")

    args = parser.parse_args()
    main(args)
