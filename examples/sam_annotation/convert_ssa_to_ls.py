import pycocotools.mask as mask_utils
import cv2
import argparse
import json
import os

try:
    from label_studio_converter.imports.label_config import generate_label_config
except:
    raise ModuleNotFoundError(
        "label_studio_converter is not installed, run `pip install label_studio_converter` to install")


def new_task(out_type, root_url, file_name):
    """create new task with Label Studio format
    copy from: https://github.com/heartexlabs/label-studio-converter/blob/master/label_studio_converter/imports/coco.py

    Args:
        out_type (str): labeling out_type in Label Studio.
        root_url (str): image root_url.
        file_name (str): image file_name.

    Returns:
        dict: task info dict
    """
    return {
        "data": {"image": root_url + file_name},
        # 'annotations' or 'predictions'
        out_type: [
            {
                "model_version": "one",
                "result": [],
                "ground_truth": False,
            }
        ],
    }


def create_segmentation(key, id2label, from_name, to_name, points, image_height, image_width):
    label = id2label['id2label'].get(key)

    item = {
        "id": label,
        "type": "polygonlabels",
        "value": {"points": points, "polygonlabels": [label]},
        "to_name": to_name,
        "from_name": from_name,
        "image_rotation": 0,
        "original_width": image_width,
        "original_height": image_height,
    }
    return item


def convert_rle_to_polygon(segmentation):
    mask = mask_utils.decode(segmentation)

    height_ori, width_ori = segmentation["size"]

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    poly_point = []
    ls_poly_point = []

    for contour in contours:
        if contour.size >= 6:
            poly_point.append(contour[:, 0, :].tolist())

    if poly_point:
        poly_point = poly_point[0]
        ls_poly_point = [[x / width_ori * 100.0, y / height_ori * 100.0] for [x, y] in poly_point]

    return ls_poly_point


def main(args):
    dataset_dir = args.dataset_dir
    config_path = args.config_path

    out_type = "predictions"
    root_url = "/data/local-files/?d="

    raw_data_dir = os.path.join(dataset_dir, 'raw_data')

    dataset_root = os.environ.get("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT")

    rel_dir = os.path.relpath(raw_data_dir, dataset_root)
    rel_dir = rel_dir.replace(os.path.sep, "/")

    ssa_result_dir = os.path.join(dataset_dir, 'ssa_output')
    if not os.path.exists(ssa_result_dir):
        os.makedirs(ssa_result_dir)

    ls_data_dir = os.path.join(dataset_dir, 'ls_data')
    if not os.path.exists(ls_data_dir):
        os.makedirs(ls_data_dir)

    with open(config_path, 'r') as config_file:
        id2label = json.load(config_file)

    image_files = [f for f in os.listdir(raw_data_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.JPG'))]

    for image_file in image_files:
        ssa_result_path = os.path.join(ssa_result_dir, image_file.replace(".png", "_semantic.json"))
        poly_ls_result_path = os.path.join(ls_data_dir, image_file.replace(".png", "_semantic_poly_ls.json"))

        file_name = rel_dir + '/' + image_file
        task = new_task(out_type, root_url, file_name)

        with open(ssa_result_path, 'r') as f:
            ssa_data = json.load(f)

        for key, value in ssa_data["semantic_mask"].items():
            segmentation = value
            image_height, image_width = segmentation["size"]
            points = convert_rle_to_polygon(segmentation)

            if points:
                item = create_segmentation(
                    key,
                    id2label,
                    'labels',
                    'image',
                    points,
                    image_height,
                    image_width
                )
                task[out_type][0]['result'].append(item)

        with open(poly_ls_result_path, 'w') as out:
            json.dump(task, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Converts Semantic Segment Anything model output JSON to Label Studio input data format.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the JSON configuration file containing ID-to-label mappings.")

    args = parser.parse_args()
    main(args)
