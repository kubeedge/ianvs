import logging
import numpy as np
from tqdm import tqdm
import torch
from pathlib import Path
from ultralytics.utils.metrics import ap_per_class, box_iou
from sedna.common.class_factory import ClassType, ClassFactory

logger = logging.getLogger(__name__)
__all__ = ('map90')

def xywh2xyxy_rel(xywh):
    """
    Convert relative
      [x_center, y_center, w, h] 
      to
      absolute [x1, y1, x2, y2]
    """
    x_center, y_center, w, h = xywh
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    return [x1, y1, x2, y2]

def read_label_file(label_path, img_width=640, img_height=480):
    """
        Read YOLO txt labels files，,
        convert to absolute coordinates
        return :list[list[...]]
    """
    if not Path(label_path).exists():
        return []  # empty labels return empty list []

    boxes = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x_c, y_c, w, h = map(float, parts)
        x1, y1, x2, y2 = xywh2xyxy_rel([x_c, y_c, w, h])
        boxes.append([x1 * img_width, y1 * img_height,
                      x2 * img_width, y2 * img_height,
                      1.0, int(cls)])
    return boxes  # return  list[list[...]]

@ClassFactory.register(ClassType.GENERAL, alias="map90")
def map90(y_true_paths, y_pred, **kwargs):
    """
    compute mAP90 
    y_pred : class: dict: 
    Args:
        y_true_paths: labels files path(numpy.ndarray 或 list)
        y_pred: dict, key: images path, value: predictions [[x1,y1,x2,y2,conf,cls], ...]
    """
    img_width=640
    img_height=480

    stats = []

    if isinstance(y_true_paths, np.ndarray):
        y_true_paths = y_true_paths.tolist()

    for label_path in tqdm(y_true_paths, desc="Evaluating"):
        label_path = Path(label_path)
        # read the true labels
        y_true_i = read_label_file(label_path, img_width=img_width, img_height=img_height)
        y_true_i = np.array(y_true_i, dtype=np.float32) if len(y_true_i) > 0 else np.zeros((0, 6), dtype=np.float32)

        # construct the image path
        dataset_root = label_path.parents[2]        # .../RoboDK_Palletizing_Dataset
        sub_dir = label_path.parent.name            # test / train / val
        img_path = str(dataset_root / "images" / sub_dir / (label_path.stem + ".png"))

        # from the y_pred dict attain the predictions
        pred_list = y_pred.get(img_path, [])
        y_pred_i = np.array(pred_list, dtype=np.float32) if len(pred_list) > 0 else np.zeros((0, 6), dtype=np.float32)

        # Extract real boxes
        tbox = y_true_i[:, :4] if y_true_i.shape[0] > 0 else np.zeros((0, 4), dtype=np.float32)
        tcls = y_true_i[:, 5].astype(int) if y_true_i.shape[0] > 0 else np.zeros((0,), dtype=int)

        # Extract predict boxes
        pbox = y_pred_i[:, :4] if y_pred_i.shape[0] > 0 else np.zeros((0, 4), dtype=np.float32)
        pcls = y_pred_i[:, 5].astype(int) if y_pred_i.shape[0] > 0 else np.zeros((0,), dtype=int)
        pconf = y_pred_i[:, 4] if y_pred_i.shape[0] > 0 else np.zeros((0,), dtype=np.float32)

        if len(tbox) == 0 and len(pbox) == 0:
            continue

        correct = np.zeros(len(pbox), dtype=bool)
        if len(tbox) > 0 and len(pbox) > 0:
            ious = box_iou(torch.from_numpy(pbox), torch.from_numpy(tbox)).numpy()
            sorted_idx = np.argsort(pconf)[::-1]
            pbox = pbox[sorted_idx]
            pcls = pcls[sorted_idx]
            ious = ious[sorted_idx]
            correct = correct[sorted_idx]
            for j, pred_cls in enumerate(pcls):
                i = np.argmax(ious[j])
                if ious[j, i] >= 0.9 and pred_cls == tcls[i]:
                    correct[j] = True
                    ious[:, i] = 0

        # convert correct to array of 2 dimentions, make sure ap_per_class not crossing boundaries
        stats.append((correct[:, None], pconf, pcls, tcls))

    # merge stats and compute AP
    if len(stats) == 0:
        return 0.0

    tp, conf, pred_cls, target_cls = [np.concatenate(x, 0) for x in zip(*stats)]
    #Extract AP array by index
    #Call ap_per_class to retrieve all 12 return values
    all_return = ap_per_class(tp, conf, pred_cls, target_cls)
    # From return extract ap array
    ap = all_return[5]

    # compute mAP50：The 0th column of the ap array corresponds to IoU=0.5, take the average of all categories
    map90_value = ap[:, 0].mean() if ap.size > 0 else 0.0
    logger.info(f"mAP@90: {map90_value:.4f}")

    return map90_value  