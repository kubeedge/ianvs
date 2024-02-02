import json

from mmdet.datasets import build_dataset
from sedna.common.class_factory import ClassType, ClassFactory
__all__ = ["map"]


@ClassFactory.register(ClassType.GENERAL, alias="map")
def map(y_true, y_pred):
    img_prefix = y_true[0]
    ann_file = y_true[1]
    fp = open(ann_file, mode="r", encoding="utf-8")
    test_anno_json = json.load(fp)
    categories = test_anno_json["categories"]
    categories = [i['name'] for i in categories]
    test_cfg = dict(
        type='CocoDataset',
        classes=categories,
        ann_file=ann_file,
        img_prefix=img_prefix,
        pipeline=[],
        test_mode=True
    )
    dataset = build_dataset(test_cfg)
    real_eval_items = list(y_pred.keys())
    predict_results = []
    for item in real_eval_items:
        cur_predict = y_pred[item]
        new_predict = []
        for result in cur_predict:
            bbox = result['bbox']
            new_predict.append(bbox)
        predict_results.append(new_predict)
    metric = dataset.evaluate(predict_results)
    return metric['bbox_mAP_50']
