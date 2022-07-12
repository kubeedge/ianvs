import numpy as np
from FPN_TensorFlow.libs.label_name_dict.label_dict import NAME_LABEL_MAP
from FPN_TensorFlow.data.io.read_tfrecord import convert_labels
from FPN_TensorFlow.interface import get_single_label_dict, single_label_eval
from sedna.common.class_factory import ClassType, ClassFactory

__all__ = ["f1_score"]


@ClassFactory.register(ClassType.GENERAL, "f1_score")
def f1_score(y_true, y_pred):
    predict_dict = {}

    for k, v in y_pred.items():
        k = f"b'{k}'"
        if not predict_dict.get(k):
            predict_dict[k] = v

    gtboxes_dict = convert_labels(y_true)

    R, P, AP, F, num = [], [], [], [], []

    for label in NAME_LABEL_MAP.keys():
        if label == 'back_ground':
            continue

        rboxes, gboxes = get_single_label_dict(predict_dict, gtboxes_dict, label)
        # print('label',label)
        rec, prec, ap, box_num = single_label_eval(rboxes, gboxes, 0.3, False)
        # print("rec",rec)
        # print("prec", prec)
        recall = 0 if rec.shape[0] == 0 else rec[-1]
        precision = 0 if prec.shape[0] == 0 else prec[-1]
        F_measure = 0 if not (recall + precision) else (2 * precision * recall / (recall + precision))
        print('\n{}\tR:{}\tP:{}\tap:{}\tF:{}'.format(label, recall, precision, ap, F_measure))
        R.append(recall)
        P.append(precision)
        AP.append(ap)
        F.append(F_measure)
        num.append(box_num)
    print("num:", num)
    R = np.array(R)
    P = np.array(P)
    AP = np.array(AP)
    F = np.array(F)
    num = np.array(num)
    weights = num / np.sum(num)
    Recall = np.sum(R) / 2
    Precision = np.sum(P) / 2
    mAP = np.sum(AP) / 2
    F_measure = np.sum(F) / 2
    print('\n{}\tR:{}\tP:{}\tmAP:{}\tF:{}'.format('Final', Recall, Precision, mAP, F_measure))

    return Recall
