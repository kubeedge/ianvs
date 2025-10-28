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

from FPN_TensorFlow.libs.label_name_dict.label_dict import NAME_LABEL_MAP
from FPN_TensorFlow.data.io.read_tfrecord import convert_labels
from FPN_TensorFlow.help_utils.tools import get_single_label_dict, single_label_eval
from sedna.common.class_factory import ClassType, ClassFactory

__all__ = ["f1_score"]


@ClassFactory.register(ClassType.GENERAL, alias="f1_score")
def f1_score(y_true, y_pred):
    predict_dict = {}

    for k, v in y_pred.items():
        k = f"b'{k}'"
        if not predict_dict.get(k):
            predict_dict[k] = v

    gtboxes_dict = convert_labels(y_true)

    f1_score_list = []

    for label in NAME_LABEL_MAP.keys():
        if label == 'back_ground':
            continue

        rboxes, gboxes = get_single_label_dict(predict_dict, gtboxes_dict, label)
        rec, prec, ap, box_num = single_label_eval(rboxes, gboxes, 0.3, False)
        recall = 0 if rec.shape[0] == 0 else rec[-1]
        precision = 0 if prec.shape[0] == 0 else prec[-1]
        f1_score = 0 if not (recall + precision) else (2 * precision * recall / (recall + precision))

        f1_score_list.append(f1_score)

    f1_score_avg = 0
    if f1_score_list:
        f1_score_avg = round(float(sum(f1_score_list)) / len(f1_score_list), 4)
        
    print(f"f1_score_avg: {f1_score_avg}")

    return f1_score_avg
