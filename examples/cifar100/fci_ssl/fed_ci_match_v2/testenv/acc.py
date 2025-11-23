<<<<<<< HEAD
# Copyright 2021 The KubeEdge Authors.
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

import tensorflow as tf
import numpy as np
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ["acc"]


@ClassFactory.register(ClassType.GENERAL, alias="accuracy")
def accuracy(y_true, y_pred, **kwargs):
    y_pred_arr = [val for val in y_pred.values()]
    y_true_arr = []
    for i in range(len(y_pred_arr)):
        y_true_arr.append(np.full(y_pred_arr[i].shape, int(y_true[i])))
    y_pred = tf.cast(tf.convert_to_tensor(np.concatenate(y_pred_arr, axis=0)), tf.int64)
    y_true = tf.cast(tf.convert_to_tensor(np.concatenate(y_true_arr, axis=0)), tf.int64)
    total = tf.shape(y_true)[0]
    correct = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.int32))
    print(f"correct:{correct}, total:{total}")
    acc = float(int(correct) / total)
    print(f"acc:{acc}")
    return acc
=======
version https://git-lfs.github.com/spec/v1
oid sha256:be2cd0bb4fef1d36871d5a73fcb1befe935b53e9553a4b62baf99ab4eaad8aa6
size 1399
>>>>>>> 9676c3e (ya toh aar ya toh par)
