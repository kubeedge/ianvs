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
from agumentation import Base_Augment


class Dataset_Preprocessor:
    def __init__(
        self,
        dataset_name: str,
        weak_augment_helper: Base_Augment,
        strong_augment_helper: Base_Augment,
    ) -> None:
        self.weak_augment_helper = weak_augment_helper
        self.strong_augment_helper = strong_augment_helper
        self.mean = 0.0
        self.std = 1.0
        if dataset_name == "cifar100":
            self.mean = np.array((0.5071, 0.4867, 0.4408), np.float32).reshape(1, 1, -1)
            self.std = np.array((0.2675, 0.2565, 0.2761), np.float32).reshape(1, 1, -1)
        print(f"mean: {self.mean}, std: {self.std}")

    def preprocess_labeled_dataset(self, x, y, batch_size):
        return (
            tf.data.Dataset.from_tensor_slices((x, y))
            .shuffle(100000)
            .map(
                lambda x, y: (
                    (tf.cast(x, dtype=tf.float32) / 255.0 - self.mean) / self.std,
                    tf.cast(y, dtype=tf.int32),
                )
            )
            .batch(batch_size)
        )

    def preprocess_unlabeled_dataset(self, ux, uy, batch_size):

        wux = self.weak_augment_helper(ux)
        sux = self.strong_augment_helper(ux)
        return (
            tf.data.Dataset.from_tensor_slices((ux, wux, sux, uy))
            .shuffle(100000)
            .map(
                lambda ux, wux, sux, uy: (
                    (tf.cast(ux, dtype=tf.float32) / 255.0 - self.mean) / self.std,
                    (tf.cast(wux, dtype=tf.float32) / 255.0 - self.mean) / self.std,
                    (tf.cast(sux, dtype=tf.float32) / 255.0 - self.mean) / self.std,
                    tf.cast(uy, dtype=tf.int32),
                )
            )
            .batch(batch_size)
        )
=======
version https://git-lfs.github.com/spec/v1
oid sha256:1216115359a953b6c62b4385e29da0ddf6f473eb81c8098817cd616b86a2cb1e
size 2408
>>>>>>> 9676c3e (ya toh aar ya toh par)
