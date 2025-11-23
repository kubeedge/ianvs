<<<<<<< HEAD
# Copyright 2025 The KubeEdge Authors.
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

"""Metric callback that computes ROUGE-1 only."""
from evaluate import load as load_metric
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ["rouge1_metric"]

@ClassFactory.register(ClassType.GENERAL, alias="rouge1_metric")
def rouge1_metric(y_pred, y_true, **kwargs):
    # normalise inputs
    preds = list(y_pred.values()) if isinstance(y_pred, dict) else list(y_pred)
    refs  = list(y_true.values()) if isinstance(y_true, dict) else list(y_true)
    rouge = load_metric("rouge")
    score = rouge.compute(predictions=preds, references=refs, use_stemmer=True)["rouge1"]
    return round(float(score), 4)
=======
version https://git-lfs.github.com/spec/v1
oid sha256:0ac973d425de482e12eaa05216d670a3a55c0b330d6b7c20547154d59295583b
size 1220
>>>>>>> 9676c3e (ya toh aar ya toh par)
