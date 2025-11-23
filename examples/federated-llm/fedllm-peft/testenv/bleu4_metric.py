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

"""Metric callback that computes BLEU-4 only."""
from evaluate import load as load_metric
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ["bleu4_metric"]

@ClassFactory.register(ClassType.GENERAL, alias="bleu4_metric")
def bleu4_metric(y_pred, y_true, **kwargs):
    preds = list(y_pred.values()) if isinstance(y_pred, dict) else list(y_pred)
    refs  = list(y_true.values()) if isinstance(y_true, dict) else list(y_true)
    bleu = load_metric("bleu")
    score = bleu.compute(predictions=preds, references=[[r] for r in refs])["bleu"]
    return round(float(score), 4)
=======
version https://git-lfs.github.com/spec/v1
oid sha256:5335008df58bc75f689dd9d969684913e30841278293583ca89cb9807e12f211
size 1184
>>>>>>> 9676c3e (ya toh aar ya toh par)
