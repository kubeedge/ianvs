<<<<<<< HEAD
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

import re
from sedna.common.class_factory import ClassType, ClassFactory
from core.common.log import LOGGER

__all__ = ["llm_judgement"]

def extract_comprehensive_score(input_str):
    # extract overall points
    match = re.search(r"'Overall Points': (\d+)", input_str)
    if match:
        return int(match.group(1))
    else:
        return None


@ClassFactory.register(ClassType.GENERAL, alias="llm_judgement")
def llm_judgement(y_true, y_pred):
    y_pred = [extract_comprehensive_score(pred) for pred in y_pred]
        
    valid_scores = [score for score in y_pred if score is not None]

    LOGGER.info(f"Extracted {len(valid_scores)} datas from {len(y_pred)} datas")
    
    if valid_scores:
        average_score = sum(valid_scores) / len(valid_scores)
        return average_score
    else:
        return -1
=======
version https://git-lfs.github.com/spec/v1
oid sha256:14ce624b21a740839324110d6a885df397dcb0ad4f2b5487cb9139e4fdb51af7
size 1411
>>>>>>> 9676c3e (ya toh aar ya toh par)
