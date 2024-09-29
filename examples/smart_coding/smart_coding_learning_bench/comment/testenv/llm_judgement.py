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
    # 使用正则表达式匹配综合得分及其分数
    match = re.search(r"'综合得分': (\d+)", input_str)
    if match:
        # 提取分数并返回
        return int(match.group(1))
    else:
        # 如果没有找到匹配项，返回None或其他适当的值
        return None


@ClassFactory.register(ClassType.GENERAL, alias="llm_judgement")
def llm_judgement(y_true, y_pred):
    y_pred = [extract_comprehensive_score(pred) for pred in y_pred]
        
    # 过滤掉None值（如果有）
    valid_scores = [score for score in y_pred if score is not None]

    LOGGER.info(f"Extracted {len(valid_scores)} datas from {len(y_pred)} datas")
    
    # 计算平均值
    if valid_scores:
        average_score = sum(valid_scores) / len(valid_scores)
        return average_score
    else:
        # 如果没有有效的分数，返回None或其他适当的值
        return -1
