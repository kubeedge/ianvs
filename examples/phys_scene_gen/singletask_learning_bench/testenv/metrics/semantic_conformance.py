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

import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sedna.common.class_factory import ClassFactory, ClassType
import compress_json

model = SentenceTransformer('all-MiniLM-L6-v2')

@ClassFactory.register(ClassType.GENERAL, alias="semantic_conformance")
def semantic_conformance(y_true, y_pred, **kwargs) -> float:
    try:
        scene_data = compress_json.load(y_pred)
    except Exception as e:
        # print(f"Error loading scene file at {y_pred}: {e}")
        return 0.0

    try:
        if not y_true or not y_true[0]:
            raise ValueError("Received empty or invalid y_true data from framework.")
        original_query = y_true[0]
    except (IndexError, ValueError) as e:
        # print(f"Error parsing original query from y_true data: {e}")
        return 0.0

    if not scene_data or not original_query:
        return 0.0

    scene_elements = set()
    if "rooms" in scene_data and scene_data["rooms"]:
        for room in scene_data["rooms"]:
            if "roomType" in room:
                room_type = room["roomType"].split('-')[0].replace('_', ' ')
                scene_elements.add(room_type)

    if "objects" in scene_data and scene_data["objects"]:
        for obj in scene_data["objects"]:
            if "object_name" in obj:
                object_name = obj["object_name"].split('-')[0].replace('_', ' ')
                scene_elements.add(object_name)

    if not scene_elements:
        # print("Warning: No rooms or objects found in the scene data to evaluate.")
        return 0.0

    scene_description = ", ".join(list(scene_elements))
    embedding_query = model.encode(original_query, convert_to_tensor=True)
    embedding_scene = model.encode(scene_description, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embedding_query, embedding_scene)
    normalized_score = (cosine_score.item() + 1) / 2

    return normalized_score
=======
version https://git-lfs.github.com/spec/v1
oid sha256:2ac54ce986181df35a86a54cfd76e299e8ec8eef6b5ee8880213af4e95ab6858
size 2522
>>>>>>> 9676c3e (ya toh aar ya toh par)
