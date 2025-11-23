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

import os
import compress_json
import compress_pickle
import numpy as np
import torch
import torch.nn.functional as F

from ai2holodeck.constants import (
    OBJATHOR_ANNOTATIONS_PATH,
    HOLODECK_THOR_ANNOTATIONS_PATH,
    OBJATHOR_FEATURES_DIR,
    HOLODECK_THOR_FEATURES_DIR,
)
from ai2holodeck.generation.utils import get_bbox_dims

class UnifiedRetriever:
    def __init__(
        self,
        clip_model,
        clip_preprocess,
        clip_tokenizer,
        sbert_model,
        retrieval_threshold,
        gapartnet_annotations_path,
        gapartnet_clip_features_path,
        gapartnet_sbert_features_path,
    ):
        # print("Loading annotations...")
        objathor_annotations = compress_json.load(OBJATHOR_ANNOTATIONS_PATH)
        thor_annotations = compress_json.load(HOLODECK_THOR_ANNOTATIONS_PATH)
        
        try:
            gapartnet_annotations = compress_json.load(gapartnet_annotations_path)
            # print(f"Successfully loaded {len(gapartnet_annotations)} GAPartNet annotations.")
        except FileNotFoundError:
            # print(f"Warning: GAPartNet annotations file not found at {gapartnet_annotations_path}. Interactive assets unavailable.")
            gapartnet_annotations = {}
            
        self.database = {**objathor_annotations, **thor_annotations, **gapartnet_annotations}
        # print(f"Total assets in combined database: {len(self.database)}")

        # print("Loading features...")
        objathor_clip_dict = compress_pickle.load(os.path.join(OBJATHOR_FEATURES_DIR, "clip_features.pkl"))
        objathor_sbert_dict = compress_pickle.load(os.path.join(OBJATHOR_FEATURES_DIR, "sbert_features.pkl"))
        assert objathor_clip_dict["uids"] == objathor_sbert_dict["uids"]
        objathor_uids = objathor_clip_dict["uids"]
        objathor_clip_features = objathor_clip_dict["img_features"].astype(np.float32)
        objathor_sbert_features = objathor_sbert_dict["text_features"].astype(np.float32)

        thor_clip_dict = compress_pickle.load(os.path.join(HOLODECK_THOR_FEATURES_DIR, "clip_features.pkl"))
        thor_sbert_dict = compress_pickle.load(os.path.join(HOLODECK_THOR_FEATURES_DIR, "sbert_features.pkl"))
        assert thor_clip_dict["uids"] == thor_sbert_dict["uids"]
        thor_uids = thor_clip_dict["uids"]
        thor_clip_features = thor_clip_dict["img_features"].astype(np.float32)
        thor_sbert_features = thor_sbert_dict["text_features"].astype(np.float32)

        try:
            gapartnet_clip_dict = compress_pickle.load(gapartnet_clip_features_path)
            gapartnet_sbert_dict = compress_pickle.load(gapartnet_sbert_features_path)
            assert gapartnet_clip_dict["uids"] == gapartnet_sbert_dict["uids"]
            gapartnet_uids = gapartnet_clip_dict["uids"]
            gapartnet_clip_features = gapartnet_clip_dict["img_features"].astype(np.float32)
            gapartnet_sbert_features = gapartnet_sbert_dict["text_features"].astype(np.float32)
            # print(f"Successfully loaded features for {len(gapartnet_uids)} GAPartNet assets.")
            
            expected_clip_shape = objathor_clip_features.shape[1:]
            if gapartnet_clip_features.shape[1:] != expected_clip_shape:
                 # print(f"Warning: GAPartNet CLIP features shape {gapartnet_clip_features.shape} "
                    #    f"differs from expected shape {(-1,) + expected_clip_shape}. Trying to reshape.")
                 if len(expected_clip_shape) == 2 and expected_clip_shape[0] == 1 and len(gapartnet_clip_features.shape) == 2:
                     gapartnet_clip_features = np.expand_dims(gapartnet_clip_features, axis=1)
                     # print(f"Reshaped GAPartNet CLIP features to {gapartnet_clip_features.shape}")
                 else:
                     # print(f"Error: Could not safely reshape GAPartNet CLIP features. Disabling them.")
                     gapartnet_clip_features = np.empty((0,) + expected_clip_shape, dtype=np.float32)

            expected_sbert_dim = objathor_sbert_features.shape[1]
            if gapartnet_sbert_features.shape[1] != expected_sbert_dim:
                 # print(f"Error: GAPartNet SBERT features dimension {gapartnet_sbert_features.shape[1]} "
                    #    f"differs from expected dimension {expected_sbert_dim}. Disabling GAPartNet SBERT.")
                 gapartnet_sbert_features = np.empty((0, expected_sbert_dim), dtype=np.float32)

        except FileNotFoundError:
            # print(f"Warning: GAPartNet feature files not found. GAPartNet assets will not be searchable.")
            gapartnet_uids = []
            clip_shape = objathor_clip_features.shape[1:]
            sbert_dim = objathor_sbert_features.shape[1]
            gapartnet_clip_features = np.empty((0,) + clip_shape, dtype=np.float32)
            gapartnet_sbert_features = np.empty((0, sbert_dim), dtype=np.float32)

        # print("Concatenating features and asset IDs...")
        self.clip_features = torch.from_numpy(
            np.concatenate([objathor_clip_features, thor_clip_features, gapartnet_clip_features], axis=0)
        )
        self.clip_features = F.normalize(self.clip_features, p=2, dim=-1)

        self.sbert_features = torch.from_numpy(
            np.concatenate([objathor_sbert_features, thor_sbert_features, gapartnet_sbert_features], axis=0)
        )

        self.asset_ids = objathor_uids + thor_uids + gapartnet_uids
        # print(f"Total searchable assets with features: {len(self.asset_ids)}")

        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer
        self.sbert_model = sbert_model
        self.retrieval_threshold = retrieval_threshold
        self.use_text = True

        # print("UnifiedRetriever initialized successfully.")


    def retrieve(self, queries, threshold=None, interactive_requested=False):
        if threshold is None:
             threshold = self.retrieval_threshold

        with torch.no_grad():
            query_feature_clip = self.clip_model.encode_text(self.clip_tokenizer(queries)).cpu()
            target_device_clip = self.clip_features.device
            query_feature_clip = query_feature_clip.to(target_device_clip)
            query_feature_clip = F.normalize(query_feature_clip, p=2, dim=-1)
            clip_similarities_all_views = 100 * torch.einsum("id, avd -> iav", query_feature_clip, self.clip_features)
            clip_similarities = torch.max(clip_similarities_all_views, dim=-1).values
            target_device_sbert = self.sbert_features.device
            query_feature_sbert = self.sbert_model.encode(queries, convert_to_tensor=True, show_progress_bar=False, device=target_device_sbert)
            sbert_features_tensor = self.sbert_features.to(query_feature_sbert.device)
            sbert_similarities = query_feature_sbert @ sbert_features_tensor.T
            similarities = clip_similarities.to(sbert_similarities.device) + sbert_similarities

        threshold_tensor = torch.tensor(threshold, device=clip_similarities.device)
        threshold_indices = torch.where(clip_similarities > threshold_tensor)

        unsorted_results_per_query = {} 
        for query_idx, asset_idx in zip(threshold_indices[0].tolist(), threshold_indices[1].tolist()):
            score = similarities[query_idx, asset_idx].item() 
            asset_id = self.asset_ids[asset_idx]
            if query_idx not in unsorted_results_per_query:
                unsorted_results_per_query[query_idx] = []
            unsorted_results_per_query[query_idx].append((asset_id, score))

        final_results_for_all_queries = []
        num_queries = len(queries)
        for query_idx in range(num_queries):
            query_results = unsorted_results_per_query.get(query_idx, [])
            sorted_results = sorted(query_results, key=lambda x: x[1], reverse=True)

            perfect_matches = []
            fallbacks = []
            perfect_candidates = []
            fallback_candidates = []
            for uid, score in sorted_results:
                metadata = self.database.get(uid, {})
                is_interactive = metadata.get("is_interactive", False)
                interaction_path = metadata.get("interaction_data_path")

                matches_requested_interactivity = (interactive_requested == is_interactive)

                if matches_requested_interactivity:
                    perfect_candidates.append((uid, score, interaction_path if is_interactive else None))
                else:
                    fallback_candidates.append((uid, score, interaction_path if is_interactive else None))

            if perfect_candidates:
                current_query_final_results = perfect_candidates
            elif fallback_candidates:
                fallbacks = fallback_candidates
                # print(f"[UnifiedRetriever Warning] Query '{queries[query_idx]}' requested interactive={interactive_requested}, only fallbacks found.")
                current_query_final_results = fallbacks
            else:
                current_query_final_results = []

            final_results_for_all_queries.extend(current_query_final_results)

        final_sorted_results = sorted(final_results_for_all_queries, key=lambda x: x[1], reverse=True)
        return final_sorted_results
    def compute_size_difference(self, target_size, candidates):
        if candidates and len(candidates[0]) == 3:
             candidates_tuples = [(uid, score) for uid, score, _ in candidates]
        else:
             candidates_tuples = candidates

        if not candidates_tuples:
             return []

        candidate_sizes = []
        valid_candidates_tuples = []
        for uid, score in candidates_tuples:
            if uid not in self.database:
                # print(f"[SizeDiff Warning] UID {uid} not found in database for size calculation.")
                continue
            try:
                size = get_bbox_dims(self.database[uid])
                if not all(k in size and isinstance(size[k], (int, float)) for k in ['x', 'y', 'z']):
                    # print(f"[SizeDiff Warning] Invalid bbox data for UID {uid}: {size}. Skipping.")
                    continue

                size_list = [size["x"] * 100, size["y"] * 100, size["z"] * 100]
                size_list.sort()
                candidate_sizes.append(size_list)
                valid_candidates_tuples.append((uid, score))
            except Exception as e:
                # print(f"[SizeDiff Error] Could not get/process bbox for UID {uid}: {e}. Skipping.")
                continue
        
        if not valid_candidates_tuples:
             return []

        candidate_sizes = torch.tensor(candidate_sizes, dtype=torch.float32)

        target_size_list = list(target_size)
        target_size_list.sort()
        target_size_tensor = torch.tensor(target_size_list, dtype=torch.float32)

        if target_size_tensor.shape != (3,):
             # print(f"[SizeDiff Error] Invalid target_size: {target_size}. Expected 3 dimensions.")
             return valid_candidates_tuples 
        
        size_difference = abs(candidate_sizes - target_size_tensor).mean(axis=1) / 100
        size_difference_list = size_difference.tolist()

        candidates_with_size_difference = []
        for i, (uid, score) in enumerate(valid_candidates_tuples):
            new_score = score - size_difference_list[i] * 10
            candidates_with_size_difference.append((uid, new_score))

        candidates_with_size_difference = sorted(
            candidates_with_size_difference, key=lambda x: x[1], reverse=True
        )

        return candidates_with_size_difference
=======
version https://git-lfs.github.com/spec/v1
oid sha256:cc03fd1618d72fdc7a1af5b25cc6f08adac7f1cf18429f681cd08aa4e5bd9956
size 12297
>>>>>>> 9676c3e (ya toh aar ya toh par)
