import os
import shutil
import compress_json
from typing import Dict, Any, List
import json
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from sedna.common.class_factory import ClassFactory, ClassType
from ai2holodeck.generation.holodeck import Holodeck

class SceneData:
    def __init__(self, query=""):
        self.query = query

    @property
    def x(self):
        return self.query

@ClassFactory.register(ClassType.GENERAL, alias="HolodeckSceneGen")
class HolodeckGenerator:
    def __init__(self, **kwargs):
        print("Initializing HolodeckGenerator...")
        self.openai_api_key = kwargs.get("openai_api_key")
        self.objaverse_asset_dir = kwargs.get("objaverse_asset_dir")
        self.single_room = kwargs.get("single_room", False)
        self.data_path = kwargs.get("data_path")
        
        if not self.openai_api_key or not self.objaverse_asset_dir or not self.data_path:
            raise ValueError("Missing required arguments: 'openai_api_key', 'objaverse_asset_dir', and 'data_path'")

        self.holodeck_engine = Holodeck(
            openai_api_key=self.openai_api_key,
            openai_org=None,
            objaverse_asset_dir=self.objaverse_asset_dir,
            single_room=self.single_room,
        )

        self.generated_scene_path = None
        self.loaded_scene_data = None
        
        print("HolodeckGenerator initialized successfully.")


    def train(self, train_data, valid_data=None, **kwargs):
        print(f"Bypassing framework loader. Reading data from: {self.data_path}")
        queries = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_dict = json.loads(line)
                    queries.append(data_dict['answer'])

        print(f"Successfully loaded {len(queries)} queries.")

        last_generated_path = None

        for query in queries:
            
            print(f"Starting scene generation for query: '{query}'")
            
            output_dir = kwargs.get("output_dir", "./generated_scenes")
            os.makedirs(output_dir, exist_ok=True)
            
            scene_data, save_dir = self.holodeck_engine.generate_scene(
                scene=self.holodeck_engine.get_empty_scene(),
                query=query,
                save_dir=output_dir,
                generate_image=True,
                use_milp=False 
            )

            query_name = query.replace(" ", "_").replace("'", "")[:30]
            saved_file_path = None
            for root, _, files in os.walk(save_dir):
                for file in files:
                    if file.endswith(f"{query_name}.json"):
                        saved_file_path = os.path.join(root, file)
                        break
                if saved_file_path:
                    break
            
            if not saved_file_path:
                raise FileNotFoundError(f"Could not find generated scene file for query '{query}' in '{save_dir}'")
            
            last_generated_path = saved_file_path
            print(f"Scene for '{query}' generated successfully. Saved to: {saved_file_path}")

        self.generated_scene_path = last_generated_path
        return self.generated_scene_path

    def predict(self, data, **kwargs):
        if not self.generated_scene_path:
            print("Warning: predict called before train. No scene has been generated.")
            return None
        return self.generated_scene_path

    def save(self, model_path: str) -> str:
        if not self.generated_scene_path:
            raise RuntimeError("Save called before a scene was generated in train.")

        final_path = os.path.join(model_path, os.path.basename(self.generated_scene_path))
        os.makedirs(os.path.dirname(final_path), exist_ok=True)

        shutil.copy(self.generated_scene_path, final_path)
        print(f"Saved/Copied scene file to final destination: {final_path}")
        
        return final_path

    def load(self, model_url: str):
        print(f"Loading scene from: {model_url}")
        if not os.path.exists(model_url):
            raise FileNotFoundError(f"Scene file not found at {model_url}")
            
        self.loaded_scene_data = compress_json.load(model_url)
        print("Scene loaded successfully for evaluation.")
        
        return self

    def evaluate(self, data, **kwargs) -> Dict[str, Any]:
        if not self.loaded_scene_data:
            raise RuntimeError("Evaluate called before a scene was loaded.")
        
        metrics = kwargs.get("metrics")
        if not metrics:
            print("Warning: No metrics provided for evaluation.")
            return {}
            
        results = {}
        original_query = "a living room"

        for name, metric_func in metrics.items():
            if callable(metric_func):
                score = metric_func(self.loaded_scene_data, original_query)
                results[name] = score
            else:
                print(f"Warning: Metric '{name}' is not a callable function.")
        
        print(f"Evaluation completed. Results: {results}")
        return results